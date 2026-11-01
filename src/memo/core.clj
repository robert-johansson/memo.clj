(ns memo.core
  "memo.clj's public API: defmemo, and calling a memo.

  (defenum Coin HEADS TAILS)

  (defmemo flip [[c Coin]] [p]
    \"A biased coin.\"
    (alice :chooses [c Coin] :wpp (if (= c Coin.HEADS) p (- 1 p)))
    (return (Pr (= alice.c c))))

  (flip 0.3)           ; a gen.array array of shape [2]: the table over c
  (flip {:p 0.3})      ; parameters by name, in a trailing map

  The trailing map also takes a call's options, namespaced: :memo/budget,
  bounded minds (M3, experimental; memo.bounded), with :memo/limit and
  :memo/key.

  Parsing happens when the macro expands; analysis and lowering when the def
  is evaluated, so domains must be defined first (Python memo compiles when
  its decorator runs). A call returns the table over the memo's axes, in
  declaration order and at their full sizes; several returns are stacked on
  a leading axis. A memo without axes returns a 0-d value: on the JVM backend,
  a number."
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [gen.array :as array]
            [gen.random :as rng]
            [memo.analyze :as analyze]
            [memo.ast :as ast]
            [memo.bounded :as bounded]
            [memo.cost :as cost]
            [memo.domains :as domains]
            [memo.errors :as err]
            [memo.ir :as ir]
            [memo.lower :as lower]
            [memo.parse :as parse]))

(defn- check-param
  "A parameter's value: a scalar, or for a parameter marked ^:array (Python
  memo's `p: ...`) an array or a Clojure vector, which can only be passed to
  host functions and memo calls."
  [mname {:keys [name exotic]} v]
  (if exotic
    (when-not (or (sequential? v) (and (array/array? v) (seq (array/shape v))))
      (err/raise (str "Parameter " name " was not an array, despite being annotated as ^:array "
                      name ".")
                 {:loc {:name mname}}))
    (cond
      (and (integer? v) (not (clojure.core/<= Long/MIN_VALUE v Long/MAX_VALUE)))
      (err/raise (str "Parameter " name " was " v ", an integer beyond 64 bits")
                 {:hint "Python memo cannot take it either (an OverflowError). Use a double."
                  :loc {:name mname}})

      (or (number? v) (boolean? v)) nil
      ;; Python memo's message, for an array (and None)
      (or (nil? v) (sequential? v) (array/array? v))
      (when-not (and (array/array? v) (empty? (array/shape v)))
        (err/raise (str "Parameter " name " was not a numeric scalar, but rather an array. "
                        "By default, all memo parameters must be scalars. Annotate this "
                        "parameter as ^:array " name " if you really did intend to pass in "
                        "a non-scalar.")
                   {:loc {:name mname}}))
      :else
      (err/raise (str "Parameter " name " was not a numeric scalar, but " (pr-str v)
                      ", a " (.getSimpleName (class v)) ".")
                 {:hint "A memo parameter is a number or a boolean (or, marked ^:array, an array)."
                  :loc {:name mname}})))
  v)

(defonce ^{:private true
           :doc "Bumped whenever a memo is (re)defined: a cached memo drops its table
  then, since a memo it calls may have changed (memo-clj's lesson)."}
  definitions (atom 0))

(defn- cached
  "call, memoized by its arguments when they are host numbers or booleans
  (so never for values being differentiated), its budget, and whether it
  returns masses, until any memo is redefined; never a Monte Carlo call (one
  with a key)."
  [call]
  (let [cache (atom {:epoch -1 :table {}})
        host? #(or (number? %) (boolean? %))]
    (fn [args kwargs opts]
      (if-not (and (every? host? args) (every? host? (vals kwargs)) (nil? (:key opts)))
        (call args kwargs opts)
        (let [epoch @definitions
              k [(vec args) kwargs (select-keys opts [:budget :limit :mass])]
              c (swap! cache #(if (= epoch (:epoch %)) % {:epoch epoch :table {}}))]
          (if-let [[_ v] (find (:table c) k)]
            v
            (let [v (call args kwargs opts)]
              (swap! cache #(if (= epoch (:epoch %)) (assoc-in % [:table k] v) %))
              v)))))))

(defn- bind-params
  "The parameter environment of a call: positional arguments, then keyword
  arguments (a map from names, as keywords or symbols), then defaults."
  [mname params defaults args kwargs]
  (when (> (count args) (count params))
    (err/raise (str "Too many arguments to memo " mname
                    " (it takes " (count params) ")")
               (when (some keyword? args)
                 {:hint (str "Outside a memo, named arguments go in a trailing map: (" mname
                             " ... {:p 0.5}), not (" mname " ... :p 0.5).")})))
  (doseq [k (keys kwargs)]
    (when-not (and (or (keyword? k) (symbol? k)) (nil? (namespace k)))
      (err/raise (str "memo " mname " takes named arguments by keywords or symbols without a "
                      "namespace, not " (pr-str k))
                 {:hint (str "(" mname " ... {:p 0.5}) or {'p 0.5}; keys under :memo/ are the call's "
                             "options.")})))
  (let [kwargs (into {} (map (fn [[k v]] [(symbol (name k)) v])) kwargs)]
    (doseq [k (keys kwargs)]
      (when-not (some #(= k (:name %)) params)
        (err/raise (str "memo " mname " has no parameter " k))))
    (into {}
          (map-indexed
           (fn [i {pname :name :as p}]
             (let [v (cond
                       (< i (count args))
                       (if (contains? kwargs pname)
                         (err/raise (str "Parameter " pname " of memo " mname
                                         " passed both positionally and by name"))
                         (nth args i))
                       (contains? kwargs pname) (get kwargs pname)
                       (contains? defaults pname) (get defaults pname)
                       :else (err/raise (str "Missing parameter " pname " in call to memo " mname)))]
               [pname (check-param mname p v)])))
          params)))

(defn- memo-resolver
  "From a memo's name to the memo, in namespace ns, when it is called (so a
  memo may call itself, or memos defined after it)."
  [ns]
  (fn [sym]
    (let [v (ns-resolve (the-ns ns) sym)
          f (some-> v deref)]
      (if (:memo.impl/call (meta f))
        f
        (err/raise (str "Unknown memo " sym)
                   {:hint (str "A memo call (" sym " [axes ...] args ...) needs " sym
                               " to be a defmemo, defined by the time it is called.")})))))

(defn- callees
  "The memos m calls (as memos or as host functions), resolved now."
  [m]
  (let [{:memo.impl/keys [compiled ns]} (meta m)]
    (into #{}
          (keep (fn [{:keys [i name]}]
                  (when (#{:call-memo :ffi} i)
                    (let [f (some-> (ns-resolve (the-ns ns) name) deref)]
                      (when (:memo.impl/call (meta f)) f)))))
          (:instrs compiled))))

(defn- reachable-agents
  "The agents of m and of the memos it calls, transitively: {:agents
  :snapshots} as memo.ast/agents gives them."
  [m]
  (loop [todo [m] seen #{} acc {:agents #{} :snapshots {}}]
    (if-let [f (peek todo)]
      (if (seen f)
        (recur (pop todo) seen acc)
        (recur (into (pop todo) (callees f)) (conj seen f)
               (merge-with into acc (:memo.impl/agents (meta f)))))
      acc)))

(defn- check-agents
  "Each agent a budget names is one of m's, or of a memo it calls."
  [m budget]
  (let [{:keys [agents snapshots]} (reachable-agents m)
        mname (:memo/name (meta m))]
    (doseq [a (keys budget)]
      (cond
        (contains? agents a) nil
        (contains? snapshots a)
        (err/raise (str "The budget names " a ", " (snapshots a) "'s snapshot")
                   {:hint (str "A snapshot belongs to the agent that took it, and has its budget: give "
                               (snapshots a) " the budget.")
                    :loc {:name mname}})
        :else
        (err/raise (str "The budget names " a ", who is not an agent of memo " mname)
                   {:hint (if (seq agents)
                            (str "Its agents (and those of the memos it calls) are "
                                 (str/join ", " (sort-by str agents)) ".")
                            "It has no agents.")
                    :loc {:name mname}})))))

(defn- check-effect
  "Each part of each agent's budget bounds something in m or a memo it calls:
  :samples a query the agent evaluates about what it is uncertain of,
  :options a choice it narrows. Otherwise the call would give the exact
  table, as if that part were not there."
  [m budget]
  (let [bounded (loop [todo [m] seen #{} acc #{}]
                  (if-let [f (peek todo)]
                    (if (seen f)
                      (recur (pop todo) seen acc)
                      (recur (into (pop todo) (callees f)) (conj seen f)
                             (into acc ((:memo.impl/bounded-parts (meta f)) budget))))
                    acc))
        mname (:memo/name (meta m))]
    (doseq [[a {:keys [samples options]}] budget]
      (when (and samples (not (bounded [a :samples])))
        (err/raise (str "The :samples of " a "'s budget bound nothing in memo " mname)
                   {:hint (str a " evaluates no query (E, Pr, EU, ...) about what it is uncertain of, "
                               "here or in the memos called, so its choices would be the exact ones. "
                               "To bound how it imagines what follows a choice, make that a choice in "
                               "its model and read it through a query: (" a " :thinks (world :chooses "
                               "...)), then (E ...) or (EU ...) in its choice's weights.")
                    :loc {:name mname}}))
      (when (and options (not (bounded [a :options])))
        (err/raise (str "The :options of " a "'s budget bound nothing in memo " mname)
                   {:hint (str ":options narrows a choice " a " makes by weight, :to-maximize, "
                               ":to-minimize or :uniformly, here or in the memos called; " a
                               " makes none.")
                    :loc {:name mname}})))))

(def ^:private call-keys #{:memo/budget :memo/limit :memo/key})

(defn- call-options
  "A call's trailing map split into the parameters by name and the call's
  options {:budget :limit :key} (nil when it has none), the budget
  normalized."
  [mname m]
  (let [mine (into {} (filter (fn [[k _]] (and (keyword? k) (= "memo" (namespace k))))) m)]
    (when-let [k (first (remove call-keys (keys mine)))]
      (err/raise (str "Unknown call option " k)
                 {:hint "A memo call's options are :memo/budget, :memo/limit and :memo/key."
                  :loc {:name mname}}))
    (when (and (contains? mine :memo/key) (not (rng/key? (:memo/key mine))))
      (err/raise (str ":memo/key is " (pr-str (:memo/key mine)) ", not a key")
                 {:hint "A key is made by gen.random/key, e.g. (gen.random/key 0)." :loc {:name mname}}))
    [(apply dissoc m (keys mine))
     (when (seq mine)
       (let [budget (bounded/normalize-budget mname (:memo/budget mine))
             limit (some->> (:memo/limit mine) (bounded/limit-value ":memo/limit"))]
         (when (and (bounded/monte-carlo? budget) (not (:memo/key mine)))
           (err/raise "A Monte Carlo budget needs a key: call with :memo/key"
                      {:hint (str "The budget declares :monte-carlo, so the call draws samples: "
                                  "(f args... {:memo/budget {...} :memo/key (gen.random/key 0)}).")
                       :loc {:name mname}}))
         (when budget
           {:budget budget :limit limit :key (:memo/key mine) :check-agents true})))]))

(defn- make-memo
  "Analyze and lower a parsed memo; the callable model. Called by defmemo's
  expansion, through its var (it is private).
  fns maps each host function's name to its var (nil for the memo itself,
  resolved when called); inlines are the #{...} forms as functions of the
  parameters."
  [{:keys [name ns parsed domains defaults opts fns inlines]}]
  (let [domain-values (into {} (map (fn [[k d]]
                                      (when (instance? clojure.lang.Var$Unbound d)
                                        (err/raise (str "Unknown domain " k)
                                                   {:hint (str k " is declared but has no value: its "
                                                               "definition failed, or is still to come.")
                                                    :loc {:name name}}))
                                      (when-not (domains/domain? d)
                                        (err/raise (str k " is not a domain: " (pr-str d))
                                                   {:hint (domains/not-a-domain-hint d)
                                                    :loc {:name name}}))
                                      (let [vs (domains/values d)]
                                        ;; Python memo the same: some queries
                                        ;; compare positions, others values
                                        (when (and (sequential? d) (not= (count vs) (count (set vs))))
                                          (err/warn-at {:name name} "Domain " k " has repeated values "
                                                       (pr-str vs) ": a choice's value then names "
                                                       "more than one member, and a query compares "
                                                       "them by value or by position as Python memo "
                                                       "does, so (Pr (= alice.x y)) and "
                                                       "(Pr (= alice.x 1)) can disagree."))
                                        [k vs])))
                            domains)
        callee (fn [callee]
                 (when (not= callee name)
                   (let [m (some-> (ns-resolve (the-ns ns) callee) deref meta)]
                     (when (:memo/axes m)
                       {:axes (count (:memo/axes m)) :params (:memo/params m)
                        :defaults (:memo.impl/defaults m) :num-results (:memo/num-results m)}))))
        params (:params parsed)
        ;; the program, exact (budget nil) or for a normalized budget (M3);
        ;; with mass, it also returns the masses its returns weigh by
        ;; (memo.gen/deterministic's check)
        compile-for (fn [budget limit report-only mass]
                      (let [compiled (analyze/analyze parsed domain-values
                                                      (assoc opts :callee callee :bounded budget :limit limit
                                                             :report-only report-only :mass mass))
                            problems (ir/check compiled)]
                        (when (seq problems)
                          (err/raise (str "memo.clj built an ill-formed program for " name)
                                     {:user false :hint (str/join "; " problems)}))
                        {:compiled compiled
                         :run (lower/program compiled {:resolve-memo (memo-resolver ns)
                                                       :name name
                                                       :domain-values domain-values
                                                       :fns fns
                                                       :inlines inlines
                                                       :param-order (mapv :name params)})}))
        {:keys [compiled run]} (compile-for nil nil false false)
        agents (ast/agents parsed)
        programs (atom {})
        other-program (fn [budget limit report-only mass]
                        (let [k [budget limit report-only mass]]
                          (or (get @programs k)
                              (let [p (compile-for budget limit report-only mass)]
                                (swap! programs assoc k p)
                                p))))
        call (cond-> (fn [args kwargs {:keys [budget limit mass] :as call-opts}]
                       (let [env (bind-params name params defaults args kwargs)]
                         (cond
                           budget ((:run (other-program budget limit false (boolean mass))) env call-opts)
                           mass ((:run (other-program nil nil false true)) env)
                           :else (run env))))
               (:cache opts) cached)
        ;; the model itself, for checking a budget's agents against the memos
        ;; it calls (once per budget, until any memo is redefined)
        self (promise)
        checked (atom {})
        check (fn [budget]
                (let [epoch @definitions]
                  (when-not (= epoch (get @checked budget))
                    (check-agents @self budget)
                    (check-effect @self budget)
                    (swap! checked assoc budget epoch))))
        _ (swap! definitions inc)
        ;; a call, with extra internal options (mass) merged into its own
        invoke (fn [args extra]
                 (try
                   (if (map? (last args))
                     (let [[kwargs call-opts] (call-options name (last args))]
                       (when (:check-agents call-opts) (check (:budget call-opts)))
                       (call (butlast args) kwargs (merge (dissoc call-opts :check-agents) extra)))
                     (call args {} extra))
                   ;; memo calls nest on the JVM's stack (Python memo's RecursionError)
                   (catch StackOverflowError _
                     (err/raise (str "Maximum recursion depth exceeded in memo " name)
                                {:hint (str "A recursive memo needs a base case: a condition on a parameter, "
                                            "as in (if (> t 0) (f [..] (- t 1)) 0), that stops its memo calls. "
                                            "A deep one needs a bigger stack: call it on a thread with one, "
                                            "(Thread. nil f \"memo\" (* 1024 1024 1024)), or warm a ^:cache "
                                            "memo in steps.")
                                 :loc {:name name}}))))]
    (let [m (with-meta
          (fn [& args] (invoke args nil))
          {:memo/name name
           :memo/doc (:doc parsed)
           :memo/params (mapv :name params)
           :memo/axes (:axes compiled)
           :memo/domain-names (:domains compiled)
           :memo/domains (mapv domains (:domains compiled))
           :memo/shape (:shape compiled)
           :memo/num-results (count (:results compiled))
           ;; internal (:memo.impl/): not part of the API
           :memo.impl/defaults (set (keys defaults))
           :memo.impl/call call
           ;; a call returning [table masses], the totals of the beliefs its
           ;; returns weigh by (memo.analyze's :mass), for memo.gen/deterministic
           :memo.impl/call-with-masses (fn [& args] (invoke args {:mass true}))
           :memo.impl/compiled compiled
           ;; size @'s measure (memo.cost): a property of the text alone
           :memo.impl/size (cost/size parsed)
           :memo.impl/ns ns
           :memo.impl/agents agents
           :memo.impl/compile (fn [budget limit report-only]
                           (other-program (bounded/normalize-budget name budget)
                                          (some->> limit (bounded/limit-value ":memo/limit"))
                                          (boolean report-only) false))
           :memo.impl/callees #(callees @self)
           ;; [agent part] for each part of a normalized budget bounding something here
           :memo.impl/bounded-parts #(:bounded-parts (:compiled (other-program % nil true false)))
           :memo.impl/check-budget (fn [budget] (some-> (bounded/normalize-budget name budget) check))})]
      (deliver self m)
      m)))

(defn- member-resolver
  "Resolves Enum.MEMBER when the macro expands, in namespace ns."
  [ns]
  (fn [head member]
    (let [v (or (try (ns-resolve ns head) (catch Exception _ nil))
                (domains/builtins head))]
      (when (and (var? v) (domains/enum? @v))
        (try (@v member)
             (catch clojure.lang.ExceptionInfo e
               (err/raise (ex-message e) {:hint (str "Its members are "
                                                     (str/join ", " (:members @v)) ".")})))))))

(defn- inline-members
  "An inline form with each Enum.MEMBER in it replaced by the member's value,
  as in memo expressions: Python memo's {Action.WAIT} is Python, where
  Action.WAIT is a value. Any other agent.choice-shaped symbol is an error
  (Clojure would read it as a class name); dotted symbols that name classes
  are left to Clojure. So is a bare name in hidden, the memo's axes and
  choices."
  [mname form resolve-member hidden]
  (walk/postwalk (fn [x]
                   (let [parts (when (and (symbol? x) (nil? (namespace x))) (parse/dotted-parts x))
                         v (when (= 2 (count parts)) (resolve-member (first parts) (second parts)))]
                     (cond
                       (some? v) v
                       (or (and (= 2 (count parts))
                                (not (class? (try (resolve x) (catch Exception _ nil)))))
                           (contains? hidden x))
                       (err/raise (str "Cannot reference " x " inside the inline escape hatch #{...}")
                                  {:hint (str "#{...} is Clojure, evaluated with the memo's parameters "
                                              "only: it cannot reference memo constructs (agents or "
                                              "their choices).")
                                   :loc {:name mname}})
                       :else x)))
                 form))

(defmacro defmemo
  "Define a memo.

  (defmemo name [[axis Domain] ...] [param (param default) ...]
    \"docstring\"
    statements...
    (return expr) ...)

  Options as metadata on the name: :variant (:adult, the default, :child,
  :no-thinks, :no-knows or :no-wants, the theory of the top level); :cache
  (memoize calls whose arguments are all host numbers or booleans, as
  Python memo's cache=True). Other metadata, such as ^:private, stays on
  the var.

  A parameter marked ^:array takes an array or a vector, passed whole to
  host functions and memo calls (Python memo's `p: ...`).

  The var holds the memo, a function: (name args... named), with the
  parameters positionally in order (those with defaults may be left out)
  and an optional trailing map of parameters by name and the call's
  options. It returns the table, a gen.array array over the axes (several
  returns stacked on a first axis); without axes a 0-d value (a number on
  the JVM backend), which array/item reads on every backend. The docstring is the var's :doc and the
  memo's :memo/doc. A model memo.clj cannot compile throws a memo error
  when the defmemo is evaluated."
  [mname & body]
  (let [opts (select-keys (meta mname) [:variant :cache])
        _ (when-let [v (:variant opts)]
            (when-not (contains? (set (vals parse/variants)) v)
              (err/raise "Unknown variant"
                         {:hint (str "`" (pr-str v) "` is not a recognized memo variant. The "
                                     "available variants are: :adult, :child, :no-thinks, "
                                     ":no-knows, :no-wants")
                          :loc {:name mname}})))
        [axes params & forms] body
        resolve-member (member-resolver *ns*)
        parsed (parse/parse-defmemo mname axes params (vec forms)
                                    {:resolve-member resolve-member})
        [parsed inline-forms] (ast/assign-inline-ids parsed)
        refs (ast/collect parsed)
        domain-syms (sort (into (:domain-names refs) (map second axes)))
        param-syms (mapv :name (:params parsed))
        ;; names #{...} cannot see: the axes and choices (not parameters)
        inline-hidden (into (set (map first axes))
                            (comp (keep #(when (and (map? %) (= :choose (:node %))) (:choices %)))
                                  cat (map first)
                                  (remove (set param-syms)))
                            (tree-seq coll? seq (:stmts parsed)))
        _ (doseq [d domain-syms]
            (when (some #{d} param-syms)
              ;; Python memo would take the parameter's value as the domain
              (err/raise (str "The name `" d "` is already used for a static parameter and "
                              "cannot be used as a domain.")
                         {:hint (str "Python memo would use the parameter's value as the domain "
                                     "(" d " = 10 gives the domain {10}); memo.clj reads domains "
                                     "from vars. Rename the parameter.")
                          :loc {:name mname}}))
            (when-not (or (contains? &env d) (resolve d) (domains/builtins d))
              (err/raise (str "Unknown domain " d)
                         {:hint (str "Domains must be defined (as vectors or with defenum) before "
                                     "the defmemo that uses them.")
                          :loc {:name mname}})))
        fn-ref (fn [f]
                 (let [v (resolve f)]
                   (cond
                     (and (var? v) (bound? v) (seq (:memo/axes (meta @v))))
                     (err/raise (str f " is a memo with axes, called without them")
                                {:hint (str "A memo with axes is called with a vector of choices "
                                            "for them: (" f " [x ...] args ...).")
                                 :loc {:name mname}})

                     (var? v) (list 'var (symbol (str (.ns ^clojure.lang.Var v))
                                                 (str (.sym ^clojure.lang.Var v))))
                     (= f mname) nil

                     (and (namespace f) (class? (try (resolve (symbol (namespace f))) (catch Exception _ nil))))
                     (err/raise (str "Unknown function " f)
                                {:hint (str f " is a Java method, not a function: wrap it in one "
                                            "defined before the memo, e.g. (defn " (name f) " [x] ("
                                            f " x)).")
                                 :loc {:name mname}})

                     :else (err/raise (str "Unknown function " f)
                                      {:hint (str "Host functions must be defined before the "
                                                  "defmemo that calls them. Or should this be "
                                                  "a memo call, (" f " [axes ...] args ...)?")
                                       :loc {:name mname}}))))
        defaults (into {} (keep (fn [p]
                                  (when (some? (:default p))
                                    [(list 'quote (:name p)) (:default p)])))
                       (:params parsed))
        _ (doseq [{:keys [name default]} (:params parsed)
                  :when (and (symbol? default) (not (some #{default} param-syms))
                             (not (contains? &env default)) (not (resolve default)))]
            ;; [[a b]] reads as a with the default b
            (err/raise (str "The default of parameter " name ", " default ", is not defined")
                       {:hint (str "A parameter is a name, with a default as (" name " 1.0); a "
                                   "default is a value, defined before the memo.")
                        :loc {:name mname}}))
        _ (doseq [p (:params parsed)
                  :let [used (some (set param-syms) (tree-seq coll? seq (:default p)))]
                  :when (and (some? (:default p)) used)]
            ;; a default is computed once, when the memo is defined
            (err/raise (str "The default of parameter " (:name p) " refers to the parameter " used)
                       {:hint (str "A default is a value computed when the memo is defined, "
                                   "outside it, so it cannot use another parameter. (Python memo: "
                                   "a NameError.)")
                        :loc {:name mname}}))]
    `(def ~(vary-meta mname merge
                      (cond-> {:arglists (list 'quote (let [ps (mapv (comp symbol name) param-syms)]
                                                        (list ps (conj ps 'named))))}
                        (:doc parsed) (assoc :doc (:doc parsed))))
       ((var make-memo) {:name '~mname
                   :ns '~(ns-name *ns*)
                   :parsed '~parsed
                   :opts '~opts
                   :defaults ~defaults
                   :domains ~(into {} (map (fn [d] [(list 'quote d)
                                                    (if (and (not (resolve d)) (domains/builtins d))
                                                      (symbol "memo.domains" (name d))
                                                      d)]))
                                   domain-syms)
                   :fns ~(into {} (map (fn [f] [(list 'quote f) (fn-ref f)])) (:ffi-names refs))
                   :inlines ~(mapv (fn [form] `(with-meta (fn [~@param-syms]
                                                            ~(inline-members mname form resolve-member
                                                                             inline-hidden))
                                                    {:memo.impl/form '~form}))
                                   inline-forms)}))))

(defn memo?
  "Whether x is a memo: the model a defmemo defines (not its var)."
  [x]
  (boolean (:memo.impl/call (meta x))))

(defn program
  "A memo's compiled IR program, as text (Python memo's
  debug_print_compiled). m is a memo or its var. The text is for reading:
  its format may change in any release."
  [m]
  (let [m (if (var? m) @m m)]
    (when-not (memo? m)
      (err/raise (str "program takes a memo (defmemo), not " (pr-str m))
                 {:hint "(program m), with m a defmemo's model."}))
    (ir/format-program (:memo.impl/compiled (meta m)))))

(defn print-table
  "Print a memo's result as a table, one row per cell of its axes (Python
  memo's print_table)."
  [m result]
  (let [m (if (var? m) @m m)
        {:memo/keys [axes domains domain-names num-results name shape]} (meta m)
        _ (when-not name
            (err/raise "print-table takes a memo and its result"
                       {:hint "(print-table m (m args...)), with m a defmemo's model."}))
        _ (when-not (or (number? result) (boolean? result) (array/array? result) (sequential? result))
            (err/raise (str "print-table: the result is " (pr-str result) ", not " name "'s table")
                       {:hint (str "Pass " name "'s own result: (print-table " name " (" name " ...)).")}))
        result (if (or (number? result) (boolean? result) (array/array? result))
                 result
                 (array/array result))
        want (if (= 1 num-results) shape (into [num-results] shape))
        _ (when (not= want (array/shape result))
            (err/raise (str "print-table: the result has shape " (array/shape result)
                            ", but " name "'s results have shape " want)
                       {:hint (str "Pass " name "'s own result: (print-table " name " (" name " ...)).")}))
        data (array/->clj (array/stop-gradient result))    ; printed: no gradient to lose
        cells (reduce (fn [acc d] (for [a acc i (range (count (domains/values d)))] (conj a i)))
                      [[]] domains)
        value (fn [path] (str (if (sequential? data) (get-in data path) data)))
        header (into (mapv (fn [a d] (str a ": " d)) axes domain-names)
                     (if (= 1 num-results)
                       [(str name)]
                       (map #(str name "[" % "]") (range num-results))))
        rows (for [cell cells]
               (into (mapv (fn [d i] (nth (domains/labels d) i)) domains cell)
                     (if (= 1 num-results)
                       [(value cell)]
                       (map #(value (into [%] cell)) (range num-results)))))
        widths (apply map (fn [& col] (apply max (map count col))) header rows)
        line (fn [cols] (str "| " (str/join " | " (map #(format (str "%-" %2 "s") %1) cols widths)) " |"))
        rule (str "+" (str/join "+" (map #(apply str (repeat (+ 2 %) "-")) widths)) "+")]
    (println rule)
    (println (line header))
    (println rule)
    (doseq [r rows] (println (line r)))
    (println rule)))
