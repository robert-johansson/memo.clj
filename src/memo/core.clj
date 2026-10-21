(ns memo.core
  "memo.clj's public API: defmemo, and calling a memo.

  (defenum Coin HEADS TAILS)

  (defmemo flip [[c Coin]] [p]
    \"A biased coin.\"
    (alice :chooses [c Coin] :wpp (if (= c Coin.HEADS) p (- 1 p)))
    (return (Pr (= alice.c c))))

  (flip 0.3)           ; a gen.array array of shape [2]: the table over c
  (flip {:p 0.3})      ; parameters by name, in a trailing map

  Parsing happens when the macro expands; analysis and lowering when the def
  is evaluated, so domains must be defined first (Python memo compiles when
  its decorator runs). A call returns the table over the memo's axes, in
  declaration order and at their full sizes; several returns are stacked on
  a leading axis. A memo without axes returns a 0-d value: on the JVM backend,
  a number."
  (:require [clojure.string :as str]
            [clojure.walk :as walk]
            [gen.array :as array]
            [memo.analyze :as analyze]
            [memo.ast :as ast]
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

(defonce ^{:doc "Bumped whenever a memo is (re)defined: a cached memo drops its table
  then, since a memo it calls may have changed (memo-clj's lesson)."}
  definitions (atom 0))

(defn- cached
  "call, memoized by its arguments when they are host numbers or booleans
  (so never for values being differentiated), until any memo is redefined."
  [call]
  (let [cache (atom {:epoch -1 :table {}})
        host? #(or (number? %) (boolean? %))]
    (fn [args kwargs]
      (if-not (and (every? host? args) (every? host? (vals kwargs)))
        (call args kwargs)
        (let [epoch @definitions
              k [(vec args) kwargs]
              c (swap! cache #(if (= epoch (:epoch %)) % {:epoch epoch :table {}}))]
          (if-let [[_ v] (find (:table c) k)]
            v
            (let [v (call args kwargs)]
              (swap! cache #(if (= epoch (:epoch %)) (assoc-in % [:table k] v) %))
              v)))))))

(defn- bind-params
  "The parameter environment of a call: positional arguments, then keyword
  arguments (a map from names, as keywords or symbols), then defaults."
  [mname params defaults args kwargs]
  (when (> (count args) (count params))
    (err/raise (str "Too many arguments to memo " mname
                    " (it takes " (count params) ")")))
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
      (if (:memo/call (meta f))
        f
        (err/raise (str "Unknown memo " sym)
                   {:hint (str "A memo call (" sym " [axes ...] args ...) needs " sym
                               " to be a defmemo, defined by the time it is called.")})))))

(defn make-memo
  "Analyze and lower a parsed memo; the callable model. Called by defmemo.
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
                        :defaults (:memo/defaults m) :num-results (:memo/num-results m)}))))
        compiled (analyze/analyze parsed domain-values (assoc opts :callee callee))
        problems (ir/check compiled)
        _ (when (seq problems)
            (err/raise (str "memo.clj built an ill-formed program for " name)
                       {:user false :hint (str/join "; " problems)}))
        params (:params parsed)
        run (lower/program compiled {:resolve-memo (memo-resolver ns)
                                     :name name
                                     :domain-values domain-values
                                     :fns fns
                                     :inlines inlines
                                     :param-order (mapv :name params)})
        call (cond-> (fn [args kwargs] (run (bind-params name params defaults args kwargs)))
               (:cache opts) cached)
        _ (swap! definitions inc)]
    (with-meta
      (fn [& args]
        (try
          (if (map? (last args))
            (call (butlast args) (last args))
            (call args {}))
          ;; memo calls nest on the JVM's stack (Python memo's RecursionError)
          (catch StackOverflowError _
            (err/raise (str "Maximum recursion depth exceeded in memo " name)
                       {:hint (str "A recursive memo needs a base case: a condition on a parameter, "
                                   "as in (if (> t 0) (f [..] (- t 1)) 0), that stops its memo calls. "
                                   "A deep one needs a bigger stack: call it on a thread with one, "
                                   "(Thread. nil f \"memo\" (* 1024 1024 1024)), or warm a ^:cache "
                                   "memo in steps.")
                        :loc {:name name}}))))
      {:memo/name name
       :memo/doc (:doc parsed)
       :memo/params (mapv :name params)
       :memo/defaults (set (keys defaults))
       :memo/axes (:axes compiled)
       :memo/domain-names (:domains compiled)
       :memo/domains (mapv domains (:domains compiled))
       :memo/shape (:shape compiled)
       :memo/num-results (count (:results compiled))
       :memo/call call
       :memo/compiled compiled})))

(defn- member-resolver
  "Resolves Enum.MEMBER when the macro expands, in namespace ns."
  [ns]
  (fn [head member]
    (let [v (or (try (ns-resolve ns head) (catch Exception _ nil))
                (domains/builtins head))]
      (when (and (var? v) (domains/enum? @v))
        (try (domains/member-value @v member)
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

  Options as metadata on the name: :variant (:child, :no-thinks, :no-knows
  or :no-wants, the theory of the top level); :cache (memoize calls by their
  arguments, as Python memo's cache=True).

  A parameter marked ^:array takes an array or a vector, passed whole to
  host functions and memo calls (Python memo's `p: ...`)."
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
    `(def ~mname
       (make-memo {:name '~mname
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
                                                    {:memo/form '~form}))
                                   inline-forms)}))))

(defn program
  "A memo's compiled IR program, as text (Python memo's debug_print_compiled)."
  [m]
  (ir/format-program (:memo/compiled (meta m))))

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
        data (array/->clj result)
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
