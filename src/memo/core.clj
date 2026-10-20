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
    (when-not (and (or (number? v) (boolean? v) (array/array? v))
                   (empty? (array/shape v)))
      (err/raise (str "Parameter " name " was not a numeric scalar, but rather an array. "
                      "By default, all memo parameters must be scalars. Annotate this "
                      "parameter as ^:array " name " if you really did intend to pass in "
                      "a non-scalar.")
                 {:loc {:name mname}})))
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
  (let [domain-values (into {} (map (fn [[k d]] [k (domains/values d)])) domains)
        compiled (analyze/analyze parsed domain-values opts)
        problems (ir/check compiled)
        _ (when (seq problems)
            (err/raise (str "memo.clj built an ill-formed program for " name)
                       {:user false :hint (str/join "; " problems)}))
        params (:params parsed)
        run (lower/program compiled {:resolve-memo (memo-resolver ns)
                                     :fns fns
                                     :inlines inlines
                                     :param-order (mapv :name params)})
        call (cond-> (fn [args kwargs] (run (bind-params name params defaults args kwargs)))
               (:cache opts) cached)
        _ (swap! definitions inc)]
    (with-meta
      (fn [& args]
        (if (map? (last args))
          (call (butlast args) (last args))
          (call args {})))
      {:memo/name name
       :memo/doc (:doc parsed)
       :memo/params (mapv :name params)
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
  are left to Clojure."
  [mname form resolve-member]
  (walk/postwalk (fn [x]
                   (let [parts (when (and (symbol? x) (nil? (namespace x))) (parse/dotted-parts x))
                         v (when (= 2 (count parts)) (resolve-member (first parts) (second parts)))]
                     (cond
                       (some? v) v
                       (and (= 2 (count parts))
                            (not (class? (try (resolve x) (catch Exception _ nil)))))
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
        [axes params & forms] body
        resolve-member (member-resolver *ns*)
        parsed (parse/parse-defmemo mname axes params (vec forms)
                                    {:resolve-member resolve-member})
        [parsed inline-forms] (ast/assign-inline-ids parsed)
        refs (ast/collect parsed)
        domain-syms (sort (into (:domain-names refs) (map second axes)))
        param-syms (mapv :name (:params parsed))
        fn-ref (fn [f]
                 (let [v (resolve f)]
                   (cond
                     (var? v) (list 'var (symbol (str (.ns ^clojure.lang.Var v))
                                                 (str (.sym ^clojure.lang.Var v))))
                     (= f mname) nil
                     :else (err/raise (str "Unknown function " f)
                                      {:hint (str "Host functions must be defined before the "
                                                  "defmemo that calls them. Or should this be "
                                                  "a memo call, (" f " [axes ...] args ...)?")
                                       :loc {:name mname}}))))
        defaults (into {} (keep (fn [p]
                                  (when (some? (:default p))
                                    [(list 'quote (:name p)) (:default p)])))
                       (:params parsed))]
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
                   :inlines ~(mapv (fn [form] `(fn [~@param-syms] ~(inline-members mname form resolve-member)))
                                   inline-forms)}))))

(defn program
  "A memo's compiled IR program, as text (Python memo's debug_print_compiled)."
  [m]
  (ir/format-program (:memo/compiled (meta m))))

(defn print-table
  "Print a memo's result as a table, one row per cell of its axes (Python
  memo's print_table)."
  [m result]
  (let [{:memo/keys [axes domains domain-names num-results name]} (meta m)
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
