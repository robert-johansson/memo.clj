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
            [gen.array :as array]
            [memo.analyze :as analyze]
            [memo.ast :as ast]
            [memo.domains :as domains]
            [memo.errors :as err]
            [memo.ir :as ir]
            [memo.lower :as lower]
            [memo.parse :as parse]))

(defn- check-param [mname {:keys [name exotic]} v]
  (when exotic
    (err/raise (str "Array parameters (^:array " name ") are not supported in memo.clj yet")))
  (when-not (and (or (number? v) (boolean? v) (array/array? v))
                 (empty? (array/shape v)))
    (err/raise (str "Parameter " name " was not a numeric scalar, but rather an array. "
                    "By default, all memo parameters must be scalars. Annotate this "
                    "parameter as ^:array " name " if you really did intend to pass in "
                    "a non-scalar.")
               {:loc {:name mname}}))
  v)

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
  "Analyze and lower a parsed memo; the callable model. Called by defmemo."
  [{:keys [name ns parsed domains defaults opts]}]
  (let [domain-values (into {} (map (fn [[k d]] [k (domains/values d)])) domains)
        compiled (analyze/analyze parsed domain-values opts)
        problems (ir/check compiled)
        _ (when (seq problems)
            (err/raise (str "memo.clj built an ill-formed program for " name)
                       {:user false :hint (str/join "; " problems)}))
        run (lower/program compiled {:resolve-memo (memo-resolver ns)})
        params (:params parsed)
        call (fn [args kwargs] (run (bind-params name params defaults args kwargs)))]
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
    (let [v (try (ns-resolve ns head) (catch Exception _ nil))]
      (when (and (var? v) (domains/enum? @v))
        (try (domains/member-value @v member)
             (catch clojure.lang.ExceptionInfo e
               (err/raise (ex-message e) {:hint (str "Its members are "
                                                     (str/join ", " (:members @v)) ".")})))))))

(defmacro defmemo
  "Define a memo.

  (defmemo name [[axis Domain] ...] [param (param default) ...]
    \"docstring\"
    statements...
    (return expr) ...)

  Options as metadata on the name: :variant (:child, :no-thinks, :no-knows
  or :no-wants, the theory of the top level)."
  [mname & body]
  (let [opts (select-keys (meta mname) [:variant])
        [axes params & forms] body
        parsed (parse/parse-defmemo mname axes params (vec forms)
                                    {:resolve-member (member-resolver *ns*)})
        domain-syms (sort (into (:domain-names (ast/collect parsed)) (map second axes)))
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
                   :domains ~(into {} (map (fn [d] [(list 'quote d) d])) domain-syms)}))))

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
