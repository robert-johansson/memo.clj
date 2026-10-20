(ns memo.pygen
  "The transpiler: memo.clj source to Python memo source, test tooling
  (DESIGN.md §4, §9): the differential fuzzer runs the same programs in
  both, and the corpus's twins check the transpiler by round trip.

  An unparser: memo.parse's AST mirrors Python memo's node for node, so a
  parsed defmemo maps onto Python memo's syntax. Enum members are resolved
  to their values (as memo.clj's parser does), so Python sees numbers.
  Helper (defn ...) forms go through a small translator of scalar
  expressions into @jax.jit functions; domains become IntEnums, arrays and
  memo's domain(...).

    (transpile-forms forms opts) -> {:python str :warnings [...] :skipped #{name ...}}

  opts: :resolve-member, a function of [enum-sym member-sym] (memo.parse's),
  and :models-only? to emit the defmemos alone.

  Lifted from memo-clj's memo.pygen after review against Python memo 1.3.2's
  parse.py; changes: cast is emitted, enum members are values, more scalar
  forms translate (long, double, cond, ...)."
  (:require [clojure.string :as str]
            [clojure.walk]
            [memo.domains :as domains]
            [memo.parse :as parse]))

;; -----------------------------------------------------------------------------
;; identifiers

(defn sanitize
  "Clojure name -> Python identifier (hyphens etc. to underscores)."
  [sym]
  (-> (name sym)
      (str/replace "-" "_")
      (str/replace "?" "_qm")
      (str/replace "!" "_bang")
      (str/replace "*" "_star")
      (str/replace "'" "_prime")))

(defn- dotted->py [sym]
  (str/join "." (map sanitize (parse/dotted-parts sym))))

;; -----------------------------------------------------------------------------
;; scalar-expression translator (helper fn bodies + inline #{...} forms)

(declare py-scalar)

(def ^:private scalar-binops
  {'+ "+", '- "-", '* "*", '/ "/", 'mod "%",
   '= "==", '== "==", 'not= "!=", '!= "!=",
   '< "<", '<= "<=", '> ">", '>= ">=",
   'bit-and "&", 'bit-or "|", 'bit-xor "^", 'bit-shift-left "<<", 'bit-shift-right ">>",
   'floor-div "//", 'quot "//"})

(def ^:private array-fns
  "gen.array's operators (written array/f or gen.array/f) as jax.numpy's."
  {'exp "np.exp", 'log "np.log", 'abs "np.abs", 'sqrt "np.sqrt", 'floor "np.floor",
   'where "np.where", 'clip "np.clip", 'logical-and "np.logical_and", 'logical-or "np.logical_or",
   'logical-not "np.logical_not", 'sum "np.sum", 'array "np.array", 'maximum "np.maximum",
   'minimum "np.minimum"})

(defn- unqualify
  "array/f, gen.array/f and clojure.core/f as f; other symbols as they are."
  [h]
  (if (and (symbol? h) (#{"array" "gen.array" "clojure.core"} (namespace h)))
    (symbol (name h))
    h))

(def ^:private math-fns
  {'Math/exp "np.exp", 'Math/log "np.log", 'Math/sqrt "np.sqrt",
   'Math/pow "np.power", 'Math/cos "np.cos", 'Math/sin "np.sin",
   'Math/abs "np.abs", 'Math/floor "np.floor"})

(def ^:private math-constants {'Math/PI "np.pi", 'Math/E "np.e"})

(defn- py-num [x]
  (cond
    (integer? x) (str x)
    (and (number? x) (== x (Math/floor (double x)))
         (< (abs (double x)) 1e15))
    (let [d (double x)]
      (if (= d (double (long d))) (str (long d) ".0") (str d)))
    :else (str x)))

(defn py-scalar
  "Translate a restricted Clojure scalar expression to Python/jnp source.
  Throws ex-info {:pygen/unsupported form} on untranslatable forms."
  [form]
  (cond
    (number? form) (py-num form)
    (boolean? form) (if form "True" "False")
    (nil? form) "None"
    (contains? math-constants form) (math-constants form)
    (symbol? form) (if (parse/dotted-parts form)
                     (dotted->py form)
                     (sanitize form))
    (vector? form) (str "np.array([" (str/join ", " (map py-scalar form)) "])")
    (seq? form)
    (let [[h & args] (cons (unqualify (first form)) (rest form))]
      (cond
        ;; (let [a e ...] body): pure, so translated by substitution
        (= 'let h)
        (let [[bindings body] args]
          (when-not (and (vector? bindings) (every? symbol? (take-nth 2 bindings)) (= 2 (count args)))
            (throw (ex-info (str "cannot translate to Python: " (pr-str form)) {:pygen/unsupported form})))
          (py-scalar (reduce (fn [body [sym e]] (clojure.walk/postwalk-replace {sym e} body))
                             body
                             (reverse (partition 2 bindings)))))

        ;; (D :x z) a product domain's field, (E :MEMBER) an enum's member
        (and (symbol? h) (keyword? (first args)))
        (str (sanitize h) "." (sanitize (name (first args)))
             (when (next args) (str "(" (str/join ", " (map py-scalar (rest args))) ")")))

        (contains? array-fns h)
        (str (array-fns h) "(" (str/join ", " (map py-scalar args)) ")")
        (= 'negate h) (str "(-" (py-scalar (first args)) ")")
        (= 'square h) (str "(" (py-scalar (first args)) " ** 2)")
        (= 'take h) (str "np.asarray(" (py-scalar (first args)) ")[" (py-scalar (second args)) "]")
        (= 'astype h) (str "np.asarray(" (py-scalar (first args)) ").astype(np." (name (second args)) ")")
        (contains? scalar-binops h)
        (if (and (= '- h) (= 1 (count args)))
          (str "(-" (py-scalar (first args)) ")")
          (str "(" (str/join (str " " (scalar-binops h) " ")
                             (map py-scalar args)) ")"))

        (= 'if h)
        (let [[c t f] args]
          (str "np.where(" (py-scalar c) ", " (py-scalar t) ", " (py-scalar f) ")"))

        (= 'nth h)
        (let [[coll idx] args]
          (str (py-scalar coll) "[" (py-scalar idx) "]"))

        (= 'get-in h)
        (let [[coll path] args]
          (str "np.array(" (py-scalar coll) ")["
               (str/join ", " (map py-scalar path)) "]"))

        (= 'max h) (str "np.maximum(" (str/join ", " (map py-scalar args)) ")")
        (= 'min h) (str "np.minimum(" (str/join ", " (map py-scalar args)) ")")
        (= 'not h) (str "np.logical_not(" (py-scalar (first args)) ")")
        (= 'and h) (str "(" (str/join " & " (map py-scalar args)) ")")
        (= 'or h) (str "(" (str/join " | " (map py-scalar args)) ")")
        (= 'inc h) (str "(" (py-scalar (first args)) " + 1)")
        (= 'dec h) (str "(" (py-scalar (first args)) " - 1)")
        (= 'pos? h) (str "(" (py-scalar (first args)) " > 0)")
        (= 'neg? h) (str "(" (py-scalar (first args)) " < 0)")
        (= 'zero? h) (str "(" (py-scalar (first args)) " == 0)")
        (= 'vec h) (py-scalar (first args))
        (contains? '#{long double int float} h) (py-scalar (first args))
        (= 'quot h) (str "(" (py-scalar (first args)) " // " (py-scalar (second args)) ")")
        (= 'cond h)
        (let [pairs (partition 2 args)]
          (reduce (fn [else [c v]]
                    (if (= :else c)
                      (py-scalar v)
                      (str "np.where(" (py-scalar c) ", " (py-scalar v) ", " else ")")))
                  "0.0"
                  (reverse pairs)))
        (= 'range h) (str "np.arange(" (str/join ", " (map py-scalar args)) ")")
        (= 'linspace h) (str "np.linspace(" (str/join ", " (map py-scalar args)) ")")
        (contains? math-fns h)
        (str (math-fns h) "(" (str/join ", " (map py-scalar args)) ")")

        ;; clojure.core forms that a scalar helper cannot mean in Python
        (or (and (symbol? h) (str/ends-with? (name h) "?"))
            (contains? '#{case condp when when-not if-let when-let} h))
        (throw (ex-info (str "cannot translate to Python: " (pr-str form))
                        {:pygen/unsupported form}))

        (contains? '#{map filter remove reduce apply fn fn* let let* loop for
                      doseq do str println pr prn assoc dissoc conj first rest
                      next concat into partial comp mapv filterv update get}
                   h)
        (throw (ex-info (str "cannot translate to Python: " (pr-str form))
                        {:pygen/unsupported form}))

        ;; user helper call, or a defdomain field accessor like (Test.pi_i t) —
        ;; memo's Python `domain` exposes unpackers under the same names
        (symbol? h)
        (str (if (parse/dotted-parts h) (dotted->py h) (sanitize h))
             "(" (str/join ", " (map py-scalar args)) ")")

        :else (throw (ex-info (str "cannot translate to Python: " (pr-str form))
                              {:pygen/unsupported form}))))
    :else (throw (ex-info (str "cannot translate to Python: " (pr-str form))
                          {:pygen/unsupported form}))))

;; -----------------------------------------------------------------------------
;; memo expression emission (parsed AST -> Python memo syntax)

(declare emit-expr emit-stmts)

(def ^:private op->py
  {:add ["+" 5] :sub ["-" 5] :mul ["*" 6] :div ["/" 6]
   :pow ["**" 8] :mod ["%" 6]
   :eq ["==" 4] :neq ["!=" 4] :lt ["<" 4] :lte ["<=" 4] :gt [">" 4] :gte [">=" 4]
   :and ["and" 2] :or ["or" 2]})

(defn- wrap [s inner-prec ctx-prec]
  (if (< inner-prec ctx-prec) (str "(" s ")") s))

(defn- unfold-and
  "Unfold a left-folded :and spine into its conjuncts."
  [e]
  (if (and (= :op (:node e)) (= :and (:op e)))
    (conj (unfold-and (first (:args e))) (second (:args e)))
    [e]))

(defn- choice-ref->py [[_ sw si]]
  (if (= 'self sw) (sanitize si) (str (sanitize sw) "." (sanitize si))))

(defn- emit-call-args [e]
  (if (:ellipsis? e)
    "..."
    (str/join ", "
              (concat (map #(emit-expr % 0) (:args e))
                      (map (fn [[k v]] (str (sanitize k) "=" (emit-expr v 0)))
                           (:kwargs e))))))

(defn emit-expr
  "Parsed memo expr node -> Python memo expression string, parenthesized for
  a context of precedence `prec`."
  [e prec]
  (case (:node e)
    :lit (let [v (:value e)]
           (if (symbol? v) (sanitize v) (py-num v)))
    :choice (sanitize (:id e))
    :with (let [{:keys [who expr]} e]
            (if (= :choice (:node expr))
              (str (sanitize who) "." (sanitize (:id expr)))
              (str (sanitize who) "[" (emit-expr expr 0) "]")))
    :op
    (let [{:keys [op args]} e]
      (cond
        (contains? op->py op)
        (let [[s p] (op->py op)]
          (wrap (str (emit-expr (first args) p) " " s " "
                     (emit-expr (second args) (inc p)))
                p prec))
        ;; Python's ^ binds tighter than comparison, so operands are emitted
        ;; at a context above == (4); the whole term is parenthesized anyway.
        (= :xor op) (str "(" (emit-expr (first args) 5) " ^ "
                         (emit-expr (second args) 5) ")")
        (= :ite op)
        (let [[c t f] args]
          (str "(" (emit-expr t 1) " if " (emit-expr c 1)
               " else " (emit-expr f 1) ")"))
        (= :neg op) (wrap (str "-" (emit-expr (first args) 7)) 7 prec)
        (= :uadd op) (wrap (str "+" (emit-expr (first args) 7)) 7 prec)
        (= :inv op) (wrap (str "not " (emit-expr (first args) 3)) 3 prec)
        (contains? #{:exp :log :abs} op)
        (str (name op) "(" (emit-expr (first args) 0) ")")
        :else (throw (ex-info (str "unknown op " op) {:e e}))))
    :expect
    (case (:surface e)
      Pr (let [n (:surface-n e)
               conjuncts (unfold-and (:expr e))]
           (if (and n (= n (count conjuncts)))
             (str "Pr[" (str/join ", " (map #(emit-expr % 0) conjuncts)) "]")
             (str "Pr[" (emit-expr (:expr e) 0) "]")))
      Var (str "Var[" (emit-expr (:expr e) 0) "]")
      (str "E[" (emit-expr (:expr e) 0) "]"))
    :entropy (str "H[" (str/join ", " (map (fn [[w i]]
                                             (str (sanitize w) "." (sanitize i)))
                                           (:rvs e))) "]")
    :kl (str "KL[" (sanitize (:p-who e)) "." (sanitize (:p-id e)) " | "
             (sanitize (:q-who e)) "." (sanitize (:q-id e)) "]")
    :util (str "EU[" (sanitize (:goal e)) "]")
    :predict (str "Predict[" (emit-expr (:expr e) 0) "]")
    :imagine
    (let [stmts (emit-stmts (:do e) 2)
          then (emit-expr (:then e) 0)]
      (if (empty? (:do e))
        (str "imagine[" then "]")
        (str "imagine[\n"
             (str/join ",\n" (map #(str "        " %) stmts))
             ",\n        " then "\n    ]")))
    :inline (str "{" (py-scalar (:form e)) "}")
    :ffi (str (sanitize (:name e)) "(" (emit-call-args e) ")")
    :memo (str (sanitize (:name e))
               (when (:which-retval e) (str "[" (:which-retval e) "]"))
               "[" (str/join ", " (map choice-ref->py (:ids e))) "]"
               "(" (emit-call-args e) ")")
    (throw (ex-info (str "unknown expr node " (:node e)) {:e e}))))

;; -----------------------------------------------------------------------------
;; statement emission with regrouping

(defn- group-stmts
  "Group consecutive :with-stmt (same agent) into thinks-blocks and
  consecutive :knows (same agent) into knows-groups."
  [stmts]
  (let [runs (partition-by (fn [s]
                             (case (:node s)
                               :with-stmt [:thinks (:who s)]
                               :knows [:knows (:who s)]
                               [:single (gensym)]))
                           stmts)]
    (mapcat (fn [run]
              (case (:node (first run))
                :with-stmt [{:node ::thinks
                             :who (:who (first run))
                             :stmts (group-stmts (mapv :stmt run))}]
                :knows [{:node ::knows-group
                         :who (:who (first run))
                         :refs (mapv (fn [s]
                                       (if (= 'self (:source-who s))
                                         (sanitize (:source-id s))
                                         (str (sanitize (:source-who s)) "."
                                              (sanitize (:source-id s)))))
                                     run)}]
                run))
            runs)))

(defn- variant->py
  "memo-clj's :no-thinks is Python memo's Variant.NO_THINKS."
  [v]
  (str/upper-case (str/replace (name v) "-" "_")))

(defn- spec->py [s]
  (case (:surface-spec s)
    :uniformly "uniformly"
    :wpp (str "wpp=" (emit-expr (:wpp s) 0))
    :such-that (str "such_that=" (emit-expr (:wpp s) 0))
    ;; parse desugared :to-be into an :and spine of equalities; emit the surface expression
    :to-be (str "to_be=" (emit-expr (:spec-expr s) 0))
    :to-maximize (str "to_maximize=" (emit-expr (:wpp s) 0))
    :to-minimize
    ;; parse wrapped the expression in :neg for compilation; unwrap it
    (let [w (:wpp s)
          inner (if (and (= :op (:node w)) (= :neg (:op w)))
                  (first (:args w))
                  w)]
      (str "to_minimize=" (emit-expr inner 0)))))

(defn- emit-stmt [s indent]
  (let [pad (apply str (repeat indent " "))]
    (case (:node s)
      ::thinks
      (let [inner (emit-stmts (:stmts s) (+ indent 4))]
        (str (sanitize (:who s)) ": thinks[\n"
             (str/join ",\n" (map #(str pad "    " %) inner))
             "\n" pad "]"))
      ::knows-group
      (str (sanitize (:who s)) ": knows(" (str/join ", " (:refs s)) ")")
      :choose
      (str (sanitize (:who s)) ": "
           (name (or (:surface-alias s) :chooses))
           "(" (str/join ", " (map (fn [[id dom]]
                                     (str (sanitize id) " in " (sanitize dom)))
                                   (:choices s)))
           ", " (spec->py s) ")")
      :show
      (str (sanitize (:who s)) ": " (name (or (:surface-alias s) :observes))
           " [" (sanitize (:target-who s)) "." (sanitize (:target-id s)) "] is "
           (if (= 'self (:source-who s))
             (sanitize (:source-id s))
             (str (sanitize (:source-who s)) "." (sanitize (:source-id s)))))
      :observes-factor
      (if (= :boolean (:how s))
        (str (sanitize (:who s)) ": observes_that [" (emit-expr (:what s) 0) "]")
        (str (sanitize (:who s)) ": observes_event(wpp=" (emit-expr (:what s) 0) ")"))
      :knows
      (str (sanitize (:who s)) ": knows("
           (if (= 'self (:source-who s))
             (sanitize (:source-id s))
             (str (sanitize (:source-who s)) "." (sanitize (:source-id s)))) ")")
      :wants
      (str (sanitize (:who s)) ": wants(" (sanitize (:what s)) "="
           (emit-expr (:how s) 0) ")")
      :snapshot
      (str (sanitize (:who s)) ": snapshots_self_as(" (sanitize (:alias s)) ")")
      :variant
      (str (sanitize (:who s)) ": has_theory(" (variant->py (:variant s)) ")")
      :trace (str (sanitize (:who s)) ": INSPECT()")
      :forgets-about
      (str (sanitize (:who s)) ": forgets_about("
           (str/join ", " (map sanitize (:ids s))) ")")
      :forall (str "forall: " (sanitize (:id s)) " in " (sanitize (:domain s)))
      (throw (ex-info (str "unknown stmt node " (:node s)) {:s s})))))

(defn- emit-stmts [stmts indent]
  (mapv #(emit-stmt % indent) (group-stmts stmts)))

;; -----------------------------------------------------------------------------
;; whole defmemo emission

(defn- params->py [params]
  (str/join ", "
            (map (fn [p]
                   (str (sanitize (:name p))
                        (when (:exotic p) ": ...")
                        (when (some? (:default p))
                          (str "=" (py-scalar (:default p))))))
                 params)))

(defn member-resolver
  "memo.parse's :resolve-member for the enums of namespace ns (and the
  predefined Bool)."
  [ns]
  (fn [head member]
    (let [v (or (some-> (try (ns-resolve ns head) (catch Exception _ nil)) deref)
                (some-> (domains/builtins head) deref))]
      (when (domains/enum? v)
        (domains/member-value v member)))))

(defn emit-defmemo
  "One (defmemo ...) form as Python memo source. resolve-member resolves
  Enum.MEMBER (to its value)."
  ([form] (emit-defmemo form (constantly nil)))
  ([form resolve-member]
   (let [[_ mname & body] form
         [axes params & forms] body
         parsed (parse/parse-defmemo mname axes params (vec forms) {:resolve-member resolve-member})
         n-axis (count axes)
         body-stmts (vec (drop n-axis (:stmts parsed)))
         opts (select-keys (meta mname) [:cache :variant])
         dec-args (cond-> []
                    (:cache opts) (conj "cache=True")
                    (and (:variant opts) (not= :adult (:variant opts)))
                    (conj (str "variant=Variant." (variant->py (:variant opts)))))
         decorator (if (seq dec-args)
                     (str "@memo(" (str/join ", " dec-args) ")")
                     "@memo")
         axes-py (str/join ", " (map (fn [[a d]]
                                       (str (sanitize a) ": " (sanitize d)))
                                     axes))
         header (str decorator "\n"
                     "def " (sanitize mname)
                     ;; Python rejects an empty type-parameter list, so a memo
                     ;; with no output axes gets no brackets at all.
                     (when (seq axes) (str "[" axes-py "]"))
                     "(" (params->py (:params parsed)) "):")
         doc (when (:doc parsed)
               (str "    \"\"\"" (str/replace (:doc parsed) "\"" "\\\"") "\"\"\""))
         cast (when-let [c (:cast (:pctxt parsed))]
                (str "    cast: [" (str/join ", " (map sanitize (sort c))) "]"))
         stmt-lines (map #(str "    " %) (emit-stmts body-stmts 4))
         ret-lines (map #(str "    return " (emit-expr % 0)) (:retvals parsed))]
     (str/join "\n" (concat [header]
                            (when doc [doc])
                            (when cast [cast])
                            stmt-lines
                            ret-lines)))))

;; -----------------------------------------------------------------------------
;; values: domains, enums and constants, from a live namespace

(defn- py-array [v]
  (if (sequential? v)
    (str "[" (str/join ", " (map py-array v)) "]")
    (py-num v)))

(defn emit-value
  "name = value, for a value memo.clj's models use: a number, a vector of
  numbers (nested or not), an enum, a product domain."
  [dname v]
  (let [n (sanitize dname)]
    (cond
      (number? v) (str n " = " (py-num v))
      (domains/enum? v)
      (if (not= (name (:name v)) (name dname))
        (str n " = " (sanitize (:name v)))
        (str "class " n "(IntEnum):\n"
             (str/join "\n" (map (fn [m val] (str "    " (sanitize m) " = " val))
                                 (:members v) (:values v)))))
      (domains/product? v)
      (str n " = domain(" (str/join ", " (map (fn [k] (str (sanitize (name k)) "=" (get (:sizes v) k)))
                                               (:keys v)))
           ")")
      (and (sequential? v) (seq v) (every? #(or (number? %) (sequential? %)) (flatten v)))
      (str n " = np.array(" (py-array v) ")")
      :else (throw (ex-info (str "cannot transpile the value of " dname) {:pygen/unsupported dname})))))

(defn- emit-defn [[_ fname & more :as form]]
  (let [more (if (string? (first more)) (rest more) more)
        more (if (map? (first more)) (rest more) more)
        [argv & body] more
        _ (when-not (and (vector? argv) (every? symbol? argv) (not-any? #{'&} argv))
            (throw (ex-info "helper fns must take plain positional parameters"
                            {:pygen/unsupported form})))
        _ (when (not= 1 (count body))
            (throw (ex-info "helper fns must have a single body expression"
                            {:pygen/unsupported form})))]
    (str "@jax.jit\ndef " (sanitize fname) "(" (str/join ", " (map sanitize argv)) "):\n"
         "    return " (py-scalar (first body)))))

;; -----------------------------------------------------------------------------
;; file-level transpilation

(def ^:private preamble
  (str "from enum import IntEnum\n"
       "import jax\n"
       "import jax.numpy as np\n"
       "from memo import memo, domain, Variant\n"
       "from memo.lib import Bool\n"))

(defn- guarded
  "A model's source inside try, so that its compile error is recorded in
  _compile_errors and the module goes on."
  [mname src]
  (str "try:\n"
       (str/join "\n" (map #(str "    " %) (str/split-lines src)))
       "\nexcept Exception as _e:\n"
       "    _compile_errors['" (sanitize mname) "'] = _message(_e)"))

(defn transpile-forms
  "Top-level memo.clj forms as a Python memo module: defmemos, defns (as
  @jax.jit helpers) and the values of defs, defenums and defdomains, read
  from namespace ns (where the forms have been evaluated). A form that
  cannot be transpiled is skipped with a warning. Returns {:python
  :warnings :skipped}, :skipped the names of what was skipped. With
  :guarded? each model is defined inside try (see guarded), for
  memo.python/run."
  ([forms ns] (transpile-forms forms ns {}))
  ([forms ns {:keys [guarded?]}]
  (let [resolve-member (member-resolver ns)
        out (reduce
             (fn [acc form]
               (let [h (when (and (seq? form) (symbol? (first form))) (name (first form)))
                     dname (when (seq? form) (second form))]
                 (try
                   (case h
                     "defmemo" (update acc :chunks conj
                                       (cond->> (emit-defmemo form resolve-member)
                                         guarded? (guarded dname)))
                     ("def" "defenum" "defdomain")
                     (update acc :chunks conj
                             (emit-value dname (some-> (ns-resolve ns dname) deref)))
                     "defn" (update acc :chunks conj (emit-defn form))
                     acc)
                   (catch Exception e
                     (-> acc
                         (update :warnings conj (str "skipped " h " " dname ": " (ex-message e)))
                         (update :skipped conj dname))))))
             {:chunks [] :warnings [] :skipped #{}}
             forms)]
    {:python (str preamble
                  (when guarded?
                    (str "from memo.core import MemoError\n"
                         "_compile_errors = {}\n"
                         "def _message(e):\n"
                         "    return e.message if isinstance(e, MemoError) else f'{type(e).__name__}: {e}'\n"))
                  "\n\n" (str/join "\n\n" (:chunks out)) "\n")
     :warnings (:warnings out)
     :skipped (:skipped out)})))
