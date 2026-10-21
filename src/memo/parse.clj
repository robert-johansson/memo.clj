(ns memo.parse
  "DSL forms -> memo AST. Port of parse.py for the agent-first Clojure syntax.

  Statements:  (agent :chooses [x X] :wpp e) (agent :thinks (other ...) ...)
               (agent :observes other.x :is y) (agent :knows x other.y)
               (agent :observes-that e) (agent :observes-event :wpp e)
               (agent :wants goal e) (agent :snapshots-self-as past)
               (agent :has-theory :child) (agent :inspect) (forall [x X])
  Expressions: numbers, params, choices, dotted refs (a.x), (as a e), ops,
               (E e) (Pr e ...) (Var e) (H a.x ...) (KL a.x b.y) (EU g)
               (Predict e) (imagine (a ...) ... e), FFI calls (f a b),
               memo calls (f [x a.y] arg ...), inline escape hatch #{form}.

  Lifted from memo-clj after review against parse.py, with two changes
  (DESIGN.md §8): an enum member written `Utt.SOME` is resolved to its value
  here, through the :resolve-member function in the parse context, so no var
  needs a dot in its name; and `self` cannot name an agent (Python memo 1.3.3
  accepts it, and its choices then collide with the frame's own)."
  (:require [clojure.string :as str]
            [memo.errors :as err]))

;; -----------------------------------------------------------------------------
;; helpers

(defn loc-of [form pctxt]
  (let [m (meta form)]
    {:line (:line m) :name (:name pctxt)}))

(defn dotted-parts
  "Split a dotted symbol like speaker.u into [speaker u ...]; nil when not
  dotted (including the literal ... symbol)."
  [sym]
  (when (symbol? sym)
    (let [s (name sym)]
      (when (and (str/includes? s ".")
                 (not (str/starts-with? s "."))
                 (not (str/ends-with? s ".")))
        (let [parts (str/split s #"\.")]
          (when (every? seq parts)
            (mapv symbol parts)))))))

(def ellipsis-sym (symbol "..."))

(defn- check-cast [who pctxt loc]
  (when (and (:cast pctxt)
             (not= who 'self)
             (not (contains? (:cast pctxt) who)))
    (err/raise (str "agent `" who "` is not in the cast")
               {:hint (str "Did you either misspell `" who "`, or forget to "
                           "include `" who "` in the cast?")
                :loc loc})))

(def ^:private fold-ops '{+ :add, * :mul, - :sub, / :div})
(def ^:private binary-only-ops
  '{** :pow, pow :pow, % :mod, mod :mod,
    = :eq, == :eq, not= :neq, != :neq,
    < :lt, <= :lte, > :gt, >= :gte, xor :xor})
(def ^:private bool-ops '{and :and, or :or})
(def ^:private unary-fn-ops '{exp :exp, log :log, abs :abs})

(def reserved-heads
  (into #{'if 'not 'E 'Pr 'Var 'H 'KL 'EU 'Predict 'imagine 'as
          'return 'forall 'cast}
        (concat (keys fold-ops) (keys binary-only-ops) (keys bool-ops)
                (keys unary-fn-ops))))

;; -----------------------------------------------------------------------------
;; expressions

(declare parse-expr parse-stmt-form)

(defn- op-node [op args loc & {:keys [static] :or {static ::from-args}}]
  {:node :op :op op :args (vec args) :loc loc
   :static (if (= static ::from-args)
             (every? :static args)
             static)})

(defn- parse-args-list
  "Port of parse_args_list: `...` expands to all static params; bare exotic
  param names pass through as static literals."
  [args pctxt loc]
  (if (and (= 1 (count args)) (= ellipsis-sym (first args)))
    (mapv (fn [p] {:node :lit :value p :loc loc :static true})
          (:static-parameters pctxt))
    (mapv (fn [a]
            (if (and (symbol? a) (contains? (:exotic-parameters pctxt) a))
              {:node :lit :value a :loc loc :static true}
              (parse-expr a pctxt)))
          args)))

(defn- split-kwargs
  "Split an arg seq into [positional kwarg-pairs]; kwargs are trailing
  :key val pairs."
  [args loc]
  (let [pos (vec (take-while #(not (keyword? %)) args))
        kw (drop-while #(not (keyword? %)) args)]
    (when (odd? (count kw))
      (err/raise "Malformed keyword arguments in call" {:loc loc}))
    (doseq [[k _] (partition 2 kw)]
      (when-not (keyword? k)
        (err/raise "Malformed keyword arguments in call"
                   {:hint "Expected alternating :key value pairs after positional arguments."
                    :loc loc})))
    [pos (mapv (fn [[k v]] [(symbol (name k)) v]) (partition 2 kw))]))

(defn- parse-kwargs [kw-pairs pctxt loc]
  (mapv (fn [[k v]]
          [k (if (and (symbol? v) (contains? (:exotic-parameters pctxt) v))
               {:node :lit :value v :loc loc :static true}
               (parse-expr v pctxt))])
        kw-pairs))

(defn- axis-ref [a pctxt loc]
  (cond
    (dotted-parts a)
    (let [[who & ids] (dotted-parts a)]
      (when (not= 1 (count ids))
        (err/raise "Invalid axis specification in memo call" {:loc loc}))
      [ellipsis-sym who (first ids)])

    (symbol? a) [ellipsis-sym 'self a]

    :else
    (err/raise "Invalid axis specification in memo call"
               {:hint (str "Each axis passed to the memo model you are calling must "
                           "be either the name of a choice (e.g. x) or another "
                           "agent's choice (e.g. alice.y). Nothing else is allowed.")
                :loc loc})))

(defn- parse-call [form pctxt loc]
  (let [[head & args] form
        [which-retval args] (if (and (int? (first args)) (vector? (second args)))
                              [(first args) (rest args)]
                              [nil args])]
    (if (vector? (first args))
      ;; memo call: (f [x a.y] args... :kw v...)
      (let [ids (mapv #(axis-ref % pctxt loc) (first args))
            [pos kw] (split-kwargs (rest args) loc)
            ellipsis? (and (= 1 (count pos)) (= ellipsis-sym (first pos)))]
        {:node :memo :name head :which-retval which-retval :ids ids
         :args (parse-args-list pos pctxt loc)
         :kwargs (parse-kwargs kw pctxt loc)
         :ellipsis? ellipsis?
         :loc loc :static false})
      ;; FFI call: (f args... :kw v...)
      (let [_ (when (some? which-retval)
                (err/raise "Unknown expression syntax" {:loc loc}))
            [pos kw] (split-kwargs args loc)
            ellipsis? (and (= 1 (count pos)) (= ellipsis-sym (first pos)))
            args* (parse-args-list pos pctxt loc)
            kwargs* (parse-kwargs kw pctxt loc)]
        {:node :ffi :name head
         :args args* :kwargs kwargs*
         :ellipsis? ellipsis?
         :static (and (every? :static args*)
                      (every? (comp :static second) kwargs*))
         :loc loc}))))

(defn- entropy-ref [a pctxt loc]
  (if-let [[who & ids] (dotted-parts a)]
    (if (= 1 (count ids))
      [who (first ids)]
      (err/raise "Unexpected variable in (H ...)"
                 {:hint "You can only calculate the entropy of other agents' choices, e.g. alice.x"
                  :loc loc}))
    (err/raise "Unexpected variable in (H ...)"
               {:hint "You can only calculate the entropy of other agents' choices, e.g. alice.x"
                :loc loc})))

(defn parse-expr [form pctxt]
  (let [loc (loc-of form pctxt)]
    (cond
      (number? form)
      {:node :lit :value form :loc loc :static true}

      (boolean? form)
      {:node :lit :value (if form 1 0) :loc loc :static true}

      (= ellipsis-sym form)
      (err/raise "`...` is only allowed as the argument list of a memo/FFI call"
                 {:loc loc})

      (and (dotted-parts form)
           (= 2 (count (dotted-parts form)))
           (some? ((:resolve-member pctxt (constantly nil))
                   (first (dotted-parts form)) (second (dotted-parts form)))))
      {:node :lit
       :value ((:resolve-member pctxt) (first (dotted-parts form)) (second (dotted-parts form)))
       :loc loc :static true}

      (dotted-parts form)
      (let [[who & ids] (dotted-parts form)]
        (check-cast who pctxt loc)
        (reduce (fn [inner w] {:node :with :who w :expr inner :loc loc :static false})
                {:node :choice :id (last ids) :loc loc :static false}
                (reverse (cons who (butlast ids)))))

      (symbol? form)
      (cond
        (contains? (:exotic-parameters pctxt) form)
        (err/raise "Unexpected use of non-numeric parameter"
                   {:hint (str form " is labeled as a special (non-numeric / array) "
                               "parameter. Such parameters can only be used as direct "
                               "arguments to function calls.")
                    :loc loc})

        (some #(= form %) (:static-parameters pctxt))
        {:node :lit :value form :loc loc :static true}

        :else {:node :choice :id form :loc loc :static false})

      (set? form)
      (if (= 1 (count form))
        {:node :inline :form (first form) :loc loc :static true}
        (err/raise "Inline escape hatch #{...} should contain exactly one form"
                   {:hint "Double-check for commas or extra values!" :loc loc}))

      (vector? form)
      (err/raise "Unknown expression syntax (unexpected vector)"
                 {:hint "Vectors are only used for choice/axis bindings, e.g. (f [x] ...)."
                  :loc loc})

      (seq? form)
      (let [[head & args] form]
        (cond
          (not (symbol? head))
          (err/raise "Unknown expression syntax" {:hint (pr-str form) :loc loc})

          ;; conditional
          (= head 'if)
          (do (when (not= 3 (count args))
                (err/raise "(if c t f) requires exactly three arguments" {:loc loc}))
              (let [[c t f] (mapv #(parse-expr % pctxt) args)]
                (op-node :ite [c t f] loc)))

          ;; unary math fns (static=false, matching parse.py)
          (contains? unary-fn-ops head)
          (do (when (not= 1 (count args))
                (err/raise (str "(" head " x) takes exactly one argument") {:loc loc}))
              (op-node (unary-fn-ops head) [(parse-expr (first args) pctxt)] loc
                       :static false))

          (= head 'not)
          (do (when (not= 1 (count args))
                (err/raise "(not x) takes exactly one argument" {:loc loc}))
              (op-node :inv [(parse-expr (first args) pctxt)] loc))

          ;; unary minus / left-fold arithmetic
          (contains? fold-ops head)
          (let [op (fold-ops head)
                parsed (mapv #(parse-expr % pctxt) args)]
            (cond
              (empty? parsed)
              (err/raise (str "(" head ") needs arguments") {:loc loc})

              (and (= head '-) (= 1 (count parsed)))
              (op-node :neg parsed loc)

              (and (= head '/) (= 1 (count parsed)))
              (op-node :div [{:node :lit :value 1 :loc loc :static true} (first parsed)] loc)

              (= 1 (count parsed)) (first parsed)

              :else
              (reduce (fn [acc e] (op-node op [acc e] loc)) parsed)))

          ;; binary-only operators
          (contains? binary-only-ops head)
          (do (when (not= 2 (count args))
                (err/raise (str "(" head " a b) takes exactly two arguments") {:loc loc}))
              (op-node (binary-only-ops head)
                       (mapv #(parse-expr % pctxt) args) loc))

          ;; and/or fold right (like Python BoolOp)
          (contains? bool-ops head)
          (let [op (bool-ops head)
                parsed (mapv #(parse-expr % pctxt) args)]
            (when (< (count parsed) 2)
              (err/raise (str "Unexpectedly few arguments to logical operator " head)
                         {:user false :loc loc}))
            (reduce (fn [acc e] (op-node op [e acc] loc))
                    (last parsed)
                    (reverse (butlast parsed))))

          ;; probabilistic operators
          (= head 'E)
          (do (when (not= 1 (count args))
                (err/raise "Incorrect syntax in (E ...)"
                           {:hint "E takes exactly one expression." :loc loc}))
              {:node :expect :expr (parse-expr (first args) pctxt)
               :reduction :expectation :warn true :surface 'E
               :loc loc :static false})

          (= head 'Var)
          (do (when (not= 1 (count args))
                (err/raise "Incorrect syntax in (Var ...)"
                           {:hint "Var takes exactly one expression." :loc loc}))
              {:node :expect :expr (parse-expr (first args) pctxt)
               :reduction :variance :warn true :surface 'Var
               :loc loc :static false})

          (= head 'Pr)
          (do (when (empty? args)
                (err/raise "Incorrect syntax in (Pr ...)" {:loc loc}))
              (let [parsed (mapv #(parse-expr % pctxt) args)
                    conj-expr (reduce (fn [acc e] (assoc (op-node :and [acc e] loc)
                                                         :what "Pr with several conditions"))
                                      parsed)]
                {:node :expect :expr conj-expr
                 :reduction :expectation :warn true :surface 'Pr
                 :surface-n (count parsed)
                 :loc loc :static false}))

          (= head 'H)
          (do (when (empty? args)
                (err/raise "Invalid entropy expression" {:loc loc}))
              {:node :entropy :rvs (mapv #(entropy-ref % pctxt loc) args)
               :loc loc :static false})

          (= head 'KL)
          (do (when (not= 2 (count args))
                (err/raise "(KL p.x q.y) takes exactly two choice references" {:loc loc}))
              (let [[[pw pi] [qw qi]] (mapv #(entropy-ref % pctxt loc) args)]
                {:node :kl :p-who pw :p-id pi :q-who qw :q-id qi
                 :loc loc :static false}))

          (= head 'EU)
          (do (when-not (and (= 1 (count args)) (symbol? (first args)))
                (err/raise "(EU goal) takes the name of a goal" {:loc loc}))
              {:node :util :goal (first args) :loc loc :static false})

          (= head 'Predict)
          (do (when (not= 1 (count args))
                (err/raise "(Predict e) takes a single expression"
                           {:hint "Use (Predict expr) with one expression, e.g. (Predict (+ alice.x 7))."
                            :loc loc}))
              {:node :predict :expr (parse-expr (first args) pctxt) :loc loc :static false})

          (= head 'imagine)
          (let [stmt-forms (butlast args)
                then-form (last args)]
            (when (nil? then-form)
              (err/raise "(imagine ...) needs a final expression" {:loc loc}))
            (when (and (seq? then-form) (symbol? (first then-form))
                       (keyword? (second then-form)))
              (err/raise "Invalid final form in imagine expression"
                         {:hint "The final item in an `imagine` expression should be an expression to be evaluated, not a statement."
                          :loc loc}))
            {:node :imagine
             :do (vec (mapcat #(parse-stmt-form % pctxt) stmt-forms))
             :then (parse-expr then-form pctxt)
             :loc loc :static false})

          (= head 'as)
          (do (when (not= 2 (count args))
                (err/raise "(as agent expr) takes an agent and one expression"
                           {:hint "This is the analog of Python memo's agent[expr] query."
                            :loc loc}))
              (let [who (first args)]
                (when-not (symbol? who)
                  (err/raise "(as agent expr): agent must be a name" {:loc loc}))
                (check-cast who pctxt loc)
                {:node :with :who who :expr (parse-expr (second args) pctxt)
                 :loc loc :static false}))

          ;; Python memo's cost @ f(...), time @, space @, size @: the reader
          ;; reads (cost @(f ...)) as (cost (clojure.core/deref (f ...))).
          ;; Any other (cost ...) is a host function call, as cost(...) is in
          ;; Python memo (demo-trucks has one).
          (and (#{'cost 'time 'space 'size} head)
               (= 1 (count args))
               (seq? (first args))
               (= 'clojure.core/deref (first (first args))))
          (err/raise "cost reflection is not supported in memo.clj yet (DESIGN.md M4)"
                     {:loc loc})

          (contains? reserved-heads head)
          (err/raise (str "Unexpected use of reserved form " head) {:loc loc})

          :else (parse-call form pctxt loc)))

      :else
      (err/raise "Unknown expression syntax" {:hint (pr-str form) :loc loc}))))

;; -----------------------------------------------------------------------------
;; statements

(def ^:private choose-aliases #{:chooses :given :draws :assigned :guesses})

(def variants
  "Theories an agent can be given with :has-theory (Python memo's Variant):
  :child cannot use :thinks; the three ablations (memo 1.2.25) cannot use
  :thinks, :knows or :wants respectively. Both spellings are accepted."
  {:adult :adult, :child :child,
   :no-thinks :no-thinks, :no-knows :no-knows, :no-wants :no-wants,
   'ADULT :adult, 'CHILD :child,
   'NO_THINKS :no-thinks, 'NO_KNOWS :no-knows, 'NO_WANTS :no-wants})
(def ^:private observe-aliases #{:observes :sees :hears})

(defn- parse-choice-binding [b pctxt loc]
  (if (and (vector? b) (= 2 (count b)) (symbol? (first b)) (symbol? (second b)))
    [(first b) (second b)]
    (err/raise "Unexpected item in chooses"
               {:hint "Expected a sequence of [name Domain] bindings."
                :loc loc})))

(defn- to-be-wpp
  "Port of parse.py's `to_be=`: the weight is the conjunction of (= e c) over
  every choice c bound in the statement, seeded with the literal 1, so the
  choice is deterministically the expression's value (normalize reduction)."
  [e choices loc]
  (reduce (fn [acc [id _]]
            (op-node :and
                     [acc (op-node :eq [e {:node :choice :id id :loc loc :static false}]
                                   loc)]
                     loc))
          {:node :lit :value 1 :loc loc :static true}
          choices))

(defn- parse-choose [who kw args form pctxt]
  (let [loc (loc-of form pctxt)
        choices (mapv #(parse-choice-binding % pctxt loc)
                      (take-while vector? args))
        rest-args (drop-while vector? args)]
    (when (empty? choices)
      (err/raise "`chooses` needs at least one choice, but none provided"
                 {:hint "Specify a choice as [name Domain]" :loc loc}))
    (let [kwset (set rest-args)
          uniformly? (contains? kwset :uniformly)
          [pos kw-pairs] (split-kwargs (remove #(= :uniformly %) rest-args) loc)]
      (when (seq pos)
        (err/raise "Unexpected item in chooses"
                   {:hint "After the [name Domain] bindings, chooses only takes one of :wpp / :such-that / :to-be / :to-maximize / :to-minimize / :uniformly."
                    :loc loc}))
      (when (and uniformly? (seq kw-pairs))
        (err/raise "Cannot use `:uniformly` together with a keyword argument"
                   {:hint "Use either :uniformly or one of: :wpp, :such-that, :to-be, :to-maximize, or :to-minimize"
                    :loc loc}))
      (when (and (not uniformly?) (not= 1 (count kw-pairs)))
        (err/raise "wrong number of keyword arguments to chooses"
                   {:hint "expected exactly one of: :wpp, :such-that, :to-be, :to-maximize, :to-minimize (or :uniformly)"
                    :loc loc}))
      (let [[k vform] (if uniformly? ['wpp 1] (first kw-pairs))
            surface (parse-expr vform pctxt)   ; the expression as written
            wpp surface
            [reduction wpp]
            (case (name k)
              "wpp" [:normalize wpp]
              "such-that" [:normalize wpp]
              "to-be" [:normalize (to-be-wpp wpp choices loc)]
              "to-maximize" [:maximize wpp]
              "to-minimize" [:maximize {:node :op :op :neg :args [wpp]
                                        :loc (:loc wpp) :static (:static wpp)}]
              (err/raise (str "unknown keyword argument to chooses: :" k)
                         {:hint "expected exactly one of: :wpp, :such-that, :to-be, :to-maximize, :to-minimize (or :uniformly)"
                          :loc loc}))]
        [(cond-> {:node :choose :who who :choices choices :wpp wpp
                  :reduction reduction :loc loc
                  :surface-alias kw
                  :surface-spec (if uniformly? :uniformly (keyword (name k)))}
           ;; the surface expression before desugaring, for the transpiler
           (= "to-be" (name k)) (assoc :spec-expr surface))]))))

(defn- knows-ref [a pctxt loc who]
  (cond
    (dotted-parts a)
    (let [[w & ids] (dotted-parts a)]
      (when (not= 1 (count ids))
        (err/raise "Invalid input to (:knows ...)" {:loc loc}))
      {:node :knows :who who :source-who w :source-id (first ids) :loc loc})

    (symbol? a)
    {:node :knows :who who :source-who 'self :source-id a :loc loc}

    :else
    (err/raise "Invalid input to :knows"
               {:hint "You can only supply names of choices (e.g. x or alice.y) to :knows."
                :loc loc})))

(defn- parse-observes
  "A malformed observes is Python memo's \"Unknown statement syntax\" (its
  parser matches no statement); the hint says what is wrong."
  [who kw args form pctxt]
  (let [loc (loc-of form pctxt)]
    (when-not (and (= 3 (count args)) (= :is (second args)))
      (err/raise "Unknown statement syntax"
                 {:hint (str "Expected (" who " :observes target.x :is source), where "
                             "source is a choice name or another agent's choice.")
                  :loc loc}))
    (let [[target _ source] args
          tparts (dotted-parts target)]
      (when (or (nil? tparts) (not= 2 (count tparts)))
        (err/raise "Unknown statement syntax"
                   {:hint "The observed value must be another agent's choice, e.g. speaker.u."
                    :loc loc}))
      (let [[tw ti] tparts
            [sw si] (if-let [sparts (dotted-parts source)]
                      (if (= 2 (count sparts))
                        sparts
                        (err/raise "Unknown statement syntax"
                                   {:hint "The observed value must be a choice name, e.g. x or speaker.u."
                                    :loc loc}))
                      (if (symbol? source)
                        ['self source]
                        (err/raise "Unknown statement syntax"
                                   {:hint "The observed value cannot be an arbitrary value; it must be the name of a choice. For conditioning on expressions, use :observes-that or :observes-event."
                                    :loc loc})))]
        [{:node :show :who who :target-who tw :target-id ti
          :source-who sw :source-id si :loc loc :surface-alias kw}]))))

(defn parse-agent-stmt [who args form pctxt]
  (let [loc (loc-of form pctxt)
        _ (when (= 'self who)
            (err/raise "`self` cannot be the name of an agent"
                       {:hint (str "self is how an agent refers to itself (self.x is "
                                   "its own choice x). Pick another name for this agent.")
                        :loc loc}))
        [kw & rest-args] args]
    (when-not (keyword? kw)
      (err/raise "Unknown statement syntax"
                 {:hint (str "Expected (" who " :keyword ...); got " (pr-str form))
                  :loc loc}))
    (cond
      (contains? choose-aliases kw)
      (parse-choose who kw rest-args form pctxt)

      (contains? observe-aliases kw)
      (parse-observes who kw rest-args form pctxt)

      (= kw :observes-that)
      (do (when (not= 1 (count rest-args))
            (err/raise "(:observes-that e) takes exactly one expression" {:loc loc}))
          [{:node :observes-factor :who who
            :what (parse-expr (first rest-args) pctxt)
            :how :boolean :loc loc}])

      (= kw :observes-event)
      (do (when-not (and (= 2 (count rest-args)) (= :wpp (first rest-args)))
            (err/raise "(:observes-event :wpp e) is the only supported form" {:loc loc}))
          [{:node :observes-factor :who who
            :what (parse-expr (second rest-args) pctxt)
            :how :probability :loc loc}])

      (= kw :knows)
      (mapv #(knows-ref % pctxt loc who) rest-args)   ; none is no statement, as in Python memo

      (= kw :thinks)
      (mapv (fn [s] {:node :with-stmt :who who :stmt s :loc (:loc s)})
            (vec (mapcat #(parse-stmt-form % pctxt) rest-args)))

      (= kw :wants)
      (do (when-not (and (= 2 (count rest-args)) (symbol? (first rest-args)))
            (err/raise "Malformed wants statement"
                       {:hint (str "Expected (" who " :wants goal-name expr).") :loc loc}))
          [{:node :wants :who who :what (first rest-args)
            :how (parse-expr (second rest-args) pctxt) :loc loc}])

      (= kw :snapshots-self-as)
      (do (when (empty? rest-args)
            (err/raise "(:snapshots-self-as alias) needs at least one alias" {:loc loc}))
          (mapv (fn [a]
                  (if (symbol? a)
                    {:node :snapshot :who who :alias a :loc loc}
                    (err/raise "Inputs to :snapshots-self-as must be names of agents"
                               {:hint (str "`" (pr-str a) "` is not a name") :loc loc})))
                rest-args))

      (= kw :has-theory)
      (let [v (first rest-args)
            variant (or (get variants v)
                        (err/raise "Unknown variant"
                                   {:hint (str "`" (pr-str v) "` is not a recognized memo variant. "
                                               "The available variants are: :adult, :child, "
                                               ":no-thinks, :no-knows, :no-wants")
                                    :loc loc}))]
        [{:node :variant :who who :variant variant :loc loc}])

      (= kw :forgets-about)
      (do (when (empty? rest-args)
            (err/raise "`:forgets-about` needs at least one choice, but none provided"
                       {:hint (str "Specify the choice(s) to forget, e.g. (" who
                                   " :forgets-about x y)")
                        :loc loc}))
          (doseq [a rest-args]
            (when-not (and (symbol? a) (nil? (dotted-parts a)))
              (err/raise "Invalid input to (:forgets-about ...)"
                         {:hint "You can only supply the names of your own choices (e.g. x), not another agent's choice (e.g. alice.x)."
                          :loc loc})))
          [{:node :forgets-about :who who :ids (vec rest-args) :loc loc}])

      (= kw :inspect)
      [{:node :trace :who who :loc loc}]

      :else
      (err/raise "Unknown statement syntax"
                 {:hint (str "Unrecognized statement keyword " kw " in " (pr-str form))
                  :loc loc}))))

(defn parse-stmt-form
  "Parse one statement form into a vector of Stmt nodes."
  [form pctxt]
  (let [loc (loc-of form pctxt)]
    (cond
      (and (seq? form) (= 'forall (first form)))
      (let [[_ binding] form]
        (when-not (and (vector? binding) (= 2 (count binding))
                       (every? symbol? binding))
          (err/raise "Malformed forall statement"
                     {:hint "Expected (forall [x Domain])." :loc loc}))
        (let [[id dom] binding]
          (when (some #(= id %) (:static-parameters pctxt))
            (err/raise (str "The name `" id "` is already used for a static parameter "
                            "and cannot be used as a query variable.")
                       {:loc loc}))
          [{:node :forall :id id :domain dom :loc loc}]))

      (and (seq? form) (symbol? (first form)) (not (contains? reserved-heads (first form))))
      (parse-agent-stmt (first form) (rest form) form pctxt)

      :else
      (err/raise "Unknown statement syntax" {:hint (pr-str form) :loc loc}))))

;; -----------------------------------------------------------------------------
;; whole-memo parsing

(defn- parse-param [p pctxt-name]
  (cond
    (symbol? p) {:name p :default nil :exotic (boolean (:array (meta p)))}
    (and (seq? p) (= 2 (count p)) (symbol? (first p)))
    {:name (first p) :default (second p) :exotic (boolean (:array (meta (first p))))}
    (and (vector? p) (= 2 (count p)) (symbol? (first p)))
    {:name (first p) :default (second p) :exotic (boolean (:array (meta (first p))))}
    :else (err/raise (str "Malformed parameter " (pr-str p) " in defmemo " pctxt-name)
                     {:hint "Parameters are symbols, optionally with defaults: [alpha (beta 1.0)]. Mark array/non-numeric parameters with ^:array."})))

(defn parse-defmemo
  "Parse a full defmemo body. Returns
  {:pctxt … :params [{:name :default :exotic}…] :stmts […] :retvals […] :doc s}.
  opts: :resolve-member, a function of [domain-sym member-sym] giving an enum
  member's value, or nil when domain-sym names no enum."
  ([mname axes params body] (parse-defmemo mname axes params body {}))
  ([mname axes params body {:keys [resolve-member]}]
  (when-not (and (vector? axes) (every? #(and (vector? %) (= 2 (count %))
                                              (every? symbol? %)) axes))
    (err/raise (str "Malformed axes in defmemo " mname)
               {:hint "Axes are a vector of [axis-name Domain] pairs, e.g. [[u Utt] [n Count]]."}))
  (let [param-entries (mapv #(parse-param % mname) params)
        base-pctxt {:name mname
                    :static-parameters (mapv :name param-entries)
                    :exotic-parameters (set (keep #(when (:exotic %) (:name %)) param-entries))
                    :cast nil
                    :axes (mapv (fn [[a d]] [a d]) axes)
                    :resolve-member (or resolve-member (constantly nil))}
        ;; leading docstring
        [doc body] (if (string? (first body))
                     [(first body) (rest body)]
                     [nil body])
        ;; optional (cast [...]) as first statement
        [cast body] (if (and (seq? (first body)) (= 'cast (ffirst body)))
                      (let [c (second (first body))]
                        (when-not (and (vector? c) (every? symbol? c))
                          (err/raise "Cast list entries must be plain names" {}))
                        [(set c) (rest body)])
                      [nil body])
        pctxt (assoc base-pctxt :cast cast :doc doc)
        forall-stmts (mapv (fn [[a d]] {:node :forall :id a :domain d :loc nil})
                           axes)]
    (loop [forms body
           stmts forall-stmts
           retvals []]
      (if (empty? forms)
        (do (when (empty? retvals)
              (err/raise "No return statement"
                         {:hint "All memos should end with a return statement"
                          :loc {:name mname}}))
            ;; the resolver is a function: the AST is embedded in code as data
            {:pctxt (dissoc pctxt :resolve-member) :params param-entries
             :stmts stmts :retvals retvals :doc doc})
        (let [form (first forms)]
          (cond
            (and (seq? form) (= 'return (first form)))
            (do (when (not= 2 (count form))
                  (err/raise "(return e) takes exactly one expression"
                             {:loc (loc-of form pctxt)}))
                (recur (rest forms) stmts
                       (conj retvals (parse-expr (second form) pctxt))))

            (and (seq? form) (= 'cast (first form)))
            (err/raise "(cast [...]) must be the first statement of the memo"
                       {:loc (loc-of form pctxt)})

            :else
            ;; the cast binds the memo's own statements and the agents its
            ;; expressions name, not the agents inside a mind or imagine
            ;; (as Python memo checks it)
            (do (when (and (seq? form) (symbol? (first form))
                           (not (contains? reserved-heads (first form))))
                  (check-cast (first form) pctxt (loc-of form pctxt)))
                (recur (rest forms)
                       (into stmts (parse-stmt-form form pctxt))
                       retvals)))))))))
