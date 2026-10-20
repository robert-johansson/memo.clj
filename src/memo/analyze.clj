(ns memo.analyze
  "The compiler's front end: the epistemic analysis of a parsed memo, which
  emits the IR (memo.ir). It walks the statements as Python memo's core.py
  does, threading an immutable state: the frame tree (each agent's
  perspective), what each expression's value is known to and depends on, and
  the instructions emitted so far. The known/deps checks make illegal models
  compile errors, with Python memo's messages.

  State: {:root frame :path [names] :instrs [] :next-dim n :sym-n n
          :path-conds [syms] :continuation [stmts] :forall [[dim id domain]]
          :pctxt {} :domains {domain-sym [values]}}

  eval-expr : [state expr] -> [state {:tag sym :known bool :deps #{[who id]}}]
  eval-stmt : [state stmt] -> state

  Lifted from memo-clj's compile after review against core.py. It covers the
  constructs built so far (M1, slice 1); the others are errors that say so."
  (:require [memo.errors :as err]
            [memo.frames :as fr]))

;; -----------------------------------------------------------------------------
;; state

(defn- fresh [state hint]
  (let [n (:sym-n state)]
    [(assoc state :sym-n (inc n)) (symbol (str hint "$" n))]))

(defn- emit [state instr] (update state :instrs conj instr))

(defn- frame-err-opts [state loc]
  {:loc loc :frame-ctxt (fr/frame-context state)})

(defn- domain-values [state dom loc]
  (or (get-in state [:domains dom])
      (err/raise (str "Unknown domain " dom)
                 {:hint (str "Domains must be defined (as vectors or with defenum) "
                             "before the defmemo that uses them.")
                  :loc loc})))

(defn- unsupported [what loc]
  (err/raise (str what " is not supported in memo.clj yet")
             {:hint "memo.clj is being built construct by construct (M1, DESIGN.md §1)."
              :loc loc}))

(defn- known-deps
  "The choices the current frame knows: the deps of a value it computes from
  its beliefs."
  [state]
  (set (keep (fn [[addr c]] (when (:known c) addr))
             (:choices (fr/get-frame state)))))

(defn- unknown-dims
  "The dims of the choices the current frame is uncertain about."
  [state]
  (vec (distinct (keep (fn [[_ c]] (when-not (:known c) (:idx c)))
                       (:choices (fr/get-frame state))))))

;; -----------------------------------------------------------------------------
;; expressions

(declare eval-expr)

(defn- eval-lit [state e]
  (let [v (:value e)
        [state out] (fresh state (if (symbol? v) "param" "lit"))]
    [(emit state (if (symbol? v)
                   {:i :param :out out :name v}
                   {:i :const :out out :value v}))
     {:tag out :known true :deps #{}}]))

(defn- eval-choice [state e]
  (let [id (:id e)
        frm (fr/get-frame state)
        ch (get-in frm [:choices ['self id]])]
    (when-not ch
      (err/raise (str "Unknown choice " (:name frm) "." id)
                 (assoc (frame-err-opts state (:loc e))
                        :hint
                        (str "Did you perhaps misspell " id "?"
                             (if (empty? (:path state))
                               (str " Or, did you forget to include " id
                                    " as an axis in the definition of this memo?")
                               (str " " (:name frm) " is not yet aware of any choice called "
                                    id ". Or, did you forget an earlier ("
                                    (:name frm) " :chooses [" id " ...]) or ("
                                    (:name frm) " :knows " id ") statement?"))))))
    [state {:tag (:tag ch) :known (:known ch) :deps #{['self id]}}]))

(defn- eval-op [state e]
  (let [{:keys [op args]} e]
    (cond
      (contains? #{:add :sub :mul :div :pow :mod :eq :neq :lt :lte :gt :gte
                   :and :or :xor} op)
      (let [[state l] (eval-expr state (first args))
            [state r] (eval-expr state (second args))
            [state out] (fresh state (str "op_" (name op)))]
        [(emit state {:i :op :op op :out out :args [(:tag l) (:tag r)]})
         {:tag out :known (and (:known l) (:known r)) :deps (into (:deps l) (:deps r))}])

      (contains? #{:exp :abs :log :uadd :neg :inv} op)
      (let [[state l] (eval-expr state (first args))
            [state out] (fresh state (str "op_" (name op)))]
        [(emit state {:i :op :op op :out out :args [(:tag l)]})
         {:tag out :known (:known l) :deps (:deps l)}])

      (= op :ite)
      (let [[c-ast t-ast f-ast] args
            [state c] (eval-expr state c-ast)
            ;; a static condition guards the memo calls in its branches
            ;; (path conditions, as core.py's path_depends)
            [state tv fv]
            (if (:static c-ast)
              (let [state (update state :path-conds conj (:tag c))
                    [state tv] (eval-expr state t-ast)
                    state (update state :path-conds pop)
                    [state inv] (fresh state "op_inv")
                    state (emit state {:i :op :op :inv :out inv :args [(:tag c)]})
                    state (update state :path-conds conj inv)
                    [state fv] (eval-expr state f-ast)
                    state (update state :path-conds pop)]
                [state tv fv])
              (let [[state tv] (eval-expr state t-ast)
                    [state fv] (eval-expr state f-ast)]
                [state tv fv]))
            [state out] (fresh state "op_ite")]
        [(emit state {:i :where :out out :args [(:tag c) (:tag tv) (:tag fv)]})
         {:tag out
          :known (and (:known c) (:known tv) (:known fv))
          :deps (into (into (:deps c) (:deps tv)) (:deps fv))}])

      :else (err/raise (str "Unknown operator " op) {:user false :loc (:loc e)}))))

(defn- eval-with
  "who's value of an expression, as the current frame models it: Python
  memo's agent[expr] (here (as who expr), or who.x for a choice)."
  [state e]
  (let [{:keys [who expr]} e]
    (if (= who 'self)
      (eval-expr state expr)
      (let [state (fr/ensure-child state who)
            state (fr/enter state who)
            [state val] (eval-expr state expr)
            state (fr/leave state)
            frm (fr/get-frame state)]
        (when-not (:known val)
          (err/raise "Asking an agent for an unknown value"
                     (assoc (frame-err-opts state (:loc e))
                            :hint (str who " has uncertainty about the value of the "
                                       "expression that " (:name frm) " is imagining "
                                       who " computing. Did you perhaps mean to take "
                                       who "'s *expected* value of that expression, "
                                       "using (E ...)?"))))
        (let [child (get-in frm [:children who])
              deps (reduce (fn [acc [w i :as dep]]
                             (cond
                               (contains? (:conditions child) dep)
                               (conj acc (get (:conditions child) dep))
                               (= w 'self) (conj acc [who i])
                               :else (err/raise "Internal error mapping dependencies"
                                                {:user false :loc (:loc e)})))
                           #{}
                           (:deps val))]
          [state {:tag (:tag val)
                  :known (every? #(get-in frm [:choices % :known]) deps)
                  :deps deps}])))))

(defn- eval-posterior
  "Pr[x == agent.y] (and conjunctions of such pairs), with x known and
  agent.y uncertain over the same domain: the marginal over agent.y,
  re-indexed by x, instead of the comparison over both dims (core.py's
  EPosterior). Required, not an optimization: some models do not fit in
  memory without it."
  [state {:keys [query var]}]
  (let [frm (fr/get-frame state)
        var-idxs (set (map #(get-in frm [:choices % :idx]) var))
        marg-dims (vec (distinct (keep (fn [[_ c]]
                                         (when-not (or (:known c) (var-idxs (:idx c)))
                                           (:idx c)))
                                       (:choices frm))))
        [state out] (fresh state "posterior")
        state (emit state {:i :sum :out out :arg (:ll frm) :dims marg-dims})
        [state out] (reduce (fn [[st prev] [knw-addr var-addr]]
                              (let [[st o] (fresh st "posterior")]
                                [(emit st {:i :posterior-align :out o :arg prev
                                           :knw (get-in frm [:choices knw-addr :idx])
                                           :var (get-in frm [:choices var-addr :idx])})
                                 o]))
                            [state out]
                            (map vector query var))]
    [state {:tag out :known true :deps (known-deps state)}]))

(defn- posterior-pairs
  "The [known-addr uncertain-addr] pairs of an expression of the form
  (and (= x agent.y) …), each pair eligible for the posterior shortcut; nil
  when the expression is not of that form."
  [state expr]
  (let [{:keys [node op args]} expr
        frm (fr/get-frame state)
        eligible? (fn [lc rw rc]
                    (let [lcc (get-in frm [:choices ['self lc]])
                          rcc (get-in frm [:choices [rw rc]])]
                      (and lcc rcc (:known lcc) (not (:known rcc))
                           (= (:domain lcc) (:domain rcc)))))]
    (cond
      (and (= :op node) (= :eq op))
      (let [[a b] args
            pair (fn [ec ew]
                   (when (and (= :choice (:node ec))
                              (= :with (:node ew))
                              (= :choice (get-in ew [:expr :node])))
                     (let [lc (:id ec) rw (:who ew) rc (get-in ew [:expr :id])]
                       (when (eligible? lc rw rc)
                         [[['self lc] [rw rc]]]))))]
        (or (pair a b) (pair b a)))

      (and (= :op node) (= :and op))
      (let [f (posterior-pairs state (first args))
            s (posterior-pairs state (second args))]
        (when (and f s) (into f s)))

      :else nil)))

(defn- eval-expect [state e]
  (let [{:keys [expr reduction warn]} e
        frm (fr/get-frame state)
        pairs (posterior-pairs state expr)
        distinct-axes? (when pairs
                         (let [idxs (mapcat (fn [[k v]] [(get-in frm [:choices k :idx])
                                                         (get-in frm [:choices v :idx])])
                                            pairs)]
                           (= (count (set idxs)) (count idxs))))]
    (if (and pairs distinct-axes?)
      (eval-posterior state {:query (mapv first pairs) :var (mapv second pairs)})
      (let [[state val] (eval-expr state expr)
            frm (fr/get-frame state)
            all-known? (every? #(get-in frm [:choices % :known]) (:deps val))]
        (when (and all-known? warn)
          (err/warn "Redundant " (name reduction) ", not marginalizing..."))
        ;; a redundant expectation is the value itself, without the frame's
        ;; beliefs: after an impossible observation that is not the same
        ;; number (DESIGN.md §2.4), so it is semantics, not an optimization
        (if (and all-known? (= reduction :expectation))
          [state val]
          (let [dims (unknown-dims state)
                [state ll] (if (:ll frm)
                             [state (:ll frm)]
                             (let [[st one] (fresh state "ll")]
                               [(emit st {:i :const :out one :value 1.0}) one]))
                [state lv] (fresh state "llval")
                state (emit state {:i :op :op :mul :out lv :args [ll (:tag val)]})]
            (case reduction
              :expectation
              (let [[state out] (fresh state "exp")]
                [(emit state {:i :sum :out out :arg lv :dims dims})
                 {:tag out :known true :deps (known-deps state)}])
              ;; centered, E[(X - E[X])^2], as memo-clj: Python memo's
              ;; E[X^2] - E[X]^2 is the same number without float32's
              ;; cancellation
              :variance
              (let [[state two] (fresh state "lit")
                    state (emit state {:i :const :out two :value 2})
                    [state mean] (fresh state "exp")
                    state (emit state {:i :sum :out mean :arg lv :dims dims})
                    [state dev] (fresh state "dev")
                    state (emit state {:i :op :op :sub :out dev :args [(:tag val) mean]})
                    [state sq] (fresh state "sq")
                    state (emit state {:i :op :op :pow :out sq :args [dev two]})
                    [state lv2] (fresh state "llval")
                    state (emit state {:i :op :op :mul :out lv2 :args [ll sq]})
                    [state out] (fresh state "var")]
                [(emit state {:i :sum :out out :arg lv2 :dims dims})
                 {:tag out :known true :deps (known-deps state)}]))))))))

(defn eval-expr [state e]
  (case (:node e)
    :lit (eval-lit state e)
    :choice (eval-choice state e)
    :op (eval-op state e)
    :with (eval-with state e)
    :expect (eval-expect state e)
    :ffi (unsupported "A host function call" (:loc e))
    :memo (unsupported "A memo call" (:loc e))
    :entropy (unsupported "(H ...)" (:loc e))
    :kl (unsupported "(KL ...)" (:loc e))
    :imagine (unsupported "(imagine ...)" (:loc e))
    :inline (unsupported "The inline escape hatch #{...}" (:loc e))
    :predict (unsupported "(Predict ...)" (:loc e))
    :util (unsupported "(EU ...)" (:loc e))
    (err/raise (str "Unknown expression node " (:node e)) {:user false :loc (:loc e)})))

;; -----------------------------------------------------------------------------
;; statements

(defn- eval-forall [state {:keys [id domain loc]}]
  (when (seq (:path state))
    (err/raise "forall is only allowed at the top level" {:user false :loc loc}))
  (let [idx (:next-dim state)
        state (assoc state :next-dim (inc idx))
        [state tag] (fresh state (str "forall_" id))
        state (emit state {:i :domain :out tag :dim idx
                           :values (domain-values state domain loc)})
        state (fr/update-frame state assoc-in [:choices ['self id]]
                               {:tag tag :idx idx :known true :domain domain})]
    (update state :forall conj [idx id domain])))

(defn- eval-choose
  "who chooses: the choices join who's own frame (known to who) and the
  current frame (uncertain), and the current frame's beliefs are multiplied
  by the choice's normalized weights."
  [state {:keys [who choices wpp reduction loc]}]
  (let [state (fr/ensure-child state who)
        state (fr/enter state who)
        [state dims]
        (reduce
         (fn [[st dims] [id dom]]
           (when (get-in (fr/get-frame st) [:choices ['self id]])
             (err/raise "Repeated choice"
                        (assoc (frame-err-opts st loc)
                               :hint (str who " has already chosen " id
                                          " earlier in this model! Pick a new name?"))))
           (when (some #(= id %) (get-in st [:pctxt :static-parameters]))
             (err/raise "Name conflict"
                        (assoc (frame-err-opts st loc)
                               :hint (str "The name " id " is already being used as a "
                                          "parameter to the model."))))
           (let [idx (:next-dim st)
                 st (assoc st :next-dim (inc idx))
                 [st tag] (fresh st (str who "_" id))
                 st (emit st {:i :domain :out tag :dim idx :values (domain-values st dom loc)})
                 st (fr/update-frame st assoc-in [:choices ['self id]]
                                     {:tag tag :idx idx :known true :domain dom})]
             [st (conj dims idx)]))
         [state []]
         choices)
        ;; wpp = (exp logit) is computed as a softmax of the logit, which
        ;; does not overflow; the same weight written otherwise can (core.py)
        softmax? (and (= :op (:node wpp)) (= :exp (:op wpp)))
        [state w] (eval-expr state (if softmax? (first (:args wpp)) wpp))
        _ (doseq [[id _] choices]
            (when-not (:known w)
              (err/raise "Choice based on uncertain expression"
                         (assoc (frame-err-opts state loc)
                                :hint (str who " is uncertain about the value of the "
                                           "expression (wpp/to-maximize) that " who
                                           " is using to choose " id ". Hence, " who
                                           " cannot compute the probabilities needed to "
                                           "make the choice. Perhaps you meant to take "
                                           "an expected value somewhere, using (E ...)?"))))
            (when (and (not (contains? (:deps w) ['self id])) (not= :lit (:node wpp)))
              (err/warn "When " who " chooses " id ", the probability doesn't depend "
                        "on the value of " id ". As a result, the choice will "
                        "effectively be uniform over " id " after normalization. "
                        "Are you sure this is what you want? (A uniform choice is "
                        "more easily expressed with :uniformly.)")))
        state (fr/leave state)
        state (reduce (fn [st [id dom]]
                        (let [ch (get-in (fr/get-frame st) [:children who :choices ['self id]])]
                          (fr/update-frame st assoc-in [:choices [who id]]
                                           {:tag (:tag ch) :idx (:idx ch)
                                            :known false :domain dom})))
                      state
                      choices)
        ;; the weights, with extent along every chosen dim
        [state ll] (reduce (fn [[st acc] [[_ dom] dim]]
                             (let [[st o] (fresh st "ones")
                                   st (emit st {:i :ones :out o :dim dim
                                                :size (count (domain-values st dom loc))})
                                   [st m] (fresh st "w")]
                               [(emit st {:i :op :op :mul :out m :args [acc o]}) m]))
                           [state (:tag w)]
                           (map vector choices dims))
        [state ll] (if softmax?
                     (let [[st mx] (fresh state "max")
                           st (emit st {:i :max :out mx :arg ll :dims dims})
                           [st sh] (fresh st "shift")
                           st (emit st {:i :op :op :sub :out sh :args [ll mx]})
                           [st ex] (fresh st "softmax")]
                       [(emit st {:i :op :op :exp :out ex :args [sh]}) ex])
                     [state ll])
        ;; argmax: the values equal to the maximum, exactly (core.py)
        [state ll] (if (= reduction :maximize)
                     (let [[st mx] (fresh state "max")
                           st (emit st {:i :max :out mx :arg ll :dims dims})
                           [st eq] (fresh st "argmax")]
                       [(emit st {:i :op :op :eq :out eq :args [ll mx]}) eq])
                     [state ll])
        [state total] (fresh state "total")
        state (emit state {:i :sum :out total :arg ll :dims dims})
        [state q] (fresh state "ratio")
        state (emit state {:i :op :op :div :out q :args [ll total]})
        [state nrm] (fresh state "p")
        state (emit state {:i :nan-to-num :out nrm :arg q})
        frm (fr/get-frame state)
        [state prev] (if (:ll frm)
                       [state (:ll frm)]
                       (let [[st one] (fresh state "ll")]
                         [(emit st {:i :const :out one :value 1.0}) one]))
        [state new-ll] (fresh state "ll")
        state (emit state {:i :op :op :mul :out new-ll :args [nrm prev]})]
    (fr/update-frame state assoc :ll new-ll)))

(defn eval-stmt [state s]
  (case (:node s)
    :pass state
    :forall (eval-forall state s)
    :choose (eval-choose state s)
    :with-stmt (unsupported "thinks" (:loc s))
    :show (unsupported "observes" (:loc s))
    :observes-factor (unsupported "observes-that and observes-event" (:loc s))
    :knows (unsupported "knows" (:loc s))
    :snapshot (unsupported "snapshots-self-as" (:loc s))
    :variant (unsupported "has-theory" (:loc s))
    :trace (unsupported "inspect" (:loc s))
    :wants (unsupported "wants" (:loc s))
    :forgets-about (unsupported "forgets-about" (:loc s))
    (err/raise (str "Unknown statement node " (:node s)) {:user false :loc (:loc s)})))

;; -----------------------------------------------------------------------------
;; a whole memo

(defn initial-state
  "The analysis state before a memo's first statement."
  [pctxt domains variant]
  {:root (fr/frame fr/root-name (or variant :adult))
   :path []
   :instrs []
   :next-dim 0
   :sym-n 0
   :path-conds []
   :continuation []
   :forall []
   :pctxt pctxt
   :domains domains})

(defn analyze
  "The IR program of a parsed memo:
  {:instrs [...] :results [sym ...] :forall [[dim id domain] ...]
   :axes [id ...] :domains [domain ...] :shape [n ...] :n-dims n}.
  domains maps each domain symbol to its values."
  [{:keys [pctxt stmts retvals]} domains {:keys [variant]}]
  (let [state (reduce (fn [st [i s]]
                        (eval-stmt (assoc st :continuation (subvec stmts (inc i))) s))
                      (initial-state pctxt domains variant)
                      (map-indexed vector stmts))
        [state vals] (reduce (fn [[st acc] retval]
                               (let [[st val] (eval-expr st retval)]
                                 (when-not (:known val)
                                   (err/raise "Returning a value that the observer has uncertainty over"
                                              (assoc (frame-err-opts st (:loc retval))
                                                     :hint (str "Did you mean to wrap this return "
                                                                "value in an (E ...) expression or a "
                                                                "(Pr ...) expression?"))))
                                 [st (conj acc val)]))
                             [state []]
                             retvals)
        forall (:forall state)]
    {:instrs (:instrs state)
     :results (mapv :tag vals)
     :forall forall
     :axes (mapv second forall)
     :domains (mapv #(nth % 2) forall)
     :shape (mapv (fn [[_ _ dom]] (count (get domains dom))) forall)
     :n-dims (:next-dim state)}))
