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
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [memo.bounded :as b]
            [memo.errors :as err]
            [memo.frames :as fr]))

;; -----------------------------------------------------------------------------
;; state

(defn- fresh [state hint]
  (let [n (:sym-n state)]
    [(assoc state :sym-n (inc n)) (symbol (str hint "$" n))]))

(defn- emit [state instr] (update state :instrs conj instr))

(defn- frame-err-opts [state loc]
  {:loc loc :frame-ctxt (fr/frame-context state)})

(defn- fresh-tags
  "Every choice in frame and the frames inside it with a value symbol of its
  own, an :alias of the old one: [state frame']. A copy (a snapshot, an
  imagined world) must not share value symbols with its original, or an
  observation in one would reach the other (repoint follows the symbol):
  Python memo's snapshot bug before 1.3.3 (DESIGN.md §2.4)."
  [state frame]
  (let [tags-in (fn tags-in [f]
                  (concat (keep :tag (vals (:choices f)))
                          (mapcat tags-in (vals (:children f)))))
        [state renamed]
        (reduce (fn [[st m] old]
                  (let [n (:sym-n st)
                        new (symbol (str (first (str/split (str old) #"\$")) "$" n))
                        st (-> st
                               (assoc :sym-n (inc n))
                               (update :instrs conj {:i :alias :out new :arg old}))]
                    [st (assoc m old new)]))
                [state {}]
                (sort-by str (distinct (tags-in frame))))
        rename (fn rename [f]
                 (-> f
                     (update :choices #(reduce-kv (fn [acc k ch] (assoc acc k (update ch :tag (fn [t] (get renamed t t))))) {} %))
                     (update :children #(reduce-kv (fn [acc k c] (assoc acc k (rename c))) {} %))))]
    [state (rename frame)]))

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

(defn- query-deps
  "The deps of a query's value, value-deps the known choices its expression
  reads: every choice the frame knows (known-deps), or, with a budget, those
  its beliefs vary along (:ll-dims) and value-deps. A bounded agent shares
  samples along what its belief does not vary with (sample-ll), so the
  coarser set would give options samples of their own where nothing about
  them depends on the option (acceptance pass 3)."
  [state value-deps]
  (if-not (:bounded state)
    (known-deps state)
    (let [frm (fr/get-frame state)
          own (set value-deps)]
      (set (keep (fn [[addr c]] (when (and (:known c) (or (own addr) (contains? (:ll-dims frm) (:idx c)))) addr))
                 (:choices frm))))))

(defn- unknown-dims
  "The dims of the choices the current frame is uncertain about."
  [state]
  (vec (distinct (keep (fn [[_ c]] (when-not (:known c) (:idx c)))
                       (:choices (fr/get-frame state))))))

(defn- deps->dims
  "The dims, in frame frm, of the choices a value depends on."
  [frm deps]
  (set (keep #(get-in frm [:choices % :idx]) deps)))

(defn- read-ll
  "The current frame's belief array, for a computation that reads it. An
  observation that scrambled it (eval-show) makes that a compile error:
  Python memo crashes there, with a shape error."
  [state loc]
  (let [frm (fr/get-frame state)]
    (when-let [{:keys [message hint]} (:broken frm)]
      (err/raise message (assoc (frame-err-opts state loc) :hint hint)))
    (when-let [{:keys [source target]} (:scrambled frm)]
      (err/raise "Observation against an already-used value"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) "'s beliefs already varied with " source
                                   " when " (:name frm) " observed " target " to be it: "
                                   "memo represents an observed value by sharing its axis, "
                                   "and one axis cannot carry two of " (:name frm)
                                   "'s beliefs, so they are scrambled from there on. "
                                   "Condition on the equality instead: (" (:name frm)
                                   " :observes-that (= " target " " source ")). (Python memo "
                                   "fails here with a shape error.)"))))
    (:ll frm)))

(defn- mark-unreadable
  "frame with what is inside it (itself excluded) marked unreadable: each
  choice of another agent for which (stale? ch) holds, and each agent's own
  choice whose copy the frame holding that agent marked, as :unreadable;
  then each frame for which (broken? f) holds, as :broken; both (why
  holder), {:message :hint}, holder naming the frame directly inside frame
  that holds them. Reading either is then that compile error (eval-choice,
  read-ll), where Python memo would fail."
  [frame stale? broken? why]
  (letfn [(walk [f holder inherited]
            (let [marked (set (keep (fn [[[w id :as k] ch]]
                                      (when (if (= 'self w) (inherited id) (stale? ch)) k))
                                    (:choices f)))
                  f (-> f
                        (update :choices
                                #(reduce-kv (fn [acc k ch]
                                              (assoc acc k (cond-> ch
                                                             (and (marked k) (not (:unreadable ch)))
                                                             (assoc :unreadable (why holder)))))
                                            {} %))
                        (update :children
                                #(reduce-kv (fn [acc k c]
                                              (assoc acc k (walk c holder
                                                                 (set (keep (fn [[w id]] (when (= w k) id))
                                                                            marked)))))
                                            {} %)))]
              (cond-> f (and (broken? f) (not (:broken f))) (assoc :broken (why holder)))))]
    (update frame :children #(reduce-kv (fn [acc k c] (assoc acc k (walk c (:name c) #{}))) {} %))))

(defn- repoint
  "The state with every choice in the frame tree whose value is the symbol
  old given the value new instead. An observation does this to the observed
  choice: the frames that share its value (the observer's, the observed
  agent's, the knows-copies made earlier) see the observed value from here
  on, as Python memo's rebinding makes them. Copies made by snapshots and
  imagined worlds have values of their own, so they never share it."
  [state old new]
  (letfn [(walk [f]
            (-> f
                (update :choices #(reduce-kv (fn [acc k ch]
                                               (assoc acc k (if (= old (:tag ch)) (assoc ch :tag new) ch)))
                                             {} %))
                (update :children #(reduce-kv (fn [acc k c] (assoc acc k (walk c))) {} %))))]
    (update state :root walk)))

;; -----------------------------------------------------------------------------
;; bounded minds (M3, DESIGN.md §7): with a budget, a bounded agent's queries
;; read an empirical belief, k samples of what they depend on, and its choice
;; marginalizes the samples. The state then has :bounded (the normalized
;; budget), :limit (the call's), :pending (the samples drawn since the
;; innermost choice began), :mixing (the dims that choice's reasoning mixes:
;; its options, and what enclosing queries average over), :choosing (that
;; choice: {:owner :dims :mc-dim :draws}) and :report.

(defn- alloc-dim [state]
  (let [d (:next-dim state)] [(assoc state :next-dim (inc d)) d]))

(defn- frame-budget
  "The budget of the current frame's owner, or nil."
  [state]
  (when-let [budget (:bounded state)]
    (get budget (:owner (fr/get-frame state)))))

(defn- dim-domains
  "From each dim of the frame's choices to its domain."
  [frm]
  (into {} (map (fn [[_ c]] [(:idx c) (:domain c)])) (:choices frm)))

(defn- as-count
  "A number of states as a long where it fits (a bigint otherwise)."
  [n]
  (if (<= n Long/MAX_VALUE) (long n) n))

(defn- too-many-states
  "The loud error of a size above the limit. With declared, the agent's
  budget declares Monte Carlo, which this choice cannot use."
  ([what n limit opts] (too-many-states what n limit opts false))
  ([what n limit opts declared]
   (err/raise (str what " enumerates " n " states, above the limit of " limit)
              (assoc opts :hint (if declared
                                  (str "Monte Carlo is declared, but this choice is inside another agent's mind or "
                                       "inside a choice's weights, where memo.clj keeps it exact over its samples "
                                       "(an estimate there would bias the reasoning that uses it). Use fewer "
                                       "samples, sample fewer or smaller variables, or raise the limit (:limit in "
                                       "the agent's budget, or :memo/limit for the call).")
                                  (str "Exact over samples enumerates every way the samples can fall. Use fewer "
                                       "samples, sample fewer or smaller variables (a query samples only those it "
                                       "depends on), raise the limit (:limit in the agent's budget, or :memo/limit "
                                       "for the call), or declare Monte Carlo (:monte-carlo true)."))))))

(defn- sample-ll
  "A bounded agent's query: the empirical belief of k samples of the dims in
  sample-dims (the frame's uncertain dims the query depends on), the others
  summed out, on a new dim of draws: [state sym]. The draws are pending
  until the agent's choice marginalizes them (eval-choose). Each value of a
  dim the choice's reasoning mixes, and along which the belief varies, has
  samples of its own; so does each option, with :per-option."
  [state loc ll sample-dims spec what]
  (let [frm (fr/get-frame state)
        owner (:owner frm)
        {:keys [choosing]} state
        _ (when-not (and choosing (= owner (:owner choosing)))
            (err/raise (str owner "'s bounded estimate is used outside " owner "'s own choice")
                       (assoc (frame-err-opts state loc)
                              :hint (str owner " has a budget, so this " what " is estimated from samples, and "
                                         "in M3 such an estimate can only drive " owner "'s own choice (:wpp, "
                                         ":to-maximize or :to-be). Here it is returned, read by another agent, or weights "
                                         "an observation. To model a reported estimate, make the report a "
                                         "choice of " owner "'s, e.g. (" owner " :chooses [r R] :to-be ...)."))))
        unknown (unknown-dims state)
        U (vec (sort (filter (set unknown) sample-dims)))
        R (vec (remove (set U) unknown))
        doms (dim-domains frm)
        size (fn [d] (count (domain-values state (doms d) loc)))
        known (set (keep (fn [[_ c]] (when (:known c) (:idx c))) (:choices frm)))
        mixing (set (:mixing state))
        X (vec (sort (into (set/intersection (:ll-dims frm) known mixing)
                           (when (:per-option spec) (set/intersection known (set (:dims choosing)))))))
        ;; with nothing observed, rejection keeps every sample: one-and-done's
        rejection? (and (= :rejection (:algorithm spec)) (some? (:evidence frm)))
        k (:samples spec)
        m (reduce * 1 (map size U))
        n (reduce * 1 (map size X))
        mprime (if rejection? (inc m) m)
        mc? (some? (:mc-dim choosing))
        states (if mc? (:draws spec) (b/power (b/states k mprime) n))
        limit (b/limit-of spec (:limit state))
        _ (when (and (not mc?) (> states limit) (not (:report-only state)))
            (too-many-states (str owner "'s " what " from " k " samples") states limit (frame-err-opts state loc)
                             (:monte-carlo spec)))
        [state omega] (if mc? [state (:mc-dim choosing)] (alloc-dim state))
        [state p] (cond
                    (nil? ll) (let [[st one] (fresh state "ll")] [(emit st {:i :const :out one :value 1.0}) one])
                    (empty? R) [state ll]
                    :else (let [[st o] (fresh state "belief")] [(emit st {:i :sum :out o :arg ll :dims R}) o]))
        z (when rejection? (:evidence frm))
        instr (cond-> {:arg p :omega omega :sample U :sample-sizes (mapv size U) :indep X
                       :indep-sizes (mapv size X)
                       :shared (vec (sort (set/difference (set/intersection mixing known) (set X))))
                       :k k :algorithm (if rejection? :rejection :one-and-done)
                       :draws (when mc? (:draws spec)) :who owner :loc loc}
                z (assoc :evidence z)
                (and rejection? (seq (:event-weights frm))) (assoc :weights (vec (:event-weights frm))))
        [state q] (fresh state "sampled")
        state (emit state (assoc instr :i :sample-belief :out q))
        [state w] (if mc?
                    [state nil]
                    (let [[st w] (fresh state "draws")] [(emit st (assoc instr :i :sample-weight :out w)) w]))]
    [(update state :pending conj {:kind :samples :owner owner :omega omega :weight w :states (as-count states)
                                  :what what :line (:line loc) :k k :categories mprime :sets n})
     q]))

(defn- record-mass
  "With :check-mass (a return's index, while the analysis asked for masses
  evaluates that return outside any query or statement), the total of the
  belief ll a query weighs by, over the frame's unknown dims, recorded in
  :masses with the frame, the query and n, the cells summed. It is 1 unless
  the belief lost its mass (an impossible observation, a choice whose
  weights are all zero), which memo leaves silently: memo.gen/deterministic
  checks it (memoclj-wbhh)."
  [state ll loc what]
  (let [frm (fr/get-frame state)
        cells (reduce * 1.0 (vals (into {} (keep (fn [[_ c]]
                                                   (when-not (:known c)
                                                     [(:idx c) (count (get-in state [:domains (:domain c)]))])))
                                         (:choices frm))))
        [state m] (fresh state "mass")
        state (emit state {:i :sum :out m :arg ll :dims (unknown-dims state)})]
    (update state :masses (fnil conj []) {:tag m :frame (:name frm) :what what :loc loc :cells cells
                                          :return (:check-mass state)})))

(defn- query-ll
  "The belief a query reads, [state sym-or-nil]: the frame's ll (read-ll), or,
  in a bounded agent's frame, an empirical belief from samples of the dims in
  sample-dims (sample-ll). An exact belief read for a return has its mass
  recorded (record-mass)."
  [state loc sample-dims what]
  (let [ll (read-ll state loc)
        spec (frame-budget state)]
    (if (and (:samples spec) (not (:exact state)))
      (sample-ll state loc ll sample-dims spec what)
      [(cond-> state (and ll (:check-mass state)) (record-mass ll loc what)) ll])))

(defn- add-evidence
  "With a budget, the frame's evidence (the probability of its observations,
  what a rejection sampler keeps) times the normalizer of a new observation."
  [state total]
  (if-not (:bounded state)
    state
    (if-let [ev (:evidence (fr/get-frame state))]
      (let [[st out] (fresh state "evidence")]
        (fr/update-frame (emit st {:i :op :op :mul :out out :args [ev total]}) assoc :evidence out))
      (fr/update-frame state assoc :evidence total))))

;; -----------------------------------------------------------------------------
;; expressions

(declare eval-expr eval-stmt)

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
    (when-let [{:keys [message hint]} (:unreadable ch)]
      (err/raise message (assoc (frame-err-opts state (:loc e)) :hint hint)))
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

(defn- branch-masses
  "With :check-mass, the masses recorded at indices [from, to), in a branch
  of an if with condition c (the true branch when taken?), counted as 1
  where the branch is not taken: an if computes both branches, and a query
  in the one not taken reaches no value. A query in c is checked as it is."
  [state from to c taken?]
  (if-not (and (:check-mass state) (< from to))
    state
    (let [[state one] (fresh state "lit")
          state (emit state {:i :const :out one :value 1.0})]
      (reduce (fn [st i]
                (let [m (get-in st [:masses i :tag])
                      [st out] (fresh st "mass")]
                  (-> (emit st {:i :where :out out :args (if taken? [c m one] [c one m])})
                      (assoc-in [:masses i :tag] out))))
              state (range from to)))))

(defn- eval-op [state e]
  (let [{:keys [op args]} e]
    (cond
      (contains? #{:add :sub :mul :div :pow :mod :eq :neq :lt :lte :gt :gte
                   :and :or :xor} op)
      (let [[state l] (eval-expr state (first args))
            [state r] (eval-expr state (second args))
            [state out] (fresh state (str "op_" (name op)))]
        [(emit state (cond-> {:i :op :op op :out out :args [(:tag l) (:tag r)]}
                       (:what e) (assoc :what (:what e))))
         {:tag out :known (and (:known l) (:known r)) :deps (into (:deps l) (:deps r))}])

      (contains? #{:exp :abs :log :uadd :neg :inv} op)
      (let [[state l] (eval-expr state (first args))
            [state out] (fresh state (str "op_" (name op)))]
        [(emit state {:i :op :op op :out out :args [(:tag l)]})
         {:tag out :known (:known l) :deps (:deps l)}])

      (= op :ite)
      (let [[c-ast t-ast f-ast] args
            [state c] (eval-expr state c-ast)
            n0 (count (:masses state))
            ;; a static condition guards the memo calls in its branches
            ;; (path conditions, as core.py's path_depends)
            [state tv fv n1]
            (if (:static c-ast)
              (let [state (update state :path-conds conj (:tag c))
                    [state tv] (eval-expr state t-ast)
                    n1 (count (:masses state))
                    state (update state :path-conds pop)
                    [state inv] (fresh state "op_inv")
                    state (emit state {:i :op :op :inv :out inv :args [(:tag c)]})
                    state (update state :path-conds conj inv)
                    [state fv] (eval-expr state f-ast)
                    state (update state :path-conds pop)]
                [state tv fv n1])
              (let [[state tv] (eval-expr state t-ast)
                    n1 (count (:masses state))
                    [state fv] (eval-expr state f-ast)]
                [state tv fv n1]))
            state (-> state
                      (branch-masses n0 n1 (:tag c) true)
                      (branch-masses n1 (count (:masses state)) (:tag c) false))
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
  [state {:keys [query var loc]}]
  (let [frm (fr/get-frame state)
        var-idxs (set (map #(get-in frm [:choices % :idx]) var))
        [state ll] (query-ll state loc var-idxs "Pr")
        marg-dims (vec (distinct (keep (fn [[_ c]]
                                         (when-not (or (:known c) (var-idxs (:idx c)))
                                           (:idx c)))
                                       (:choices frm))))
        [state out] (fresh state "posterior")
        state (emit state {:i :sum :out out :arg ll :dims marg-dims})
        [state out] (reduce (fn [[st prev] [knw-addr var-addr]]
                              (let [[st o] (fresh st "posterior")]
                                [(emit st {:i :posterior-align :out o :arg prev
                                           :knw (get-in frm [:choices knw-addr :idx])
                                           :var (get-in frm [:choices var-addr :idx])})
                                 o]))
                            [state out]
                            (map vector query var))]
    [state {:tag out :known true :deps (query-deps state query)}]))

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
      (eval-posterior state {:query (mapv first pairs) :var (mapv second pairs) :loc (:loc e)})
      (let [armed (:check-mass state)
            ;; a query inside this one is weighed by it: its mass is not checked
            state (assoc state :check-mass nil)
            [state val] (if (:bounded state)
                          ;; what this query averages over is mixed by it
                          (let [mixing (:mixing state)
                                [st v] (eval-expr (update state :mixing (fnil into []) (unknown-dims state)) expr)]
                            [(assoc st :mixing mixing) v])
                          (eval-expr state expr))
            state (assoc state :check-mass armed)
            frm (fr/get-frame state)
            all-known? (every? #(get-in frm [:choices % :known]) (:deps val))]
        (when (and all-known? warn)
          (err/warn-at (:loc e) "Redundant " (name reduction) ", not marginalizing..."))
        ;; a redundant expectation is the value itself, without the frame's
        ;; beliefs: after an impossible observation that is not the same
        ;; number (DESIGN.md §2.4), so it is semantics, not an optimization.
        ;; A redundant variance is zeros like the value (core.py).
        (cond
          (and all-known? (= reduction :expectation))
          [state val]

          all-known?
          (let [[state zero] (fresh state "lit")
                state (emit state {:i :const :out zero :value 0})
                [state out] (fresh state "var")]
            [(emit state {:i :op :op :mul :out out :args [(:tag val) zero]})
             {:tag out :known true :deps (:deps val)}])

          :else
          (let [dims (unknown-dims state)
                [state frame-ll] (query-ll state (:loc e)
                                           (deps->dims frm (remove #(get-in frm [:choices % :known]) (:deps val)))
                                           (str (or (:surface e) 'E)))
                [state ll] (if frame-ll
                             [state frame-ll]
                             (let [[st one] (fresh state "ll")]
                               [(emit st {:i :const :out one :value 1.0}) one]))
                [state lv] (fresh state "llval")
                state (emit state {:i :op :op :mul :out lv :args [ll (:tag val)]})]
            (case reduction
              :expectation
              (let [[state out] (fresh state "exp")]
                [(emit state {:i :sum :out out :arg lv :dims dims})
                 {:tag out :known true :deps (query-deps state (:deps val))}])
              ;; Python memo's E[X^2] - E[X]^2, over the frame's weights w of
              ;; total mass M (below 1 where a branch has zero mass), computed
              ;; as sum w (X - m)^2 + m^2 (1 - M), m = E[X]: the same number,
              ;; without float32's cancellation when M is 1 (the fuzzer found
              ;; that the centered form alone differs when M < 1). X is made
              ;; a float first: JAX has no - on booleans, and Python memo
              ;; never subtracts X itself.
              :variance
              (let [[state two] (fresh state "lit")
                    state (emit state {:i :const :out two :value 2})
                    [state mean] (fresh state "exp")
                    state (emit state {:i :sum :out mean :arg lv :dims dims})
                    [state one] (fresh state "lit")
                    state (emit state {:i :const :out one :value 1.0})
                    [state x] (fresh state "x")
                    state (emit state {:i :op :op :mul :out x :args [(:tag val) one]})
                    [state dev] (fresh state "dev")
                    state (emit state {:i :op :op :sub :out dev :args [x mean]})
                    [state sq] (fresh state "sq")
                    state (emit state {:i :op :op :pow :out sq :args [dev two]})
                    [state lv2] (fresh state "llval")
                    state (emit state {:i :op :op :mul :out lv2 :args [ll sq]})
                    [state centered] (fresh state "var")
                    state (emit state {:i :sum :out centered :arg lv2 :dims dims})]
                (if-not frame-ll
                  [state {:tag centered :known true :deps (query-deps state (:deps val))}]
                  (let [[state mass] (fresh state "mass")
                        state (emit state {:i :sum :out mass :arg ll :dims dims})
                        [state missing] (fresh state "missing")
                        state (emit state {:i :op :op :sub :out missing :args [one mass]})
                        [state m2] (fresh state "sq")
                        state (emit state {:i :op :op :pow :out m2 :args [mean two]})
                        [state corr] (fresh state "corr")
                        state (emit state {:i :op :op :mul :out corr :args [m2 missing]})
                        [state out] (fresh state "var")]
                    [(emit state {:i :op :op :add :out out :args [centered corr]})
                     {:tag out :known true :deps (query-deps state (:deps val))}]))))))))))

(def ^:private not-static-hint
  (str "When calling a memo, you can only pass in parameters that are fixed (\"static\") "
       "values that memo can compute without reasoning about agents. Such values cannot "
       "depend on any agents' choices -- only on literal numeric values and other "
       "parameters. This constraint is what enables memo to help you fit/optimize "
       "parameters fast by gradient descent."))

(defn- eval-memo-call
  "(f [x agent.y] args...): the memo f's table, with its axes on the dims of
  the choices given. Its arguments are static, so they are known before any
  agent reasons; it is called only when the path conditions around it hold,
  which is what lets a memo recurse."
  [state {:keys [name which-retval args kwargs ids ellipsis? loc]}]
  (let [[state arg-vals]
        (reduce (fn [[st acc] a]
                  (let [[st v] (eval-expr st a)]
                    (when-not (:static a)
                      (err/raise "parameter not statically known"
                                 (assoc (frame-err-opts st (:loc a)) :hint not-static-hint)))
                    [st (conj acc v)]))
                [state []] args)
        [state kw-vals]
        (reduce (fn [[st acc] [k a]]
                  (let [[st v] (eval-expr st a)]
                    (when-not (:static a)
                      (err/raise (str "keyword argument " k " not statically known")
                                 (assoc (frame-err-opts st (:loc a)) :hint not-static-hint)))
                    [st (conj acc [k (:tag v)])]))
                [state []] kwargs)
        frm (fr/get-frame state)
        chs (mapv (fn [[_ sw si]]
                    (or (get-in frm [:choices [sw si]])
                        (err/raise "Unknown choice referenced in a memo call"
                                   (assoc (frame-err-opts state loc)
                                          :hint (str (:name frm) " does not yet model " sw
                                                     "'s choice of " si ".")))))
                  ids)
        dims (mapv :idx chs)
        ;; a memo defined before this one: its axes and parameters are known
        ;; (Python memo checks them when called; a return index is checked
        ;; when called here too, as Python memo's tests expect)
        _ (when-let [{:keys [axes params defaults num-results]} (some-> (:callee state) (#(% name)))]
            (let [opts (fn [hint] (assoc (frame-err-opts state loc) :hint hint))
                  sig (str name " has " axes " axes and the parameters " (pr-str params) ".")
                  named (set (map (comp symbol clojure.core/name first) kwargs))]
              (cond
                (> (count ids) axes) (err/raise "Too many arguments to memo call!" (opts sig))
                (< (count ids) axes) (err/raise "Not enough arguments to memo call!" (opts sig)))
              ;; Python memo checks the return index first, when called
              (when (and (not ellipsis?)
                         (if (= 1 num-results)
                           (nil? which-retval)
                           (and (some? which-retval) (< -1 which-retval num-results))))
                (when (> (count args) (count params))
                  (err/raise (str "Too many arguments to memo " name " (it takes " (count params) ")")
                             (opts sig)))
                (when-let [k (first (remove (set params) named))]
                  (err/raise (str "memo " name " has no parameter " k) (opts sig)))
                (when-let [p (first (filter named (take (count args) params)))]
                  (err/raise (str "Parameter " p " of memo " name " passed both positionally and by name")
                             (opts sig)))
                (when-let [p (first (remove #(or (named %) (defaults %)) (drop (count args) params)))]
                  (err/raise (str "Missing parameter " p " in call to memo " name) (opts sig))))))
        _ (when (not= (count (set dims)) (count dims))
            (err/raise "The same choice is passed twice in a memo call"
                       (assoc (frame-err-opts state loc)
                              :hint "Each axis of the memo called needs a choice of its own.")))
        [state out] (fresh state (str "result_" name))]
    [(emit state {:i :call-memo :out out :name name :which which-retval
                  :args (mapv :tag arg-vals) :kwargs kw-vals
                  :guards (:path-conds state) :dims dims :domains (mapv :domain chs)})
     {:tag out
      :known (every? :known chs)
      :deps (set (map (fn [[_ sw si]] [sw si]) ids))}]))

(defn- eval-ffi
  "(f args...): the host function f at every element of its dynamic
  arguments. Known when all its arguments are; static when all are static."
  [state {:keys [name args kwargs]}]
  (let [[state arg-vals] (reduce (fn [[st acc] a] (let [[st v] (eval-expr st a)] [st (conj acc v)]))
                                 [state []] args)
        [state kw-vals] (reduce (fn [[st acc] [k a]]
                                  (let [[st v] (eval-expr st a)] [st (conj acc [k v (:static a)])]))
                                [state []] kwargs)
        vals (into arg-vals (map second kw-vals))
        [state out] (fresh state (str "ffi_" name))]
    [(emit state {:i :ffi :out out :name name
                  :args (mapv :tag arg-vals) :statics (mapv :static args)
                  :kwargs (mapv (fn [[k v static]] [k (:tag v) static]) kw-vals)
                  :guards (:path-conds state)})
     {:tag out :known (every? :known vals) :deps (reduce into #{} (map :deps vals))}]))

(defn- eval-inline [state {:keys [idx]}]
  (let [[state out] (fresh state "inline")]
    [(emit state {:i :inline :out out :idx idx})
     {:tag out :known true :deps #{}}]))

(defn- emit-op
  "Emit {:i :op :op op :args args} under a fresh symbol: [state out]."
  [state hint op & args]
  (let [[state out] (fresh state hint)]
    [(emit state {:i :op :op op :out out :args (vec args)}) out]))

(defn- eval-entropy
  "(H a.x ...): the entropy of the frame's joint marginal over the choices
  named (core.py's EEntropy). nan_to_num makes 0 · log 0 zero."
  [state {:keys [rvs loc]}]
  (let [frm (fr/get-frame state)
        named (set rvs)]
    (doseq [[w i :as rv] rvs]
      (let [c (get-in frm [:choices rv])]
        (when-not c
          (err/raise "Taking entropy of unknown variable"
                     (assoc (frame-err-opts state loc)
                            :hint (str (:name frm) " is not tracking choice " w "." i "."))))
        (when (:known c)
          (err/raise "Taking entropy of known variable"
                     (assoc (frame-err-opts state loc)
                            :hint (str w "." i " is already known to " (:name frm)
                                       ", so its entropy is zero."))))))
    (let [[state ll] (query-ll state loc (keep #(get-in frm [:choices % :idx]) rvs) "H")
          dims (fn [pred] (vec (distinct (keep (fn [[addr c]] (when (and (not (:known c)) (pred addr)) (:idx c)))
                                               (:choices frm)))))
          [state m] (fresh state "marginal")
          state (emit state {:i :sum :out m :arg ll :dims (dims (complement named))})
          [state lg] (emit-op state "log" :log m)
          [state z] (fresh state "log")
          state (emit state {:i :nan-to-num :out z :arg lg})
          [state t] (emit-op state "plogp" :mul m z)
          [state s] (fresh state "sum")
          state (emit state {:i :sum :out s :arg t :dims (dims named)})
          [state out] (emit-op state "entropy" :neg s)]
      [state {:tag out :known true :deps (query-deps state [])}])))

(defn- eval-kl
  "(KL p.x q.y): the KL divergence from the frame's marginal over q.y to its
  marginal over p.x, two choices over one domain (core.py's EKL)."
  [state {:keys [p-who p-id q-who q-id loc]}]
  (let [frm (fr/get-frame state)
        p (get-in frm [:choices [p-who p-id]])
        q (get-in frm [:choices [q-who q-id]])
        err-opts (frame-err-opts state loc)]
    (doseq [[c w i] [[p p-who p-id] [q q-who q-id]]]
      (when-not c
        (err/raise (str "Unknown choice " w "." i) (assoc err-opts :hint "Did you misspell something?"))))
    (doseq [[c w i] [[p p-who p-id] [q q-who q-id]]]
      (when (:known c)
        (err/raise (str "Cannot take KL-divergence over known variable " w "." i)
                   (assoc err-opts :hint "It only makes sense to take KL-divergence over uncertain variables."))))
    (when (not= (:domain p) (:domain q))
      (err/raise "KL-divergence mismatched support"
                 (assoc err-opts :hint (str "Domains of " p-who "." p-id " (" (:domain p) ") and "
                                            q-who "." q-id " (" (:domain q) ") do not match."))))
    (let [[state ll] (query-ll state loc [(:idx p) (:idx q)] "KL")
          others (fn [c] (vec (distinct (keep (fn [[_ c']] (when (and (not (:known c')) (not= c' c)) (:idx c')))
                                              (:choices frm)))))
          [state pp] (fresh state "p")
          state (emit state {:i :sum :out pp :arg ll :dims (others p)})
          [state qp] (fresh state "q")
          state (emit state {:i :sum :out qp :arg ll :dims (others q)})
          [state qs] (fresh state "q")
          state (emit state {:i :swap-dims :out qs :arg qp :d1 (:idx p) :d2 (:idx q)})
          [state lp] (emit-op state "log" :log pp)
          [state lq] (emit-op state "log" :log qs)
          [state d] (emit-op state "logratio" :sub lp lq)
          [state z] (fresh state "logratio")
          state (emit state {:i :nan-to-num :out z :arg d})
          [state t] (emit-op state "term" :mul pp z)
          [state out] (fresh state "kl")
          state (emit state {:i :sum :out out :arg t :dims [(:idx p)]})]
      [state {:tag out :known true :deps (query-deps state [])}])))

(defn- eval-imagine
  "(imagine stmts... e): e in a hypothetical world, the current frame's copy
  (named imagined_<name>, with value symbols of its own) after stmts. The
  copy is discarded afterwards; e's value must be known to the frame."
  [state {:keys [do then loc]}]
  (let [who (:name (fr/get-frame state))
        {old-root :root old-path :path old-cont :continuation} state
        old-frame (fr/get-frame state)
        imagined-name (symbol (str "imagined_" who))
        state (fr/update-frame state assoc :name imagined-name)
        [state imagined] (fresh-tags state (fr/get-frame state))
        state (fr/update-frame state (constantly imagined))
        state (reduce (fn [st [i s]] (eval-stmt (assoc st :continuation (subvec do (inc i))) s))
                      state (map-indexed vector do))
        [state val] (eval-expr state then)]
    (when-not (:known val)
      (err/raise "trying to imagine an unknown value"
                 (assoc (frame-err-opts state (:loc then))
                        :hint (str "In this hypothetical imagined world, " who " won't be able "
                                   "to compute the return value you requested. Perhaps you "
                                   "meant to wrap the return value in (E ...)?"))))
    (let [imagined-frame (fr/get-frame state)
          deps (reduce (fn [acc dep]
                         (let [c (get (:conditions imagined-frame) dep)]
                           (cond
                             (contains? (:choices old-frame) dep) (conj acc dep)
                             ;; a statement like (env :knows x) in the copy
                             (= imagined-name (first c)) (conj acc ['self (second c)])
                             :else (err/raise "Internal error mapping imagined dependencies"
                                              {:user false :loc loc}))))
                       #{} (:deps val))]
      [(assoc state :root old-root :path old-path :continuation old-cont)
       (assoc val :deps deps)])))

(defn- predict-prefix
  "The frame names from the outermost one below the root (or an imagined
  frame) down to the current one: the agents whose statements a Predict
  replays are the last of them, as modeled by the others."
  [state]
  (loop [i (count (:path state)), prefix ()]
    (if (zero? i)
      (vec prefix)
      (let [f (fr/get-frame state (subvec (:path state) 0 i))]
        (if (str/starts-with? (str (:name f)) "imagined_")
          (vec prefix)
          (recur (dec i) (cons (:name f) prefix)))))))

(defn- retarget-continuation
  "The rest of the program, as the current agent's future self would live it:
  the agent's own statements (inside the thinks of the agents modeling it),
  addressed to the future self, with each later observation becoming a guess
  of the value from the agent's current beliefs and an observation of the
  guess (core.py's EPredict)."
  [state prefix future loc]
  (reduce
   (fn [[st acc] stmt]
     (let [s (loop [s stmt, ws (butlast prefix)]
               (cond
                 (empty? ws) s
                 (and (= :with-stmt (:node s)) (= (:who s) (first ws))) (recur (:stmt s) (rest ws))
                 :else nil))]
       (cond
         (nil? s) [st acc]
         (= :pass (:node s)) [st (conj acc s)]
         (= :forall (:node s))
         (err/raise "forall cannot appear after a wants statement or a (Predict ...) expression" {:loc loc})
         (not= (:who s) (last prefix)) [st acc]
         (= :knows (:node s))
         (err/raise "A 'knows' statement cannot appear after a 'wants' statement or a 'Predict' expression."
                    (assoc (frame-err-opts st loc)
                           :hint (str "This is because " (last prefix) " cannot evaluate utilities "
                                      "without knowing all of the relevant variables. Can you move "
                                      "the knows statement before the wants statement?")))
         (= :show (:node s))
         (let [[st id] (fresh st (str (:target-id s)))]
           [st (conj acc
                     {:node :guess :who future :id id :target-who (:target-who s)
                      :target-id (:target-id s) :loc (:loc s)}
                     (assoc s :who future :source-who future :source-id id))])
         :else [st (conj acc (assoc s :who future))])))
   [state []]
   (:continuation state)))

(defn- eval-predict
  "(Predict e): the expected value of e in the current agent's imagined
  future, the rest of the program replayed by a snapshot of the agent."
  [state {:keys [expr loc surface]}]
  (when (empty? (:path state))
    (err/raise "(Predict ...) and (EU ...) cannot be used at the top level of a memo"
               {:hint (str "They evaluate an agent's goal in its imagined future, and the "
                           "observer has none. Use them in an agent's choice, e.g. "
                           "(alice :chooses [a A] :to-maximize (EU goal)).")
                :loc loc}))
  (let [n (:sym-n state)
        ;; a frame's name, as Python memo's ctxt.sym makes it (no $)
        [state future] [(assoc state :sym-n (inc n))
                        (symbol (str "future_" (:name (fr/get-frame state)) "_" n))]
        path (:path state)
        state (-> state
                  (assoc :path (pop path))
                  (eval-stmt {:node :snapshot :who (peek path) :alias future :loc loc})
                  (assoc :path path))
        [state to-imagine] (retarget-continuation state (predict-prefix state) future loc)]
    (eval-expr state {:node :imagine :do to-imagine :loc loc :static false
                      :then {:node :expect :reduction :expectation :warn false :loc loc :static false
                             :surface (or surface "Predict")
                             :expr {:node :with :who future :expr expr :loc loc :static false}}})))

(defn- eval-util
  "(EU goal): the expected value of the agent's goal, as Predict."
  [state {:keys [goal loc]}]
  (let [g (get-in (fr/get-frame state) [:goals goal])]
    (when-not g
      (err/raise (str "Unknown goal " goal)
                 (assoc (frame-err-opts state loc) :hint (str "Did you misspell " goal "?"))))
    (eval-predict state {:expr g :loc loc :surface "EU"})))

(defn eval-expr [state e]
  (case (:node e)
    :lit (eval-lit state e)
    :choice (eval-choice state e)
    :op (eval-op state e)
    :with (eval-with state e)
    :expect (eval-expect state e)
    :ffi (eval-ffi state e)
    :memo (eval-memo-call state e)
    :entropy (eval-entropy state e)
    :kl (eval-kl state e)
    :imagine (eval-imagine state e)
    :inline (eval-inline state e)
    :predict (eval-predict state e)
    :util (eval-util state e)
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
        state (fr/update-frame state fr/put-choice ['self id]
                               {:tag tag :idx idx :known true :domain domain})]
    (update state :forall conj [idx id domain])))

(def ^:private considered-specs
  "The choices whose options a budget's :options narrows: an agent's own
  :chooses by weight or value, not a :given, :draws, :assigned or :guesses,
  nor a :to-be or :such-that."
  #{:wpp :uniformly :to-maximize :to-minimize})

(defn- options-source
  "With :options k in the budget of who, k of the choice's options considered,
  drawn uniformly: [state mask-or-nil], the mask (true where considered) on a
  new dim of draws, pending as a source."
  [state {:keys [spec owner mc-dim dims sizes surface-alias surface-spec loc]}]
  (let [k (:options spec)
        cells (reduce * 1 sizes)]
    (cond
      (not (and k (= :chooses surface-alias) (considered-specs surface-spec))) [state nil]
      ;; every option considered: the budget reaches the choice, and changes nothing
      (>= k cells) [(update state :all-considered (fnil conj #{}) owner) nil]
      :else
      (let [mc? (some? mc-dim)
            states (if mc? (:draws spec) (b/binomial cells k))
            limit (b/limit-of spec (:limit state))
            _ (when (and (not mc?) (> states limit) (not (:report-only state)))
                (too-many-states (str owner " considering " k " of " cells " options") states limit {:loc loc}
                                 (:monte-carlo spec)))
            [state sigma] (if mc? [state mc-dim] (alloc-dim state))
            [state mask] (fresh state "considered")
            state (emit state {:i :subsets :out mask :omega sigma :dims dims :sizes sizes :k k
                               :draws (when mc? (:draws spec))})
            [state w] (if mc?
                        [state nil]
                        (let [[st w] (fresh state "draws")]
                          [(emit st {:i :const :out w :value (/ 1.0 (double states))}) w]))]
        [(update state :pending conj {:kind :options :owner owner :omega sigma :weight w :states (as-count states)
                                      :line (:line loc) :k k :options cells})
         mask]))))

(defn- masked
  "ll where the mask is true, value elsewhere: [state sym]."
  [state ll mask value]
  (let [[state c] (fresh state "lit")
        state (emit state {:i :const :out c :value value})
        [state out] (fresh state "w")]
    [(emit state {:i :where :out out :args [mask ll c]}) out]))

(defn- marginalize
  "nrm, a bounded agent's choice probabilities given its draws, averaged over
  the draws of sources (its samples and options considered): the sum over
  their dims of nrm times their probabilities, or, by Monte Carlo, the mean
  over the draws. [state sym]; nrm when there are no sources."
  [state nrm sources {:keys [owner spec mc-dim loc]}]
  (if (empty? sources)
    [state nrm]
    (let [mc? (some? mc-dim)
          states (if mc? (:draws spec) (reduce * 1N (map :states sources)))
          limit (b/limit-of spec (:limit state))
          _ (when (and (not mc?) (> states limit) (not (:report-only state)))
              (too-many-states (str owner "'s choice") states limit {:loc loc} (:monte-carlo spec)))
          [state prod] (reduce (fn [[st acc] w] (emit-op st "drawn" :mul acc w))
                               [state nrm] (keep :weight sources))
          [state out] (fresh state "bounded")
          state (emit state {:i :sum :out out :arg prod :dims (vec (distinct (map :omega sources)))})
          [state out] (if mc?
                        (let [[st c] (fresh state "lit")
                              st (emit st {:i :const :out c :value (/ 1.0 (double (:draws spec)))})]
                          (emit-op st "bounded" :mul out c))
                        [state out])]
      [(update state :report conj
               (cond-> {:agent owner :line (:line loc) :states (as-count states) :monte-carlo mc?
                        :sources (mapv #(dissoc % :owner :omega :weight) sources)}
                 (not mc?) (assoc :limit limit)
                 (and (not mc?) (> states limit)) (assoc :over-limit true)))
       out])))

(defn- eval-choose
  "who chooses: the choices join who's own frame (known to who) and the
  current frame (uncertain), and the current frame's beliefs are multiplied
  by the choice's normalized weights. With a budget for who, the samples its
  weights were computed from, and the options it considers, are drawn here
  and marginalized (marginalize)."
  [state {:keys [who choices wpp reduction loc surface-alias surface-spec]}]
  (let [state (fr/ensure-child state who)
        owner (get-in (fr/get-frame state) [:children who :owner])
        spec (get (:bounded state) owner)
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
                 st (fr/update-frame st fr/put-choice ['self id]
                                     {:tag tag :idx idx :known true :domain dom})]
             [st (conj dims idx)]))
         [state []]
         choices)
        ;; with a budget, the choice opens a scope for the samples drawn while
        ;; its weights are computed (sample-ll)
        outer (select-keys state [:pending :mixing :choosing])
        ;; Monte Carlo draws only at a choice the observer models directly,
        ;; outside any choice's weights: one inside another agent's mind, or
        ;; inside a choice's weights (a future self's, replayed by EU), is
        ;; exact over its samples, since the reasoning above samples from, or
        ;; weighs by, its probabilities, and an estimate there would bias it
        ;; (acceptance passes 2 and 3)
        ;; (the path is the chooser's frame here: the observer's own models
        ;; are one deep)
        [state mc-dim] (if (and (:bounded state) (:monte-carlo spec) (nil? (:choosing state))
                                (= 1 (count (:path state))))
                         (alloc-dim state)
                         [state nil])
        state (if (:bounded state)
                (assoc state :pending [] :mixing dims
                       :choosing {:owner owner :dims dims :mc-dim mc-dim})
                state)
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
              (err/warn-at loc "When " who " chooses " id ", the probability doesn't depend "
                        "on the value of " id ". As a result, the choice will "
                        "effectively be uniform over " id " after normalization. "
                        "Are you sure this is what you want? (A uniform choice is "
                        "more easily expressed with :uniformly.)")))
        state (fr/leave state)
        state (reduce (fn [st [id dom]]
                        (let [ch (get-in (fr/get-frame st) [:children who :choices ['self id]])]
                          (fr/update-frame st fr/put-choice [who id]
                                           {:tag (:tag ch) :idx (:idx ch)
                                            :known false :domain dom})))
                      state
                      choices)
        sizes (mapv (fn [[_ dom]] (count (domain-values state dom loc))) choices)
        ;; the weights, with extent along every chosen dim
        [state ll] (reduce (fn [[st acc] [size dim]]
                             (let [[st o] (fresh st "ones")
                                   st (emit st {:i :ones :out o :dim dim :size size})
                                   [st m] (fresh st "w")]
                               [(emit st {:i :op :op :mul :out m :args [acc o]}) m]))
                           [state (:tag w)]
                           (map vector sizes dims))
        [state mask] (if spec
                       (options-source state {:spec spec :owner owner :mc-dim mc-dim :dims dims :sizes sizes
                                              :surface-alias surface-alias :surface-spec surface-spec :loc loc})
                       [state nil])
        [state ll] (if softmax?
                     (let [[st lm] (if mask (masked state ll mask Double/NEGATIVE_INFINITY) [state ll])
                           [st mx] (fresh st "max")
                           st (emit st {:i :max :out mx :arg lm :dims dims})
                           [st sh] (fresh st "shift")
                           st (emit st {:i :op :op :sub :out sh :args [lm mx]})
                           [st ex] (fresh st "softmax")]
                       [(emit st {:i :op :op :exp :out ex :args [sh]}) ex])
                     [state ll])
        ;; argmax: the values equal to the maximum, exactly (core.py)
        [state ll] (if (= reduction :maximize)
                     (let [[st lm] (if (and mask (not softmax?)) (masked state ll mask Double/NEGATIVE_INFINITY) [state ll])
                           [st mx] (fresh st "max")
                           st (emit st {:i :max :out mx :arg lm :dims dims})
                           [st eq] (fresh st "argmax")
                           st (emit st {:i :op :op :eq :out eq :args [lm mx]})]
                       (if mask (emit-op st "argmax" :and eq mask) [st eq]))
                     [state ll])
        [state ll] (if (and mask (not softmax?) (not= reduction :maximize)) (masked state ll mask 0) [state ll])
        [state total] (fresh state "total")
        state (emit state {:i :sum :out total :arg ll :dims dims})
        [state q] (fresh state "ratio")
        state (emit state {:i :op :op :div :out q :args [ll total]})
        [state nrm] (fresh state "p")
        state (emit state {:i :nan-to-num :out nrm :arg q})
        [state nrm] (marginalize state nrm (:pending state) {:owner owner :spec spec :mc-dim mc-dim :loc loc})
        state (if (:bounded state)
                (assoc state :pending (:pending outer) :mixing (:mixing outer) :choosing (:choosing outer))
                state)
        frm (fr/get-frame state)
        [state prev] (if (:ll frm)
                       [state (:ll frm)]
                       (let [[st one] (fresh state "ll")]
                         [(emit st {:i :const :out one :value 1.0}) one]))
        [state new-ll] (fresh state "ll")
        state (emit state {:i :op :op :mul :out new-ll :args [nrm prev]})
        ;; the beliefs now vary along the chosen dims and every dim the
        ;; weights depended on
        ll-dims (into (set dims) (deps->dims (get-in (fr/get-frame state) [:children who])
                                             (:deps w)))]
    (fr/update-frame state #(-> % (assoc :ll new-ll) (update :ll-dims into ll-dims)))))

(declare eval-stmt)

(defn- eval-with-stmt
  "(who :thinks stmt): stmt, in who's frame as the current frame models it."
  [state {:keys [who stmt loc]}]
  (let [frm (fr/get-frame state)]
    (case (:variant frm)
      :child (err/raise "Using `thinks` with variant CHILD"
                        (assoc (frame-err-opts state loc)
                               :hint (str (:name frm) " has variant CHILD, which does not have "
                                          "a `thinks` statement. Hence, " (:name frm)
                                          " cannot model (" who " :thinks ...).")))
      :no-thinks (err/raise "Unknown syntactic construct `thinks`"
                            (assoc (frame-err-opts state loc) :hint "There is no such construct."))
      nil)
    (-> state (fr/ensure-child who) (fr/enter who) (eval-stmt stmt) (fr/leave))))

(defn- eval-observe
  "The current frame observes its model of who's choice id: the choice becomes
  known, and the beliefs are conditioned on it."
  [state {:keys [who id loc]}]
  (let [frm (fr/get-frame state)
        ch (get-in frm [:choices [who id]])]
    (when-not ch
      (err/raise "Observation of unmodeled choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) " does not have " who "'s choice of " id
                                   " in their mental model. Perhaps you meant to write ("
                                   (:name frm) " :thinks (" who " :chooses [" id
                                   " ...])) somewhere earlier in this memo?"))))
    (when (:known ch)
      (err/raise "Observation of already-known choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) " already knows " who "'s choice of " id
                                   ". It doesn't make sense for " (:name frm)
                                   " to re-observe that same choice again."))))
    (let [state (fr/update-frame state assoc-in [:choices [who id] :known] true)
          ll (:ll (fr/get-frame state))
          [state total] (fresh state "total")
          state (emit state {:i :sum :out total :arg ll :dims (unknown-dims state)})
          state (add-evidence state total)
          [state q] (fresh state "ratio")
          state (emit state {:i :op :op :div :out q :args [ll total]})
          [state nll] (fresh state "ll")
          state (emit state {:i :nan-to-num :out nll :arg q})]
      (fr/update-frame state assoc :ll nll))))

(defn- eval-show
  "(who :observes target-who.target-id :is source): who conditions on the
  target being the source's value. who's beliefs move the target onto the
  source's dim, and the target's value becomes the source's."
  [state {:keys [who target-who target-id source-who source-id loc]}]
  (let [state (fr/ensure-child state who)
        frm (fr/get-frame state)
        target [target-who target-id]
        source [source-who source-id]]
    (when-not (get-in frm [:children who :choices target])
      (err/raise "Observation of unmodeled choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) " does not yet think that " who
                                   " is modeling " target-who "'s choice of " target-id
                                   ", so it doesn't make sense for " who " to observe "
                                   "that choice. Perhaps you meant to write (" who
                                   " :thinks (" target-who " :chooses [" target-id
                                   " ...])) somewhere earlier in " (:name frm) "'s memo?"))))
    (when-not (get-in frm [:choices source])
      (err/raise "Observation of unknown choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) " does not yet think that " source-who
                                   " has chosen " source-id ", so cannot model " who
                                   " observing that value. Perhaps you misspelled "
                                   source-id "?"))))
    (let [state (eval-stmt state {:node :with-stmt :who who :loc loc
                                  :stmt {:node :observe :who target-who :id target-id :loc loc}})
          frm (fr/get-frame state)
          child (get-in frm [:children who])
          target-ch (get-in child [:choices target])
          source-ch (get-in frm [:choices source])]
      (when (not= (:domain target-ch) (:domain source-ch))
        (err/raise "Domain mismatch"
                   (assoc (frame-err-opts state loc)
                          :hint (str target-who "." target-id " is from domain "
                                     (:domain target-ch) ", while " source-who "." source-id
                                     " is from domain " (:domain source-ch) "."))))
      (let [tidx (:idx target-ch)
            sidx (:idx source-ch)
            label (fn [[w i]] (if (= 'self w) (str i) (str w "." i)))
            ;; who's beliefs already vary along the source's dim: the swap
            ;; puts two of them on one axis (read-ll reports it, where
            ;; Python memo would crash)
            scrambled (when (and (not= sidx tidx) (contains? (:ll-dims child) sidx))
                        {:source (label source) :target (label target)})
            [state sw] (fresh state "ll")
            state (emit state {:i :swap-dims :out sw :arg (:ll child) :d1 sidx :d2 tidx})
            ;; a rejection sampler's evidence moves with the beliefs
            [state ev] (if-let [ev (and (:bounded state) (:evidence child))]
                         (let [[st e] (fresh state "evidence")]
                           [(emit st {:i :swap-dims :out e :arg ev :d1 sidx :d2 tidx}) e])
                         [state nil])
            [state value] (fresh state (str target-who "_" target-id))
            state (emit state {:i :alias :out value :arg (:tag source-ch)})
            state (fr/update-frame state update-in [:children who]
                                   (fn [c]
                                     (-> c
                                         (assoc :ll sw)
                                         (cond-> ev (assoc :evidence ev))
                                         (update :ll-dims #(conj (disj % tidx) sidx))
                                         (update :scrambled #(or % scrambled))
                                         (assoc-in [:conditions target] source)
                                         (assoc-in [:choices target :idx] sidx)
                                         (assoc-in [:children target-who :choices ['self target-id] :idx] sidx))))
            state (repoint state (:tag target-ch) value)]
        ;; a mind inside who that observed the target holds its own alias of
        ;; it, which the repoint does not reach, as known, and beliefs along
        ;; its old dim, which nothing marginalizes now: reading them is Python
        ;; memo's shape error (a :knows copy shares the value, and follows; a
        ;; snapshot's copy is a belief, and its own E sums the dim)
        (if (= sidx tidx)
          state
          (fr/update-frame
           state update-in [:children who]
           mark-unreadable
           #(and (= tidx (:idx %)) (:known %) (not= value (:tag %)))
           #(and (contains? (:ll-dims %) tidx) (some :unreadable (vals (:choices %))))
           (fn [holder]
             {:message "Observation against an already-used value"
              :hint (str "In " who "'s mind, " holder " observed " (label target) " before "
                         who " observed it to be " (label source) ": memo represents an observed "
                         "value by sharing its axis, and " holder "'s copy cannot follow "
                         (label target) " onto " (label source) "'s. (Python memo fails here "
                         "with a shape error.) If " holder " simply learns " (label target)
                         ", write (" holder " :knows " (label target) ") in " who "'s mind "
                         "instead.")})))))))

(defn- eval-observes-factor
  "(who :observes-that e) and (who :observes-event :wpp e): who's beliefs
  weighted by e (as 0 or 1 for observes-that), and renormalized."
  [state {:keys [who what how]}]
  (let [state (-> state (fr/ensure-child who) (fr/enter who))
        [state ll] (if-let [ll (:ll (fr/get-frame state))]
                     [state ll]
                     (let [[st one] (fresh state "ll")]
                       [(emit st {:i :const :out one :value 1.0}) one]))
        ;; no choice consumes a bounded estimate here (sample-ll)
        choosing (:choosing state)
        [state v] (eval-expr (assoc state :choosing nil) what)
        state (assoc state :choosing choosing)
        [state w] (if (= how :boolean)
                    ;; jnp.bool(x) * 1.0
                    (let [[st z] (fresh state "lit")
                          st (emit st {:i :const :out z :value 0})
                          [st b] (fresh st "event")]
                      [(emit st {:i :op :op :neq :out b :args [(:tag v) z]}) b])
                    [state (:tag v)])
        [state f] (fresh state "factor")
        state (emit state {:i :op :op :mul :out f :args [ll w]})
        [state total] (fresh state "total")
        state (emit state {:i :sum :out total :arg f :dims (unknown-dims state)})
        state (add-evidence state total)
        [state q] (fresh state "ratio")
        state (emit state {:i :op :op :div :out q :args [f total]})
        [state nll] (fresh state "ll")
        state (emit state {:i :nan-to-num :out nll :arg q})
        dims (deps->dims (fr/get-frame state) (:deps v))]
    (-> state
        (fr/update-frame #(cond-> (-> % (assoc :ll nll) (update :ll-dims into dims))
                            ;; a rejection sampler's acceptance probabilities, checked
                            ;; to be in [0, 1] where it samples (memo.bounded)
                            (and (:bounded state) (= how :probability)) (update :event-weights (fnil conj []) w)))
        (fr/leave))))

(defn- eval-knows
  "(who :knows source): who models the source choice, as the current frame
  does, and knows it."
  [state {:keys [who source-who source-id loc]}]
  (when (= :no-knows (:variant (fr/get-frame state)))
    (err/raise "Unknown syntactic construct `knows`"
               (assoc (frame-err-opts state loc) :hint "There is no such construct.")))
  (let [state (fr/ensure-child state who)
        frm (fr/get-frame state)
        source [source-who source-id]
        src (get-in frm [:choices source])]
    (when-not src
      (err/raise "Knowing unknown choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str (:name frm) " does not yet model " source-who
                                   "'s choice of " source-id ". So, it doesn't make "
                                   "sense for " (:name frm) " to model " who
                                   " as knowing that choice."))))
    (let [state (fr/update-frame state update-in [:children who]
                                 #(-> %
                                      (fr/put-choice source (assoc src :known true))
                                      (assoc-in [:conditions source] source)))]
      (if (= 'self source-who)
        (fr/update-frame state #(-> %
                                    (fr/put-choice [who source-id] src)
                                    (assoc-in [:conditions [who source-id]] [(:name frm) source-id])))
        (let [state (-> state (fr/enter who) (fr/ensure-child source-who) (fr/leave))]
          (fr/update-frame state update-in [:children who :children source-who]
                           #(-> %
                                (fr/put-choice ['self source-id] (assoc src :known true))
                                (assoc-in [:conditions ['self source-id]] source))))))))

(defn- eval-snapshot
  "(who :snapshots-self-as alias): who keeps a copy of its present self, with
  value symbols of its own, which later statements about who do not reach."
  [state {:keys [who alias]}]
  (let [state (fr/ensure-child state who)
        path (conj (:path state) who)
        current (fr/get-frame state path)
        future (reduce (fn [f [addr _]] (assoc-in f [:conditions addr] addr))
                       (assoc current :name alias)
                       (:choices current))
        [state future] (fresh-tags state future)
        current' (reduce (fn [cur [[nm id] _]]
                           (if (= 'self nm)
                             (-> cur
                                 (fr/put-choice [alias id] (get-in future [:choices ['self id]]))
                                 (assoc-in [:conditions [alias id]] [who id]))
                             cur))
                         current
                         (:choices current))]
    (fr/update-frame* state path (constantly (assoc-in current' [:children alias] future)))))

(defn- eval-wants
  "(who :wants goal e): who's goal, an expectation evaluated by (EU goal) in
  who's imagined future."
  [state {:keys [who what how loc]}]
  (when (= :no-wants (:variant (fr/get-frame state)))
    (err/raise "Unknown syntactic construct `wants`"
               (assoc (frame-err-opts state loc) :hint "There is no such construct.")))
  (let [state (fr/ensure-child state who)
        frm (fr/get-frame state)]
    (when (get-in frm [:children who :goals what])
      (err/raise (str who " already wants " what)
                 (assoc (frame-err-opts state loc) :hint "Pick a new name for this goal?")))
    (when (get-in frm [:choices ['self what]])
      (err/raise (str "The goal name " what " is already the name of a choice")
                 (assoc (frame-err-opts state loc) :hint "Pick a new name for this goal?")))
    (fr/update-frame state assoc-in [:children who :goals what]
                     {:node :expect :expr how :reduction :expectation :warn false
                      :loc loc :static false :surface "EU's goal"})))

(defn- eval-guess
  "The future self's guess of the value it will observe: a choice weighted by
  its current belief in each value (core.py's SGuess). With a budget it is
  exact: the guessed observation is part of the imagined future, which the
  agent's query over that future samples (sampling the guess first too would
  sample it twice, and, by Monte Carlo, bias the outer query)."
  [state {:keys [who id target-who target-id loc]}]
  (let [ch (get-in (fr/get-frame state) [:children who :choices [target-who target-id]])]
    (when-not ch
      (err/raise "Observing unknown choice"
                 (assoc (frame-err-opts state loc)
                        :hint (str who " does not model " target-who "'s choice of " target-id "."))))
    (-> state
        (assoc :exact true)
        (eval-stmt
               {:node :choose :who who :choices [[id (:domain ch)]] :reduction :normalize :loc loc
                :wpp {:node :expect :reduction :expectation :warn false :loc loc :static false
                      :expr {:node :op :op :eq :loc loc :static false
                             :args [{:node :choice :id id :loc loc :static false}
                                    {:node :with :who target-who :loc loc :static false
                                     :expr {:node :choice :id target-id :loc loc :static false}}]}}})
        (assoc :exact false))))

(defn- forget-copies
  "frame after who forgot the choices of by-idx (dim -> id): the copies of
  them that other frames inside it hold (a :knows or an :observes made
  earlier), and the beliefs that vary with them, are gone. Reading one is
  a compile error; in Python memo it is a KeyError."
  [frame who by-idx]
  (let [why (fn [holder id]
              {:message (str who "." id " was forgotten")
               :hint (str who " forgot " id " with (" who " :forgets-about " id "), which sums it "
                          "out of " (:name frame) "'s beliefs, but " holder " still refers to it, "
                          "through a :knows or an :observes made earlier. Forget it after the "
                          "last use of " holder "'s beliefs, or not at all. (Python memo fails "
                          "here with a KeyError.)")})]
    (reduce (fn [frame [idx id]]
              (mark-unreadable frame #(= idx (:idx %)) #(contains? (:ll-dims %) idx) #(why % id)))
            frame by-idx)))

(defn- eval-forgets-about
  "(who :forgets-about x ...): who's own choices summed out of the current
  frame's beliefs, once, and dropped from scope: later computations need not
  marginalize them again (core.py's SForgetsAbout)."
  [state {:keys [who ids loc]}]
  (let [state (fr/ensure-child state who)
        frm (fr/get-frame state)
        child (get-in frm [:children who])
        idxs (mapv (fn [id]
                     (when (or (not (contains? (:choices child) ['self id]))
                               (contains? (:conditions child) ['self id]))
                       (err/raise (str who " has no such choice " id " to forget")
                                  (assoc (frame-err-opts state loc)
                                         :hint (str "forgets-about only applies to " who "'s own "
                                                    "choices made via chooses/given/etc. " id " is "
                                                    "either unknown to " who ", or was learned via "
                                                    "knows/observes rather than chosen."))))
                     (when-not (contains? (:choices frm) [who id])
                       (err/raise (str (:name frm) " is not aware that " who " has chosen " id ".")
                                  (assoc (frame-err-opts state loc) :hint (str "Is there a typo in " id "?"))))
                     (get-in frm [:choices [who id] :idx]))
                   ids)
        state (if-let [ll (:ll frm)]
                (let [[st m] (fresh state "ll")]
                  (-> st
                      (emit {:i :sum :out m :arg ll :dims idxs})
                      (fr/update-frame assoc :ll m)))
                state)]
    (fr/update-frame state
                     (fn [f]
                       (-> (reduce (fn [f id]
                                     (-> f
                                         (update-in [:children who :choices] dissoc ['self id])
                                         (update-in [:children who :conditions] dissoc ['self id])
                                         (update :choices dissoc [who id])
                                         (update :conditions dissoc [who id])))
                                   (update f :ll-dims #(reduce disj % idxs))
                                   ids)
                           (forget-copies who (zipmap idxs ids)))))))

(defn- eval-trace
  "(who :inspect): what who tracks, printed while the memo compiles (Python
  memo's INSPECT)."
  [state {:keys [who loc]}]
  (let [frm (fr/get-frame state)
        label (fn [[w i]] (if (= 'self w) (str i) (str w "." i)))]
    (println (str "** Tracing " who (when (:name loc) (str " in defmemo " (:name loc)))
                  (when (:line loc) (str ", line " (:line loc)))))
    (if-let [f (get-in frm [:children who])]
      (do
        (when (not= :adult (:variant f))
          ;; Python memo prints its Variant's name
          (println (str "-> " who " is using the variant "
                        (str/replace (str/upper-case (name (:variant f))) "-" "_"))))
        (println (str "-> " who " is tracking the following choices:"))
        (doseq [[addr c] (sort-by (comp :seq val) (:choices f))]
          (println (str "   - " (label addr) ": " (:domain c) " (" (if (:known c) "known" "uncertain") ")"))
          (when-let [[w i] (get (:conditions f) addr)]
            (println (str "     | (observed to be " (:name frm) "'s " w "." i ")"))))
        (if (empty? (:children f))
          (println (str "-> " who " is not yet tracking any agents."))
          (do (println (str "-> " who " is currently tracking the following agents:"))
              (doseq [c (sort-by :seq (vals (:children f)))] (println (str "   + " (:name c))))))
        (doseq [i (range (count (:path state)) -1 -1)]
          (let [inner (if (= i (count (:path state))) f (fr/get-frame state (subvec (:path state) 0 (inc i))))
                outer (fr/get-frame state (subvec (:path state) 0 i))]
            (println (str "-> " (:name inner) " is being modeled by " (:name outer) ".")))))
      (println (str "-> From " (:name frm) "'s perspective, there is not yet any model of " who
                    ". The current agents currently modeled by " (:name frm) " are: "
                    (str/join ", " (map :name (sort-by :seq (vals (:children frm))))) ".")))
    (println)
    state))

(defn- eval-variant [state {:keys [who variant]}]
  (-> state
      (fr/ensure-child who)
      (fr/update-frame assoc-in [:children who :variant] variant)))

(defn eval-stmt [state s]
  ;; a statement's queries (an imagined one's, at a return) feed its choices,
  ;; not the return: their masses are not checked (record-mass)
  (let [armed (:check-mass state)
        state (cond-> state armed (assoc :check-mass nil))
        state (case (:node s)
                :pass state
                :forall (eval-forall state s)
                :choose (eval-choose state s)
                :with-stmt (eval-with-stmt state s)
                :observe (eval-observe state s)
                :show (eval-show state s)
                :observes-factor (eval-observes-factor state s)
                :knows (eval-knows state s)
                :snapshot (eval-snapshot state s)
                :guess (eval-guess state s)
                :variant (eval-variant state s)
                :trace (eval-trace state s)
                :wants (eval-wants state s)
                :forgets-about (eval-forgets-about state s)
                (err/raise (str "Unknown statement node " (:node s)) {:user false :loc (:loc s)}))]
    (cond-> state armed (assoc :check-mass armed))))

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
   :domains domains
   :bounded nil
   :limit nil
   :pending []
   :mixing []
   :choosing nil
   :exact false
   :report []})

(defn analyze
  "The IR program of a parsed memo:
  {:instrs [...] :results [sym ...] :forall [[dim id domain] ...]
   :axes [id ...] :domains [domain ...] :shape [n ...] :n-dims n}.
  domains maps each domain symbol to its values; opts has :variant and
  :callee, from the name of a memo defined before to {:axes n :params [..]
  :defaults #{..} :num-results n}; with a budget (M3), :bounded, a normalized
  budget (memo.bounded/normalize-budget), and :limit, the call's (with
  :report-only, sizes above it are reported, not errors). A bounded program
  also has :bounded-report, its choices' sizes, and :bounded-parts, [agent
  part] for each part of the budget (:samples, :options) that reaches a query
  or a choice (:options k reaches a choice of k or fewer options too). With
  :mass (memo.gen/deterministic's check), the program also has :masses, the
  totals of the beliefs its returns weigh by (record-mass), each {:tag
  :frame :what :loc :cells :return}."
  [{:keys [pctxt stmts retvals]} domains {:keys [variant callee bounded limit report-only mass]}]
  (let [state (reduce (fn [st [i s]]
                        (eval-stmt (assoc st :continuation (subvec stmts (inc i))) s))
                      (assoc (initial-state pctxt domains variant)
                             :callee callee :bounded bounded :limit limit :report-only report-only)
                      (map-indexed vector stmts))
        [state vals] (reduce (fn [[st acc] [i retval]]
                               (let [[st val] (eval-expr (cond-> st mass (assoc :check-mass i)) retval)
                                     st (assoc st :check-mass nil)]
                                 (when-not (:known val)
                                   (err/raise "Returning a value that the observer has uncertainty over"
                                              (assoc (frame-err-opts st (:loc retval))
                                                     :hint (str "Did you mean to wrap this return "
                                                                "value in an (E ...) expression or a "
                                                                "(Pr ...) expression?"))))
                                 [st (conj acc val)]))
                             [state []]
                             (map-indexed vector retvals))
        forall (:forall state)]
    (cond-> {:instrs (:instrs state)
             :results (mapv :tag vals)
             :forall forall
             :axes (mapv second forall)
             :domains (mapv #(nth % 2) forall)
             :shape (mapv (fn [[_ _ dom]] (count (get domains dom))) forall)
             :n-dims (:next-dim state)
             :bounded-report (:report state)
             :bounded-parts (into (set (map (fn [a] [a :options]) (:all-considered state)))
                                  (for [{:keys [agent sources]} (:report state), {:keys [kind]} sources]
                                    [agent kind]))}
      mass (assoc :masses (vec (:masses state))))))
