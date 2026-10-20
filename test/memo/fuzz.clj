(ns memo.fuzz
  "The differential fuzzer (DESIGN.md §9): random memo programs run in
  memo.clj and, through memo.pygen, in Python memo, which must agree, both
  on the values of programs that compile and on rejecting those that do
  not.

  The generator tracks scope (each frame's own choices, what it knows, the
  agents it models), so most programs are well formed; the rest test that
  both reject. Constructs whose answers depend on the dtype are kept out
  (DESIGN.md §13.8: :to-maximize over non-integer values, large exp, KL
  against zeros), as are snapshots followed by observations (Python memo's
  snapshot bug, in the upstream ledger).

    (campaign {:n 200 :seed 1})   ; generate, run both, compare, report
    clojure -X:test:fuzz :n 200 :seed 1

  A disagreement is shrunk (statements and returns dropped while it
  persists) and returned for triage; a confirmed one goes into regressions
  (record-regressions! writes test/fixtures/fuzz.edn), which memo.fuzz-test
  replays without Python."
  (:require [clojure.pprint]
            [clojure.string :as str]
            [clojure.walk]
            [memo.core]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.pygen :as pygen]
            [memo.python :as py]
            [memo.test-support :as ts]))

;; -----------------------------------------------------------------------------
;; the programs' world: domains every program may use

(def B2 [0 1])
(def D3 [0 1 2])
(def D4 [0 1 2 3])
(def P3 [0.1 0.5 0.9])
(defenum Color RED GREEN BLUE)

(def domains
  "Each domain and its values (enum members as values)."
  {'B2 [0 1] 'D3 [0 1 2] 'D4 [0 1 2 3] 'P3 [0.1 0.5 0.9] 'Color [0 1 2]})

(def world-forms
  "The fuzz namespace's definitions, as forms, for the transpiler: domains,
  and w3, a model the programs call."
  '[(def B2 [0 1]) (def D3 [0 1 2]) (def D4 [0 1 2 3]) (def P3 [0.1 0.5 0.9])
    (defenum Color RED GREEN BLUE)
    (defmemo w3 [[k D3]] [t]
      (alice :knows k)
      (alice :chooses [y D3] :wpp (exp (* t (- y k))))
      (return (Pr (= alice.y k))))])

;; -----------------------------------------------------------------------------
;; random choices, from a seeded java.util.Random

(def ^:dynamic ^java.util.Random *rng* (java.util.Random. 1))

(defn- rand-int* [n] (.nextInt *rng* n))
(defn- chance [p] (< (.nextDouble *rng*) p))
(defn- pick [coll] (nth (vec coll) (rand-int* (count coll))))
(defn- shuffle* [coll]
  (let [v (java.util.ArrayList. ^java.util.Collection (vec coll))]
    (java.util.Collections/shuffle v *rng*)
    (vec v)))

(defn- pick-weighted
  "One of the [weight value] pairs, in proportion to the weights."
  [pairs]
  (let [total (reduce + (map first pairs))
        r (* total (.nextDouble *rng*))]
    (loop [[[w v] & more] pairs acc 0.0]
      (if (or (empty? more) (< r (+ acc w))) v (recur more (+ acc w))))))

;; -----------------------------------------------------------------------------
;; expressions
;;
;; A scope says what an expression may refer to: :own (bare names of the
;; frame's own choices and knowns, name -> domain), :dotted (agent.choice
;; references, symbol -> domain), :params, :beliefs (agent.choice the frame
;; is uncertain about, for E/Pr inside the frame).

(defn- lit-num []
  (pick-weighted [[3 (rand-int* 4)] [1 (pick [0.5 1.5 2.0 -1.0])]]))

(defn- refs [scope] (concat (keys (:own scope)) (keys (:dotted scope)) (:params scope)))

(declare num-expr bool-expr)

(defn- belief-expr
  "E/Pr/Var over the frame's beliefs, or nil when it has none."
  [scope depth]
  (when-let [bs (seq (keys (:beliefs scope)))]
    (let [b (pick bs)]
      (pick-weighted
       [[3 (list 'E b)]
        [3 (list 'Pr (list '= b (pick (domains (get (:beliefs scope) b)))))]
        [1 (list 'Var b)]
        [1 (list 'E (list '+ b (num-expr (dissoc scope :beliefs) (dec depth))))]]))))

(defn num-expr
  "A numeric expression in scope, of depth at most depth."
  [scope depth]
  (let [rs (refs scope)
        leaf #(if (and (seq rs) (chance 0.7)) (pick rs) (lit-num))]
    (if (<= depth 0)
      (leaf)
      (pick-weighted
       (cond-> [[4 (leaf)]
                [2 (list (pick '[+ - *]) (num-expr scope (dec depth)) (num-expr scope (dec depth)))]
                [1 (list 'if (bool-expr scope (dec depth)) (num-expr scope (dec depth)) (num-expr scope (dec depth)))]
                [1 (list 'abs (num-expr scope (dec depth)))]]
         (seq (:beliefs scope)) (conj [2 (belief-expr scope depth)]))))))

(defn bool-expr
  "A boolean expression in scope."
  [scope depth]
  (if (<= depth 0)
    (list (pick '[= < > not=]) (num-expr scope 0) (num-expr scope 0))
    (pick-weighted
     [[4 (list (pick '[= < > <= >= not=]) (num-expr scope (dec depth)) (num-expr scope (dec depth)))]
      [1 (list (pick '[and or]) (bool-expr scope (dec depth)) (bool-expr scope (dec depth)))]
      [1 (list 'not (bool-expr scope (dec depth)))]])))

(defn- weight-expr
  "A weight (wpp): positive, or a boolean (possibly zero mass), bounded."
  [scope]
  (pick-weighted
   [[2 1]
    [3 (list '+ 1 (list 'abs (num-expr scope 1)))]
    [2 (list 'exp (list '* (pick [0.5 -0.5 1.0]) (num-expr scope 1)))]
    [2 (bool-expr scope 1)]
    [1 (list '+ 0.5 (bool-expr scope 1))]]))

(defn- int-valued
  "An expression with integer values only (for :to-maximize: ties are then
  exact in every dtype), over the own choices and integer literals."
  [scope]
  (let [own (keys (filter (fn [[_ d]] (not= d 'P3)) (:own scope)))
        leaf #(if (and (seq own) (chance 0.8)) (pick own) (rand-int* 3))]
    (pick-weighted [[2 (leaf)]
                    [2 (list (pick '[+ -]) (leaf) (leaf))]
                    [1 (list '* (leaf) (leaf))]])))

;; -----------------------------------------------------------------------------
;; statements, tracking scope
;;
;; The root state: {:axes {name dom} :params [..] :agents {agent ...}}, an
;; agent {:choices {x dom} :knows {ref dom} :mind {other {x dom}}
;; :mind-knows {other {ref dom}} :deep {other {third {x dom}}} :observed
;; #{[b x]} :goals #{g} :snapshot alias :frozen? bool :forgot #{x}}.

(def agent-names '[alice bob carol])
(def choice-names '[x y z w])

(defn- dotted [a x] (symbol (str a "." x)))

(defn- agent-scope
  "What agent a may refer to in its own statements at the root."
  [st a]
  (let [{:keys [choices knows mind forgot]} (get-in st [:agents a])]
    {:own (apply dissoc (merge choices (into {} (remove (comp #(str/includes? (str %) ".") key)) knows)) forgot)
     :dotted (into {} (filter (comp #(str/includes? (str %) ".") key)) knows)
     :params (:params st)
     :beliefs (into {} (for [[b cs] mind [x d] cs] [(dotted b x) d]))}))

(defn- fresh-choice [taken] (first (remove (set taken) choice-names)))

(defn- taken-by [st a]
  (concat (keys (get-in st [:agents a :choices])) (keys (get-in st [:agents a :knows]))
          (get-in st [:agents a :forgot])))

(defn- usable? [st a] (not (get-in st [:agents a :frozen?])))

(def ^:private world-calls
  "Calls of the world's model w3 (a table over D3), as weights."
  (fn [x] (list 'w3 [x] (pick [0.5 1.0 -1.0]))))

(defn- spec-for
  "A :chooses spec for choices xs (name -> dom) in scope."
  [st a scope xs]
  (let [goals (seq (get-in st [:agents a :goals]))
        d3 (first (for [[x d] xs :when (= d 'D3)] x))]
    (pick-weighted
     (cond-> [[6 [:wpp (weight-expr scope)]]
              [2 [:uniformly]]
              [1 [:such-that (bool-expr scope 1)]]
              [1 [:to-maximize (int-valued scope)]]
              [1 [:to-minimize (int-valued scope)]]
              [1 [:to-be (if (seq (apply dissoc (:own scope) (keys xs)))
                           (pick (keys (apply dissoc (:own scope) (keys xs))))
                           (rand-int* 3))]]]
       goals (conj [3 [:wpp (list 'exp (list '* (pick [0.5 1.0 2.0]) (list 'EU (pick goals))))]])
       d3 (conj [1 [:wpp (world-calls d3)]])))))

(defn- gen-chooses
  "(a :chooses [x D] ... spec) at the root, a's frame in scope."
  [st a]
  (let [n (pick-weighted [[5 1] [1 2]])
        xs (take n (remove (set (taken-by st a)) choice-names))]
    (when (= n (count xs))
      (let [xs (into {} (map (fn [x] [x (pick (keys domains))]) xs))
            scope (update (agent-scope st a) :own merge xs)
            spec (spec-for st a scope xs)]
        [(update-in st [:agents a :choices] merge xs)
         (concat (list a :chooses) (map (fn [[x d]] [x d]) xs) spec)]))))

(defn- gen-knows
  "(a :knows u) or (a :knows b.y): something the root knows or another
  agent chose."
  [st a]
  (let [options (concat (for [[u d] (:axes st)] [u d])
                        (for [[b {cs :choices}] (:agents st) :when (not= b a) [y d] cs]
                          [(dotted b y) d]))
        options (remove (fn [[r _]] (or (contains? (get-in st [:agents a :knows]) r)
                                        (some #{r} (taken-by st a))))
                        options)]
    (when (seq options)
      (let [[r d] (pick options)]
        [(assoc-in st [:agents a :knows r] d) (list a :knows r)]))))

(defn- gen-thinks
  "(a :thinks (b :chooses ...)), (a :thinks (b :knows c.x)) or, one level
  deeper, (a :thinks (b :thinks (c :chooses ...)))."
  [st a]
  (let [b (pick (remove #{a} agent-names))
        mind (get-in st [:agents a :mind])]
    (pick-weighted
     (cond-> [[5 (let [x (fresh-choice (concat (keys (get mind b)) (keys (get-in st [:agents a :mind-knows b]))))]
                   (when x
                     (let [d (pick (keys domains))
                           known (get-in st [:agents a :mind-knows b])
                           scope {:own (assoc (get mind b) x d) :dotted known :params (:params st)}
                           spec (pick-weighted [[4 [:wpp (weight-expr scope)]] [1 [:uniformly]]])]
                       [(assoc-in st [:agents a :mind b x] d)
                        (list a :thinks (list* b :chooses [x d] spec))])))]
              [1 (let [c (pick (remove #{a b} agent-names))
                       x (fresh-choice (keys (get-in st [:agents a :deep b c])))]
                   (when x
                     (let [d (pick (keys domains))]
                       [(assoc-in st [:agents a :deep b c x] d)
                        (list a :thinks (list b :thinks (list c :chooses [x d] :wpp
                                                              (weight-expr {:own {x d} :params (:params st)}))))])))]]
       ;; b knows, in a's mind, a choice a believes another agent made
       (some (fn [[c cs]] (and (not= c b) (seq cs))) mind)
       (conj [2 (let [[c cs] (pick (filter (fn [[c cs]] (and (not= c b) (seq cs))) mind))
                      [x d] (pick cs)
                      r (dotted c x)]
                  (when-not (contains? (get-in st [:agents a :mind-knows b]) r)
                    [(assoc-in st [:agents a :mind-knows b r] d)
                     (list a :thinks (list b :knows r))]))])))))

(defn- gen-observes
  "(a :observes b.x :is v): v a root value of the same domain."
  [st a]
  (let [targets (for [[b cs] (get-in st [:agents a :mind]) [x d] cs
                      :when (not (contains? (get-in st [:agents a :observed]) [b x]))]
                  [b x d])]
    (when (seq targets)
      (let [[b x d] (pick targets)
            sources (concat (for [[u du] (:axes st) :when (= du d)] u)
                            (for [[c {cs :choices}] (:agents st) :when (not= c a) [y dy] cs :when (= dy d)]
                              (dotted c y)))]
        (when (seq sources)
          [(update-in st [:agents a :observed] (fnil conj #{}) [b x])
           (list a :observes (dotted b x) :is (pick sources))])))))

(defn- gen-observes-that [st a]
  (let [scope (agent-scope st a)]
    (when (seq (:beliefs scope))
      (let [scope {:own {} :dotted (:beliefs scope) :params (:params st)}]
        [st (if (chance 0.6)
              (list a :observes-that (bool-expr scope 1))
              (list a :observes-event :wpp (weight-expr scope)))]))))

(defn- gen-wants
  "(a :wants g e): a goal over a's beliefs and its next choice, x (which
  EU then maximizes, through :wpp (exp (EU g)), so near-ties never pick)."
  [st a]
  (let [scope (agent-scope st a)
        x (fresh-choice (taken-by st a))
        g (first (remove (set (get-in st [:agents a :goals])) '[g1 g2]))]
    (when (and x g (seq (:beliefs scope)))
      (let [b (pick (keys (:beliefs scope)))]
        [(update-in st [:agents a :goals] (fnil conj #{}) g)
         (list a :wants g (pick-weighted [[2 (list '= x b)]
                                          [1 (list '- (list 'abs (list '- x b)))]]))]))))

(defn- gen-forgets [st a]
  (let [cs (keys (get-in st [:agents a :choices]))]
    (when (and (seq cs) (empty? (get-in st [:agents a :goals])))
      (let [x (pick cs)]
        [(-> st
             (update-in [:agents a :choices] dissoc x)
             (update-in [:agents a :forgot] (fnil conj #{}) x))
         (list a :forgets-about x)]))))

(defn- gen-snapshot
  "(a :snapshots-self-as past_a), after which a does nothing more (an
  observation after a snapshot is Python memo's snapshot bug)."
  [st a]
  (when (and (seq (get-in st [:agents a :mind])) (not (get-in st [:agents a :snapshot])))
    (let [alias (symbol (str "past_" a))]
      [(-> st (assoc-in [:agents a :snapshot] alias) (assoc-in [:agents a :frozen?] true))
       (list a :snapshots-self-as alias)])))

(defn- gen-statement [st]
  (let [a (pick (filter #(usable? st %) agent-names))]
    (when a
      (some (fn [g] (g st a))
            (shuffle* [(pick-weighted [[6 gen-chooses] [2 gen-knows] [4 gen-thinks] [2 gen-observes]
                                       [1 gen-observes-that] [1 gen-wants] [1 gen-forgets] [1 gen-snapshot]])
                       gen-chooses])))))

;; -----------------------------------------------------------------------------
;; returns, programs

(defn- root-beliefs [st]
  (into {} (for [[a {cs :choices}] (:agents st) [x d] cs] [(dotted a x) d])))

(defn- root-scope
  "The root's own references: axes and params, and every agent's choices as
  beliefs (the root is uncertain about them)."
  [st]
  {:own (:axes st) :params (:params st) :beliefs (root-beliefs st)})

(defn- gen-return [st]
  (let [scope (root-scope st)
        beliefs (root-beliefs st)
        minds (for [[a {:keys [mind]}] (:agents st) :when (seq mind)] a)
        deep (for [[a {:keys [deep]}] (:agents st) [b cs] deep [c xs] cs [x _] xs] [a b c x])
        same-dom (for [[u du] (:axes st) [r d] beliefs :when (= du d)] [u r])
        snaps (for [[a {:keys [snapshot mind]}] (:agents st) :when snapshot] [a snapshot mind])]
    (pick-weighted
     (cond-> [[3 (or (belief-expr scope 2) (num-expr scope 1))]
              [1 (num-expr (dissoc scope :beliefs) 1)]
              [1 (list 'imagine (list 'world :chooses ['s 'D3] :wpp (weight-expr {:own {'s 'D3} :params (:params st)}))
                       (list 'E (list '+ 'world.s (num-expr (dissoc scope :beliefs) 0))))]]
       (seq minds)
       (conj [3 (let [a (pick minds)]
                  (list 'E (list 'as a (or (belief-expr (assoc (agent-scope st a) :dotted {}) 2) 0))))])
       (seq deep)
       (conj [2 (let [[a b c x] (pick deep)]
                  (list 'E (list 'as a (list 'E (list 'as b (list 'E (dotted c x)))))))])
       (seq same-dom)                                  ; the posterior shortcut
       (conj [3 (let [pairs (take (inc (rand-int* 2)) (shuffle* same-dom))]
                  (list* 'Pr (map (fn [[u r]] (if (chance 0.5) (list '= u r) (list '= r u))) pairs)))])
       (seq beliefs)
       (conj [2 (let [rs (take (inc (rand-int* 2)) (shuffle* (keys beliefs)))] (list* 'H rs))])
       (< 1 (count beliefs))
       (conj [1 (let [[r1 r2] (shuffle* (keys beliefs))] (list 'KL r1 r2))])
       (seq snaps)
       (conj [2 (let [[a alias mind] (pick snaps)
                      [b cs] (pick mind)]
                  (list 'E (list 'as a (list 'as alias (list 'E (dotted b (key (first cs))))))))])))))

(defn gen-program
  "A random defmemo named mname, and the calls to make of it."
  [mname]
  (let [axes (vec (take (pick-weighted [[3 0] [3 1] [1 2]]) (shuffle* (for [[n d] [['u 'D3] ['v 'B2]]] [n d]))))
        params (vec (take (rand-int* 3) '[p q]))
        st0 {:axes (into {} axes) :params params :agents {}}
        [st stmts] (loop [st st0 stmts [] k (+ 1 (rand-int* 7))]
                     (if (zero? k)
                       [st stmts]
                       (if-let [[st' s] (gen-statement st)]
                         (recur st' (conj stmts s) (dec k))
                         (recur st stmts (dec k)))))
        returns (vec (repeatedly (pick-weighted [[4 1] [1 2]]) #(gen-return st)))
        form (concat (list 'defmemo mname axes params) stmts (map #(list 'return %) returns))
        call (fn [] (let [vals (mapv (fn [_] (pick [0.3 0.7 1.0 1.5])) params)]
                      (if (and (seq params) (chance 0.3))
                        {:args [] :kwargs (zipmap (map keyword params) vals)}
                        {:args vals :kwargs {}})))]
    {:form form :calls (vec (distinct (repeatedly 2 call)))}))

;; -----------------------------------------------------------------------------
;; mutations: ill-formed programs both must reject

(defn mutate
  "A copy of form with one fault planted, or nil when none applies: a
  choice reference renamed to one nobody chose, or a statement repeated.
  (Faults memo.parse rejects, such as a missing return, cannot be
  transpiled, which works from the parsed program.)"
  [form]
  (let [[head mname axes params & body] form
        stmts (vec (remove #(= 'return (first %)) body))
        returns (vec (filter #(= 'return (first %)) body))
        rebuild (fn [stmts returns] (concat (list head mname axes params) stmts returns))
        dotted (distinct (filter #(and (symbol? %) (str/includes? (name %) ".")) (flatten (map seq body))))]
    (pick-weighted
     (cond-> [[1 (when (seq stmts) (rebuild (conj stmts (pick stmts)) returns))]]  ; repeated statement
       (seq dotted)
       (conj [3 (let [r (pick dotted)
                      [a _] (str/split (name r) #"\.")]
                  (clojure.walk/postwalk-replace {r (symbol (str a ".nowhere"))} form))])))))

;; -----------------------------------------------------------------------------
;; running both, comparing

(defn- eval-in [ns form]
  (binding [*ns* ns err/*warnings?* false] (eval form)))

(defn- memo-clj-outcome
  "memo.clj's outcome for a program: :ce (with message) or each call's."
  [ns {:keys [form calls]}]
  (if-let [e (try (eval-in ns form) nil (catch Throwable e (or (err/root-memo-error e) e)))]
    {:outcome :ce :message (ex-message e) :memo-error (err/memo-error? e)}
    (let [f @(ns-resolve ns (second form))]
      {:outcome :compiled
       :calls (mapv (fn [{:keys [args kwargs]}]
                      (try {:outcome :ok
                            :value (ts/host (ts/deep #(apply f (cond-> (vec args) (seq kwargs) (conj kwargs)))))}
                           (catch Throwable e {:outcome :re :message (ex-message e)
                                               :memo-error (err/memo-error? (or (err/root-memo-error e) e))})))
                    calls)})))

(defn- near?
  "Values agree within float32 accumulation over a random program's tables
  (up to thousands of cells): 1e-4 relative, 1e-5 absolute. NaN matches
  NaN."
  [expected actual]
  (cond
    (and (sequential? expected) (sequential? actual))
    (and (= (count expected) (count actual)) (every? true? (map near? expected actual)))
    (and (number? expected) (number? actual))
    (let [e (double expected) a (double actual)]
      (or (and (Double/isNaN e) (Double/isNaN a))
          (= e a)
          (<= (Math/abs (- e a)) (+ 1e-5 (* 1e-4 (Math/abs e))))))
    :else false))

(defn- beyond-float32?
  "Whether every value that differs is at or past float32's largest number:
  KL against a zero probability, p times the dtype's largest number (or
  their sum overflowing), dtype-dependent by design (DESIGN.md §2.4)."
  [expected actual]
  (let [huge? #(or (Double/isInfinite (double %)) (> (Math/abs (double %)) 1e30))
        pairs (map vector (flatten [expected]) (flatten [actual]))]
    (every? (fn [[e a]] (or (near? e a) (and (huge? e) (huge? a)))) pairs)))

(defn- agree?
  "Whether memo.clj's outcome and Python memo's (one per call) agree, and
  how they differ when they do not."
  [ours theirs]
  (let [py-ce (some #(when (= :ce (:outcome %)) %) theirs)]
    (cond
      (some #(= :missing (:outcome %)) theirs) [true :not-transpiled]
      ;; memo.clj refuses with a memo error, never by crashing (CLAUDE.md rule 4)
      (and (= :ce (:outcome ours)) (not (:memo-error ours))) [false [:memo-clj-crashed (:message ours)]]
      (some #(and (= :re (:outcome %)) (not (:memo-error %))) (:calls ours))
      [false [:memo-clj-crashed (:message (first (filter #(= :re (:outcome %)) (:calls ours))))]]
      (and (= :ce (:outcome ours)) py-ce)
      (if (= (:message ours) (:message py-ce))
        [true :both-reject]
        [true [:both-reject-saying (:message ours) (:message py-ce)]])
      (= :ce (:outcome ours)) [false :only-memo-clj-rejects]
      py-ce [false :only-python-rejects]
      :else
      (let [diffs (keep-indexed
                   (fn [i [o t]]
                     (cond
                       (and (= :re (:outcome o)) (= :re (:outcome t))) nil
                       (not= (:outcome o) (:outcome t)) [i :outcome (:outcome o) (:outcome t)]
                       (near? (:value t) (:value o)) nil
                       (beyond-float32? (:value t) (:value o)) nil
                       :else [i :values]))
                   (map vector (:calls ours) theirs))]
        (if (empty? diffs) [true :same-values] [false diffs])))))

(defn world-ns
  "A fresh namespace with the programs' world defined in it."
  []
  (let [ns (create-ns (gensym "memo.fuzz.run"))]
    (binding [*ns* ns err/*warnings?* false]
      (refer-clojure)
      (require '[memo.core :refer [defmemo]] '[memo.domains :refer [defenum]])
      (doseq [f world-forms] (eval f)))
    ns))

(defn run-batch
  "Run programs (from gen-program, each with a distinct name) in memo.clj
  and Python memo. Each program with its outcomes and verdict."
  [programs]
  (let [ns (world-ns)
        ours (mapv #(memo-clj-outcome ns %) programs)
        {:keys [python]} (pygen/transpile-forms (concat world-forms (map :form programs)) ns {:guarded? true})
        calls (for [{:keys [form calls]} programs {:keys [args kwargs]} calls]
                {:model (pygen/sanitize (second form)) :args args :kwargs kwargs})
        theirs (py/run python calls)
        per-program (loop [ps programs ts theirs acc []]
                      (if (empty? ps)
                        acc
                        (let [n (count (:calls (first ps)))]
                          (recur (rest ps) (drop n ts) (conj acc (vec (take n ts)))))))]
    (remove-ns (ns-name ns))
    (mapv (fn [p o t] (let [[ok how] (agree? o t)] (assoc p :memo-clj o :python t :agree ok :how how)))
          programs ours per-program)))

(defn shrink
  "A smaller program that still disagrees: one statement (or one of several
  returns) dropped at a time, each round's candidates run as one batch,
  until none of them disagrees."
  [d]
  (loop [d d]
    (let [[head _ axes params & body] (:form d)
          body (vec body)
          returns (count (filter #(= 'return (first %)) body))
          candidates (for [i (range (count body))
                           :when (or (not= 'return (first (body i))) (< 1 returns))]
                       (assoc d :form (concat (list head (symbol (str "s" i)) axes params)
                                              (concat (subvec body 0 i) (subvec body (inc i))))))
          smaller (when (seq candidates) (first (remove :agree (run-batch candidates))))]
      (if smaller (recur (merge d (select-keys smaller [:form :memo-clj :python :how]))) d))))

(defn campaign
  "Generate n programs (and mutants of a third of them) from seed, run them
  in batches, and return {:counts .. :messages .. :disagreements [...]},
  each disagreement shrunk."
  [{:keys [n seed batch] :or {n 100 seed 1 batch 50}}]
  (binding [*rng* (java.util.Random. (long seed))]
    (let [programs (vec (for [i (range n)]
                          (let [p (gen-program (symbol (str "m" i)))]
                            (if-let [m (when (chance 0.33) (mutate (:form p)))]
                              (assoc p :form m :mutant true)
                              p))))
          results (vec (mapcat run-batch (partition-all batch programs)))]
      {:counts (frequencies (map (fn [r] [(if (:mutant r) :mutant :program)
                                          (cond (not (:agree r)) :DISAGREE
                                                (vector? (:how r)) :both-reject-other-message
                                                :else (:how r))])
                                 results))
       :messages (frequencies (keep #(when (vector? (:how %)) (rest (:how %))) results))
       :disagreements (mapv shrink (remove :agree results))})))

(defn cli
  "clojure -X:test:fuzz :n 200 :seed 1 — a campaign, printed."
  [opts]
  (if-not (py/available?)
    (println "Python memo is not available (memo-mcp/venv): no campaign")
    (let [{:keys [counts disagreements]} (campaign opts)]
      (prn counts)
      (doseq [d disagreements]
        (prn (select-keys d [:form :calls :how :memo-clj :python])))))
  (shutdown-agents))

;; -----------------------------------------------------------------------------
;; regressions: what the fuzzer found, replayed without Python

(def regressions
  "Programs from the fuzzer's findings, made minimal by hand, each with why.
  record-regressions! runs them in Python memo and writes
  test/fixtures/fuzz.edn; memo.fuzz-test replays them."
  [{:id :var-zero-mass
    :why (str "Var when a branch has zero mass (y cannot be x = 2 or 3, so the frame's total "
              "mass is 1/2): Python memo's E[X^2] - E[X]^2 over the unnormalized weights; "
              "memo.clj's centered form alone gave another number (found 2026-10-20, seed 3)")
    :form '(defmemo var-zero-mass [] []
             (bob :chooses [x D4] :wpp 1)
             (bob :chooses [y B2] :to-be x)
             (bob :chooses [z D4] :wpp (exp (* 0.5 z)))
             (return (Var bob.z)))
    :calls [{:args [] :kwargs {}}]}
   {:id :var-zero-mass-axes
    :why "the same, over axes and params, as the fuzzer generated it (m146, seed 2)"
    :form '(defmemo var-zero-mass-axes [[v B2] [u D3]] [p q]
             (carol :chooses [x D4] [y P3] :to-maximize (- x x))
             (carol :chooses [z D4] :to-be x)
             (carol :chooses [w D3] :to-be x)
             (alice :chooses [x D3] :to-maximize (* x x))
             (alice :thinks (bob :chooses [x B2] :uniformly))
             (return (Var carol.y)))
    :calls [{:args [0.3 0.7] :kwargs {}}]}])

(defn record-regressions!
  "Run the regressions in Python memo and write test/fixtures/fuzz.edn."
  []
  (let [ns (world-ns)
        {:keys [python]} (pygen/transpile-forms (concat world-forms (map :form regressions)) ns {:guarded? true})
        outcomes (py/run python (for [{:keys [form calls]} regressions {:keys [args kwargs]} calls]
                                  {:model (pygen/sanitize (second form)) :args args :kwargs kwargs}))
        entries (loop [rs regressions os outcomes acc []]
                  (if (empty? rs)
                    acc
                    (let [n (count (:calls (first rs)))]
                      (recur (rest rs) (drop n os) (conj acc (assoc (first rs) :python (vec (take n os))))))))]
    (remove-ns (ns-name ns))
    (spit "test/fixtures/fuzz.edn"
          (str ";; Written by memo.fuzz/record-regressions! from Python memo: do not edit.\n"
               (with-out-str (clojure.pprint/pprint {:oracle {:memo "1.3.2"} :regressions entries}))))
    entries))

(defn replay
  "A saved regression in memo.clj: [agree? memo.clj's outcome]."
  [{:keys [form calls python]}]
  (let [ns (world-ns)
        ours (memo-clj-outcome ns {:form form :calls calls})]
    (remove-ns (ns-name ns))
    [(first (agree? ours python)) ours]))
