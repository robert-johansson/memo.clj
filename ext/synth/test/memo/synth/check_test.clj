(ns memo.synth.check-test
  "The shape checks (memoclj-k7a6) on the programs Qwen submitted in repl1
  (memoclj-pcw0, 2026-11-04): the shortcuts the grader passed are found, the
  models that reason as asked are not, and every reference passes its own
  spec. Then the tools: check! names the shortcut, submit!'s grading carries
  it."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.synth.check :as check]
            [memo.synth.tasks :as tasks]
            [memo.synth.tools :as tools]))

(defn- load-code
  "Evaluates code's forms in a fresh namespace; the namespace."
  [code]
  (let [n (check/fresh-ns)
        ev (check/evaluate n (check/read-forms code n))]
    (is (:ok ev) (str "the program loads: " ev))
    n))

(defn- shape-of [code sym spec]
  (let [n (load-code code)]
    (check/shape sym @(ns-resolve n sym) spec)))

(def ^:private kinds (comp set (partial map :kind)))

;; repl1's submissions, as session/program gave them (the last definition of each)

(def sally-shortcut
  "(defenum Loc BASKET BOX)
(defmemo sally [[l Loc]] [p]
  (sally :thinks (anne :chooses [moved Bool] :wpp (if moved p (- 1 p))))
  (sally :chooses [where Loc] :to-maximize (if (= p (- 1 p)) 1 (if (= where Loc.BASKET) (- 1 p) p)))
  (return (Pr (= sally.where l))))")

(def monty-shortcut
  "(def Door [0 1 2])
(defmemo monty [[d Door]] []
  (prize :chooses [p Door] :uniformly)
  (monty :knows prize.p)
  (monty :chooses [m Door] :wpp (if (and (not= m 0) (not= m prize.p)) 1 0))
  (return (/ (Pr (= prize.p d) (= monty.m 2)) (Pr (= monty.m 2)))))")

(def forecast-in-host
  "(defenum Act QUIET SPEAK)
(defn alice-expectation [act leak harsh p0] (if (== act 0) p0 (* leak p0)))
(defmemo forecast [[a Act]] [leak harsh p0] (return (alice-expectation a leak harsh p0)))")

(def coin-genuine
  "(def Bias [0.1 0.3 0.5 0.7 0.9])
(defmemo coin [] []
  (bob :thinks (bob :chooses [bias Bias] :uniformly)
               (bob :chooses [outcome Bool] :wpp (if (= outcome Bool.YES) bias (- 1 bias))))
  (bob :observes-that (= bob.outcome Bool.YES))
  (return (as bob (E bob.bias))))")

(def act-genuine
  "(defenum Action WAIT JUMP RUN)
(defn utility [action] (condp = action (Action :WAIT) 0 (Action :JUMP) 1 (Action :RUN) 3))
(defmemo act [[a Action]] [beta] (alice :chooses [a Action] :wpp (exp (* beta (utility a)))) (return (Pr (= alice.a a))))")

(def taste-observer
  "(defenum Taste SWEET SOUR)
(defenum Fruit APPLE BANANA CHERRY)
(defn utility [taste fruit] (nth (nth [[1 2 0] [2 0 1]] taste) fruit))
(defmemo taste [[f Fruit] [t Taste]] [beta]
  (observer :thinks (child :chooses [t Taste] :uniformly)
                    (child :chooses [f Fruit] :wpp (exp (* beta (utility t f)))))
  (observer :observes child.f :is f)
  (observer :knows t)
  (return (as observer (Pr (= child.t t)))))")

(def forecast-half
  "Alice's mind modeled, Bob's posterior computed by a host function: one
  agent reads its beliefs where the task asks for two."
  "(defenum Act QUIET SPEAK)
(defenum Sound SILENT STEADY SHAKY)
(defn posterior [s p0] (if (== s 2) (* 0.5 p0) p0))
(defmemo forecast [[a Act]] [leak harsh p0]
  (alice :knows a)
  (alice :thinks (voice :knows a)
                 (voice :chooses [s Sound] :wpp (if (== a 0) (= s 0) (if (= s 2) leak (if (= s 1) (- 1 leak) 0)))))
  (return (as alice (E (posterior voice.s p0)))))")

(deftest shortcuts-are-found
  (testing "sally's choice reads no belief: a formula in p"
    (let [ps (shape-of sally-shortcut 'sally {:beliefs 1})]
      (is (= #{:beliefs-unused} (kinds ps)))
      (is (str/includes? (:text (first ps)) "no agent reads its beliefs"))
      (is (= :shortcut (tools/reasoning ps)))))
  (testing "monty: Bayes' rule at the top level, the contestant's mind gone"
    (is (= #{:beliefs-unused} (kinds (shape-of monty-shortcut 'monty {:beliefs 1})))))
  (testing "forecast and implicature: the whole model in host functions"
    (let [ps (shape-of forecast-in-host 'forecast {:beliefs 2})]
      (is (= #{:no-agents} (kinds ps)))
      (is (= :shortcut (tools/reasoning ps)))))
  (testing "one agent's belief where the task asks for two"
    (let [ps (shape-of forecast-half 'forecast {:beliefs 2})]
      (is (= #{:beliefs-unused} (kinds ps)))
      (is (= ['alice] (:reading (first ps))))
      (is (str/includes? (:text (first ps)) "only alice reads its beliefs")))))

(deftest reasoning-as-asked-passes
  (testing "a belief read outside its choice, (as bob (E …))"
    (is (= [] (shape-of coin-genuine 'coin {:beliefs 1}))))
  (testing "a task that asks for no beliefs: an agent choosing is enough"
    (is (= [] (shape-of act-genuine 'act {}))))
  (testing "a memo with no agents fails even when the task asks for no beliefs"
    (is (= #{:no-agents} (kinds (shape-of forecast-in-host 'forecast {})))))
  (testing "an agent named observer: reported, its beliefs not checkable"
    (let [ps (shape-of taste-observer 'taste {:beliefs 1})]
      (is (= #{:observer-name :unchecked} (kinds ps)))
      (is (= :unchecked (tools/reasoning ps))))))

(deftest belief-use-by-agent
  (let [n (load-code (:reference (first (filter #(= :sally (:id %)) tasks/tasks))))
        m @(ns-resolve n 'sally)]
    (is (= :choice (check/belief-use m 'sally)) "sally's choice weighs a query")
    (is (= :none (check/belief-use m 'anne)) "anne's choice is a coin flip"))
  (let [n (load-code (:reference (first (filter #(= :forecast (:id %)) tasks/tasks))))
        m @(ns-resolve n 'forecast)]
    (is (= :outside (check/belief-use m 'alice)))
    (is (= :outside (check/belief-use m 'bob)))))

(deftest references-pass-their-own-spec
  (let [ts (tasks/with-expected tasks/tasks)]
    (is (= 9 (count ts)))
    (is (= 2 (get-in (first (filter #(= :forecast (:id %)) ts)) [:signatures 'forecast :beliefs])))
    (is (= 0 (get-in (first (filter #(= :sally (:id %)) ts)) [:signatures 'marble :beliefs])))))

(deftest tools-report-the-shortcut
  (let [{id :session} (tools/open! :sally)]
    (is (= :ok (:kind (tools/eval! id sally-shortcut))))
    (is (= :ok (:kind (tools/eval! id "(defmemo marble [[l Loc]] [p]
  (anne :chooses [moved Bool] :wpp (if moved p (- 1 p)))
  (return (Pr (= (if anne.moved Loc.BOX Loc.BASKET) l))))"))))
    (let [c (tools/check! id)]
      (is (false? (:passed? c)))
      (is (str/includes? (:text c) "sally: no agent reads its beliefs"))
      (is (str/includes? (:text c) "marble: signature as asked")))
    (let [{:keys [grading]} (tools/submit! id)]
      (is (= :correct (:outcome grading)) "the tables are right")
      (is (= :shortcut (:reasoning grading)) "the reasoning is not")
      (is (= [{:memo 'sally :kind :beliefs-unused :reading []}] (:shape grading))))))

(deftest a-missing-memo-is-no-model
  (let [{id :session} (tools/open! :compete)]
    (is (= :ok (:kind (tools/eval! id "(def A [0 1 2])"))))
    (let [{:keys [grading]} (tools/submit! id)]
      (is (= :missing (:outcome grading)))
      (is (= :no-model (:reasoning grading))))))
