(ns memo.slice5-test
  "M1, slice 5: imagine, snapshots-self-as, wants, EU and Predict. Every model
  here has a twin of the same name in test/fixtures/generate.py, and
  slice5.edn holds Python memo's results, errors and a gradient."
  (:require [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(def B2 [0 1])
(defenum Bool NO YES)

(defmemo observes-other-imagine [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp 1)
  (alice :observes bob.x :is charlie.x)
  (return (E (as alice (imagine (env :knows bob.x) (+ bob.x 1))))))

(defmemo imagine-ok [] []
  (return (as alice (imagine (bob :chooses [y X] :wpp 1) (E bob.y)))))

(defmemo imagine-knows [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (as alice (imagine (world :knows x) world.x)))))

(defmemo imagine-knows-other [[z X]] []
  (alice :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [z X] :wpp 1))
  (alice :observes bob.z :is z)
  (return (as alice (imagine (world :knows bob.z) (as world bob.z)))))

(defmemo imagine-future-stress [] []
  (alice :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [z X] :wpp 1))
  (alice :snapshots-self-as future_alice)
  (return (E (as alice
                 (imagine
                  (world :knows x bob.z)
                  (world :chooses [z X] :wpp 1)
                  (future_alice :chooses [y X] :wpp (+ x y))
                  (future_alice :thinks (world :chooses [z X] :wpp 1))
                  (future_alice :observes world.z :is world.z)
                  (E (+ future_alice.y (as future_alice (+ world.z y)) world.z (as world bob.z))))))))

(defmemo imagine-toplevel [] []
  (return (imagine (alice :chooses [x X] :wpp 1) (E alice.x))))

(defmemo predict-own [] []
  (alice :chooses [x X] :to-maximize (Predict x))
  (return (E alice.x)))

(defmemo predict [] []
  (bob :chooses [z X] :wpp 1)
  (bob :thinks
    (alice :chooses [x X] :to-maximize (Predict z))
    (alice :chooses [y X] :to-maximize (Predict z))
    (alice :thinks (world :knows x y) (world :chooses [z X] :wpp (+ z x y)))
    (alice :observes world.z :is z)
    (alice :chooses [z X] :wpp (+ world.z z)))
  (return (E (as bob (E alice.x)))))

(defmemo two-goals [[x X] [y X]] []
  (bob :thinks
    (alice :wants u (+ x y))
    (alice :wants v (- x y))
    (alice :chooses [x X] :to-maximize (EU u))
    (alice :chooses [y X] :to-maximize (EU v)))
  (bob :knows x y)
  (return (as bob (Pr (= alice.x x) (= alice.y y)))))

(defenum Cups Near_Vizzini Near_Westley)

(defmemo vizzini-pick [[cup Cups]] [level]
  (vizzini :wants survive (!= my_cup westley.poison))
  (vizzini :thinks
    (westley :wants kill (= vizzini.my_cup poison))
    (westley :chooses [poison Cups] :to-maximize (EU kill))
    (westley :thinks
      (vizzini :chooses [my_cup Cups]
               :wpp (if (> level 0) (vizzini-pick [my_cup] (- level 1)) (= my_cup Cups.Near_Vizzini)))))
  (vizzini :chooses [my_cup Cups] :to-maximize (EU survive))
  (return (Pr (= vizzini.my_cup cup))))

(def Value (mapv #(/ % 100.0) (range 101)))

(defn ber
  "The Bernoulli pmf of k, as jax.scipy.stats.bernoulli.pmf (p may be a boolean)."
  [k p]
  (let [p (if (boolean? p) (if p 1.0 0.0) p)]
    (if (== k 1) p (- 1.0 p))))

(defmemo lemons-buyer [] []
  (buyer :wants payoff (if seller.sell (- (* seller.quality 1.5) bid) 0))
  (buyer :chooses [bid Value] :to-maximize (EU payoff))
  (buyer :thinks
    (seller :given [quality Value] :wpp 1)
    (seller :knows bid)
    (seller :chooses [sell Bool] :wpp (ber sell (>= bid quality))))
  (return (E buyer.bid)))

(defenum Box FST_BOX SND_BOX)
(defenum FoodStatus EMPTY_BOX FOODY_BOX)
(defenum EvidenceType CRUMBS NOISES VISUAL)

(defn p-evidence [et fs] (get-in [[0.5 0.6] [0.3 0.6] [0.01 0.9]] [et fs]))

(defmemo first-choice [[et1 EvidenceType] [b Box]] []
  (chimp :knows et1)
  (chimp :wants get_food (= (if (= b Box.FST_BOX) world.box1_fs world.box2_fs) FoodStatus.FOODY_BOX))
  (chimp :thinks
    (world :chooses [box1_fs FoodStatus] :wpp 1)
    (world :chooses [box2_fs FoodStatus] :wpp 1)
    (world :knows et1)
    (world :chooses [e1 FoodStatus] :wpp (ber e1 (p-evidence et1 box1_fs))))
  (chimp :observes-that (= world.e1 FoodStatus.FOODY_BOX))
  (chimp :chooses [b Box] :wpp (exp (* 10.0 (EU get_food))))
  (return (Pr (= chimp.b b))))

(defenum Vocab Ball Bear Duck Milk Shoe)

(defmemo novel-event-inform [[fixed_event Vocab] [novel_event Vocab] [utterance Vocab]] [beta]
  (baby :knows fixed_event novel_event)
  (baby :wants inform (Pr (= parent.referent_guess novel_event)))
  (baby :thinks
    (parent :thinks
      (scene :chooses [fixed_event Vocab] :wpp 1)
      (scene :chooses [novel_event Vocab] :wpp 1))
    (parent :observes scene.fixed_event :is fixed_event)
    (parent :thinks
      (baby :knows scene.fixed_event scene.novel_event)
      (baby :chooses [referent Vocab] :wpp (or (= referent scene.fixed_event) (= referent scene.novel_event)))
      (baby :chooses [utterance Vocab] :wpp (+ (= utterance referent) 0.25))))
  (baby :chooses [utterance Vocab] :wpp (exp (* beta (EU inform))))
  (baby :thinks
    (parent :observes baby.utterance :is utterance)
    (parent :guesses [referent_guess Vocab] :wpp (Pr (= referent_guess baby.referent))))
  (return (Pr (= baby.utterance utterance))))

(defenum Sp Hungry Sated)
(defenum Ap Feed Sing Ignore)
(defenum Op Crying Quiet)
(def Bp (mapv #(/ % 49.0) (range 50)))

(defn get-belief [b s] (nth [b (- 1 b)] s))
(defn tr [s a s_] (let [z (get-in [[0.0 1.0 1.0] [0.0 0.1 0.1]] [s a])] (nth [z (- 1 z)] s_)))
(defn obs-p [o s a] (let [z (get-in [[0.8 0.9 0.8] [0.1 0.0 0.1]] [s a])] (nth [z (- 1 z)] o)))
(defn reward [s a] (+ (nth [-10 0] s) (nth [-5 -0.5 0] a)))

;; demo-pomdp.ipynb
(defmemo Q-pomdp [[b Bp] [a Ap]] [t]
  (alice :knows b a)
  (alice :thinks
    (env :knows b a)
    (env :chooses [s Sp] :wpp (get-belief b s))
    (env :chooses [s_ Sp] :wpp (tr s a s_))
    (env :chooses [o Op] :wpp (obs-p o s_ a)))
  (alice :snapshots-self-as future_alice)
  (return (as alice (+ (E (reward env.s a))
                       (if (<= t 0)
                         0.0
                         (* 0.9 (imagine
                                 (future_alice :observes env.o :is env.o)
                                 (future_alice :chooses [b_ Bp] :wpp (exp (* -100.0 (abs (- (E (= env.s_ 0)) b_)))))
                                 (future_alice :chooses [a_ Ap] :to-maximize (Q-pomdp [b_ a_] (- t 1)))
                                 (E (as future_alice (Q-pomdp [b_ a_] (- t 1)))))))))))

;; the snapshot bug (DESIGN.md §2.4): Python memo got these wrong until 1.3.3

(defmemo snap-f [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E bob.x)))))

(defmemo snap-g [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1) (bob :chooses [y X] :wpp (= y x)))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E (+ bob.x bob.y))))))

(defmemo snap-pr [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (Pr (= bob.x 2))))))

(defmemo eu-leak [] []
  (world :chooses [s B2] :wpp 1)
  (alice :thinks (world :chooses [s B2] :wpp (+ 1 (* 2 s))))
  (alice :wants win (= world.s g))
  (alice :chooses [look B2] :to-maximize (- (EU win) (* 0.1 look)))
  (alice :chooses [m B2] :wpp (exp (* (E world.s) m)))
  (alice :observes world.s :is world.s)
  (alice :chooses [g B2] :wpp (Pr (= world.s g)))
  (return (Pr (= alice.m 1))))

(def fixture (ts/fixture "slice5.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.slice5-test fixture))

(deftest gradients-match-jax
  (ts/check-grads 'memo.slice5-test fixture))

(def error-forms
  {:imagine-unknown-err
   '(defmemo imagine-unknown-err [] []
      (return (as alice (imagine (bob :chooses [y X] :wpp 1) bob.y))))
   :predict-fail-knows
   '(defmemo predict-fail-knows [] []
      (bob :chooses [z X] :wpp 1)
      (bob :thinks
        (alice :chooses [x X] :to-maximize (Predict x))
        (alice :knows z))
      (return (E (as bob (E alice.x)))))
   :unknown-goal
   '(defmemo unknown-goal [] []
      (alice :chooses [x X] :to-maximize (EU g))
      (return 1))
   :no-wants-err
   '(defmemo ^{:variant :no-wants} no-wants-err [] []
      (alice :wants g 1)
      (return 0))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.slice5-test (error-forms k)) ex-message))))))

(deftest hinted-where-python-memo-asserts
  ;; Python memo raises a bare AssertionError for each of these
  (testing "Predict at the top level"
    (is (re-find #"cannot be used at the top level"
                 (ex-message (ts/compile-error 'memo.slice5-test
                                               '(defmemo top-predict [] []
                                                  (alice :chooses [x X] :wpp 1)
                                                  (return (Predict alice.x))))))))
  (testing "a goal named twice"
    (is (= "alice already wants g"
           (ex-message (ts/compile-error 'memo.slice5-test
                                         '(defmemo twice [] []
                                            (alice :wants g 1)
                                            (alice :wants g 2)
                                            (alice :chooses [x X] :to-maximize (EU g))
                                            (return 1))))))))

(deftest copies-keep-their-own-values
  ;; memo-clj's test of the same name: an observation reaches a knows copy
  ;; made before it, and never a snapshot
  (is (ts/close? [1.0 1.0 1.0] (ts/host (snap-f))))
  (is (ts/close? [2.0 2.0 2.0] (ts/host (snap-g)))))
