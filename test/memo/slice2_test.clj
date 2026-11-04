(ns memo.slice2-test
  "M1, slice 2: thinks, observes, observes-that and observes-event, knows,
  has-theory. Every model here has a twin of the same name in
  test/fixtures/generate.py, and slice2.edn holds Python memo's results."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(defenum Bool NO YES)
(defenum Coin HEADS TAILS)

(defmemo observes-other [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp 1)
  (alice :observes bob.x :is charlie.x)
  (return (E (as alice bob.x))))

(defmemo observes-that-const [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :observes-that (= bob.x 0))
  (return (as alice (E bob.x))))

(defmemo observes-event-float [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :observes-event :wpp (/ bob.x 3))
  (return (as alice (E bob.x))))

(defmemo observes-event-own [] []
  (alice :chooses [x X] :wpp 1)
  (alice :observes-event :wpp (/ x 3))
  (return (E alice.x)))

(defmemo observes-event-void [] []
  (alice :observes-event :wpp 3.14)
  (return (as alice 2)))

(defmemo post-diag-1 [[x X]] []
  (bob :knows x)
  (bob :chooses [y X] :wpp (+ x y))
  (return (Pr (= bob.y x))))

(defmemo post-diag-2 [[x X]] []
  (bob :thinks (alice :chooses [x X] :wpp 1))
  (bob :observes alice.x :is x)
  (bob :chooses [y X] :wpp (+ alice.x y))
  (return (Pr (= bob.y x))))

(defmemo post-diag-3 [[x X]] []
  (bob :chooses [x X] :wpp 1)
  (bob :chooses [y X] :wpp (+ x y))
  (return (Pr (= bob.y x))))

(defmemo jianlin [[x X]] []
  (a :thinks (b :chooses [u X] :wpp 1.0))
  (a :observes b.u :is x)
  (a :chooses [v X] :wpp (= b.u v))
  (return (Pr (= a.v x))))

(defmemo var-redundant-known [] []
  (alice :chooses [coffee Bool] :uniformly)
  (bob :knows alice.coffee)
  (return (E (as bob (Var (= alice.coffee 1))))))

(defmemo child-thinks-ok [] []
  (child :has-theory :child)
  (child :thinks (adult :chooses [x X] :wpp 1))
  (return (as child (E adult.x))))

(defmemo ^{:variant :child} child-toplevel [] []
  (child :chooses [x X] :uniformly)
  (return (E child.x)))

(defmemo knows-copy [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :thinks (charlie :knows bob.x))
  (alice :observes bob.x :is x)
  (return (as alice (as charlie bob.x))))

(defmemo via-knows [[y X]] []
  (bob :chooses [y X] :wpp (+ y 1))
  (alice :knows bob.y)
  (alice :chooses [z X] :wpp (exp (* bob.y z)))
  (return (Pr (= alice.z y))))

(defmemo via-observes [[y X]] []
  (bob :chooses [y X] :wpp (+ y 1))
  (alice :thinks (bob :chooses [y X] :wpp 1))
  (alice :observes bob.y :is bob.y)
  (alice :chooses [z X] :wpp (exp (* bob.y z)))
  (return (Pr (= alice.z y))))

(defmemo impossible-obs [] []
  (bob :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [x X] :wpp (< x 2)))
  (alice :observes bob.x :is bob.x)
  (return (E (as alice (Pr (= bob.x 2))))))

(defmemo impossible-obs-ll [] []
  (bob :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [x X] :wpp (< x 2))
                 (bob :chooses [y X] :wpp (+ (= y x) 1.0)))
  (alice :observes bob.x :is bob.x)
  (return (E (as alice (E bob.y)))))

(defmemo scramble-unread [[s X] [t X]] []
  (obs :knows t)
  (obs :knows s)
  (obs :thinks (w :knows s) (w :chooses [x X] :wpp (if (= x s) 2 1)))
  (obs :observes w.x :is s)
  (return (as obs (Pr (= w.x t)))))

(def Door [0 1 2])

(defmemo monty-hall [[pick Door] [reveal Door] [d Door]] []
  (alice :thinks (monty :chooses [prize Door] :wpp 1))
  (alice :knows pick)
  (alice :thinks
    (monty :knows pick)
    (monty :chooses [reveal Door] :wpp (and (!= reveal prize) (!= reveal pick))))
  (alice :observes monty.reveal :is reveal)
  (alice :knows d)
  (return (as alice (Pr (= monty.prize d)))))

(defenum Opinion BAD_POLICY GOOD_POLICY)
(defenum Outcome NO_SPENDING SPENDING_INCREASE)
(defenum Result NO_SPENDING SPENDING_INCREASE)
(defenum Policy CONSERVATIVE LIBERAL)
(defenum Economist ALICE BOB)

(defmemo polarization [[e Economist] [d Result]] []
  (economist :knows e)
  (economist :thinks
    (world :knows e)
    (world :chooses [v1 Policy]
           :wpp (if (= e Economist.ALICE)
                  (if (= v1 Policy.LIBERAL) 0.9 0.1)
                  (if (= v1 Policy.LIBERAL) 0.1 0.9)))
    (world :chooses [v2 Outcome] :wpp 1)
    (world :chooses [d Result] :wpp (if (= d v2) 0.9 0.1))
    (world :chooses [h Opinion]
           :wpp (if (= v2 Outcome.NO_SPENDING)
                  0.5
                  (if (= v1 Policy.CONSERVATIVE)
                    (if (= h Opinion.GOOD_POLICY) 0.1 0.9)
                    (if (= h Opinion.GOOD_POLICY) 0.9 0.1)))))
  (economist :observes world.d :is d)
  (return (as economist (Pr (= world.h 1)))))

(defenum Loc BOX BASKET)
(defenum Action ACT_STAY ACT_MOVE)
(defenum Obs [OBS_NONE -1] [OBS_STAY 0] [OBS_MOVE 1])

(defmemo sally-anne [[marble_pos_t0 Loc] [obs Obs] [where_look Loc]] []
  (child :knows marble_pos_t0 obs where_look)
  (child :thinks
    (sally :knows marble_pos_t0)
    (sally :thinks
      (anne :knows marble_pos_t0)
      (anne :chooses [a Action] :wpp (if (= a Action.ACT_MOVE) 0.01 0.99))
      (anne :chooses [marble_pos_t1 Loc]
            :wpp (= (if (= a Action.ACT_STAY) marble_pos_t0 (- 1 marble_pos_t0)) marble_pos_t1))
      (anne :chooses [o Obs] :wpp (if (or (= o Obs.OBS_NONE) (= o a)) 1 0)))
    (sally :observes anne.o :is obs)
    (sally :chooses [where_look Loc] :wpp (Pr (= anne.marble_pos_t1 where_look))))
  (return (as child (Pr (= sally.where_look where_look)))))

(def fixture (ts/fixture "slice2.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.slice2-test fixture))

(def error-forms
  {:obs-unmodeled
   '(defmemo obs-unmodeled [[x X]] []
      (alice :observes bob.x :is x)
      (return 1))
   :obs-known
   '(defmemo obs-known [[x X]] []
      (alice :thinks (bob :chooses [x X] :wpp 1))
      (alice :observes bob.x :is x)
      (alice :observes bob.x :is x)
      (return 1))
   :obs-unknown-source
   '(defmemo obs-unknown-source [] []
      (alice :thinks (bob :chooses [x X] :wpp 1))
      (alice :observes bob.x :is z)
      (return 1))
   :obs-domain-mismatch
   '(defmemo obs-domain-mismatch [[c Coin]] []
      (alice :thinks (bob :chooses [x X] :wpp 1))
      (alice :observes bob.x :is c)
      (return 1))
   :knows-unknown
   '(defmemo knows-unknown [] []
      (alice :knows bob.x)
      (return 1))
   :uncertain-wpp
   '(defmemo uncertain-wpp [] []
      (alice :thinks (bob :chooses [y X] :wpp 1))
      (alice :chooses [x X] :wpp bob.y)
      (return 1))
   :ask-unknown
   '(defmemo ask-unknown [] []
      (alice :thinks (bob :chooses [y X] :wpp 1))
      (return (as alice bob.y)))
   :child-thinks-err
   '(defmemo child-thinks-err [] []
      (child :has-theory :child)
      (child :thinks (adult :thinks (mom :chooses [x X] :wpp 1)))
      (return 0))
   :child-toplevel-err
   '(defmemo ^{:variant :child} child-toplevel-err [] []
      (child :thinks (adult :chooses [x X] :wpp 1))
      (return 0))
   :no-thinks-err
   '(defmemo ^{:variant :no-thinks} no-thinks-err [] []
      (alice :thinks (bob :chooses [x X] :wpp 1))
      (return 0))
   :no-knows-err
   '(defmemo ^{:variant :no-knows} no-knows-err [[x X]] []
      (alice :knows x)
      (return 0))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.slice2-test (error-forms k)) ex-message))))))

(deftest observing-a-value-hints-observes-that
  ;; Python memo's message stays; memo.clj's hint names the fix when the value
  ;; observed is an enum member (read as agent.choice) or a parameter
  (let [hint (fn [form] (let [e (ts/compile-error 'memo.slice2-test form)]
                          [(ex-message e) (:memo/hint (ex-data e))]))
        [m h] (hint '(defmemo obs-member [] []
                       (alice :thinks (bob :chooses [c Coin] :wpp 1))
                       (alice :observes bob.c :is Coin.TAILS)
                       (return 1)))]
    (is (= "Observation of unknown choice" m))
    (is (str/includes? h "Coin.TAILS is a member of the domain Coin"))
    (is (str/includes? h "(alice :observes-that (= bob.c Coin.TAILS))"))
    (let [[m h] (hint '(defmemo obs-param [] [k]
                         (alice :thinks (bob :chooses [x X] :wpp 1))
                         (alice :observes bob.x :is k)
                         (return 1)))]
      (is (= "Observation of unknown choice" m))
      (is (str/includes? h "k is a parameter"))
      (is (str/includes? h "(alice :observes-that (= bob.x k))")))
    (testing "inside a mind, the statement suggested is the one written there"
      (let [[_ h] (hint '(defmemo obs-member-nested [] []
                           (carol :thinks
                             (alice :thinks (bob :chooses [c Coin] :wpp 1))
                             (alice :observes bob.c :is Coin.HEADS))
                           (return 1)))]
        (is (str/includes? h "(alice :observes-that (= bob.c Coin.HEADS))"))))
    (testing "a choice that does not exist keeps Python memo's hint"
      (let [[m h] (hint (error-forms :obs-unknown-source))]
        (is (= "Observation of unknown choice" m))
        (is (str/includes? h "Perhaps you misspelled z?"))))))

(deftest scrambled-beliefs-are-a-compile-error
  ;; Python memo answers scramble-unread (above), and crashes with a shape
  ;; error once the scrambled beliefs are read; memo.clj reports the cause
  (let [e (ts/compile-error 'memo.slice2-test
                            '(defmemo scramble-read [[s X]] []
                               (obs :knows s)
                               (obs :thinks (w :knows s)
                                            (w :chooses [x X] :wpp (if (= x s) 2 1))
                                            (w :chooses [y X] :wpp (+ (= y x) 1.0)))
                               (obs :observes w.x :is s)
                               (return (as obs (E w.y)))))]
    (is (= "Observation against an already-used value" (ex-message e)))
    (is (re-find #"observes-that" (:memo/hint (ex-data e))))))
