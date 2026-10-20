(ns memo.demos.demo-schelling-test
  "memo's demo/demo-schelling.ipynb: Schelling coordination by mutual
  recursion. The demo then redefines Bar, prior, alice and bob over 100
  bars; those are Bar-100, prior-100, alice-2 and bob-2 here (the fixture's
  names for the redefined models), each calling the other's new self."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def Bar [0 1])
(defn prior [b] (nth [0.55 0.45] (long b)))

(defmemo alice [[b Bar]] [depth]
  (alice :thinks (bob :chooses [b Bar] :wpp (bob [b] (- depth 1))))
  (alice :chooses [b Bar] :wpp (* (prior b) (Pr (= b bob.b))))
  (return (Pr (= alice.b b))))

(defmemo bob [[b Bar]] [depth]
  (bob :thinks (alice :chooses [b Bar] :wpp (if (> depth 0) (alice [b] depth) 1)))
  (bob :chooses [b Bar] :wpp (* (prior b) (Pr (= b alice.b))))
  (return (Pr (= bob.b b))))

(defmemo alice-confidence [] [depth]
  (alice :thinks (bob :chooses [b Bar] :wpp (bob [b] (- depth 1))))
  (alice :chooses [b Bar] :wpp (* (prior b) (Pr (= b bob.b))))
  (return (E (as alice (Pr (= b bob.b))))))

(defmemo obs-confidence [] [depth]
  (alice :chooses [b Bar] :wpp (alice [b] depth))
  (bob :chooses [b Bar] :wpp (bob [b] depth))
  (return (Pr (= alice.b bob.b))))

(def Bar-100 (vec (range 100)))
(defn prior-100 [b] 1)

(defmemo alice-2 [[b Bar-100]] [depth]
  (alice :thinks (bob :chooses [b Bar-100] :wpp (bob-2 [b] (- depth 1))))
  (alice :chooses [b Bar-100] :wpp (* (prior-100 b) (Pr (= b bob.b))))
  (return (Pr (= alice.b b))))

(defmemo bob-2 [[b Bar-100]] [depth]
  (bob :thinks (alice :chooses [b Bar-100] :wpp (if (> depth 0) (alice-2 [b] depth) 1)))
  (bob :chooses [b Bar-100] :wpp (* (prior-100 b) (Pr (= b alice.b))))
  (return (Pr (= bob.b b))))

(deftest demo-schelling
  (ts/check-demo 'memo.demos.demo-schelling-test "demo-schelling"))
