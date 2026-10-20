(ns memo.demos.demo-monty-test
  "memo's demo/demo-monty.ipynb: the Monty Hall problem."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def Door [0 1 2])

(defmemo monty [[pick Door] [reveal Door] [d Door]] []
  (alice :thinks (monty :chooses [prize Door] :wpp 1))
  (alice :knows pick)
  (alice :thinks
    (monty :knows pick)
    (monty :chooses [reveal Door] :wpp (and (not= reveal prize) (not= reveal pick))))
  (alice :observes monty.reveal :is reveal)
  (alice :knows d)
  (return (as alice (Pr (= monty.prize d)))))

(deftest demo-monty
  (ts/check-demo 'memo.demos.demo-monty-test "demo-monty"))
