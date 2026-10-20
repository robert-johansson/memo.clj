(ns memo.demos.demo-23-test
  "memo's demo/demo-23.ipynb: guess 2/3 of the average, by level-k
  reasoning; the demo then fits a mixture of levels to NYT data by gradient
  descent, and its calls outside jax.grad are recorded."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def N (vec (range 101)))   ; the space of possible guesses

(defmemo player [[n N]] [level beta]
  (reader :thinks
    (everyone_else :chooses [n N] :wpp (if (> level 0) (player [n] (- level 1) beta) 1)))
  (reader :chooses [n N] :wpp (exp (* beta (- (abs (- n (* (/ 2 3) (E everyone_else.n))))))))
  (return (Pr (= reader.n n))))

(deftest demo-23
  (ts/check-demo 'memo.demos.demo-23-test "demo-23"))
