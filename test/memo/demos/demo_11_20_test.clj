(ns memo.demos.demo-11-20-test
  "memo's demo/demo-11-20.ipynb: the 11-20 money request game (Arad &
  Rubinstein), level-k players with a Poisson prior over k."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def N (vec (range 11 21)))
(def K (vec (range 6)))
(def lambda_ 2.36)

(defn poisson-pmf
  "jax.scipy.stats.poisson.pmf at a whole k."
  [k mu]
  (Math/exp (- (* k (Math/log mu)) mu (reduce + (map #(Math/log %) (range 1 (inc (long k))))))))

(defn is-salient [n] (== n 20))

(defmemo game [[k K] [n N]] [(limit 100)]
  (player :knows k)
  (player :wants payoff (+ n (if (= n (- opponent.n 1)) 20 0)))
  (player :thinks
    (opponent :knows k)
    (opponent :given [k_ K] :wpp (* (poisson-pmf k_ #{lambda_}) (< k_ k)))
    (opponent :chooses [n N] :wpp (if (< limit 0)
                                    1
                                    (if (<= k_ 0) (is-salient n) (game [k_ n] (- limit 1))))))
  (player :chooses [n N] :to-maximize (if (> k 0) (EU payoff) (is-salient n)))
  (return (Pr (= player.n n))))

(deftest demo-11-20
  (ts/check-demo 'memo.demos.demo-11-20-test "demo-11-20"))
