(ns memo.demos.demo-lemons-test
  "memo's demo/demo-lemons.ipynb: Akerlof's market for lemons. The demo's
  Bool is the one Python memo predefines in every model; so is memo.clj's."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def Value (mapv #(/ % 100.0) (range 101)))   ; np.linspace(0, 1, 101)

(defn ber
  "jax.scipy.stats.bernoulli.pmf: p for 1, 1 - p for 0; p may be a boolean."
  [k p]
  (let [p (if (boolean? p) (if p 1.0 0.0) p)]
    (if (== k 1) p (- 1 p))))

(defmemo buyer [] []
  (buyer :wants payoff (if seller.sell (- (* seller.quality 1.5) bid) 0))
  (buyer :chooses [bid Value] :to-maximize (EU payoff))
  (buyer :thinks
    (seller :given [quality Value] :wpp 1)
    (seller :knows bid)
    (seller :chooses [sell Bool] :wpp (ber sell (>= bid quality))))
  (return (E buyer.bid)))

(deftest demo-lemons
  (ts/check-demo 'memo.demos.demo-lemons-test "demo-lemons"))
