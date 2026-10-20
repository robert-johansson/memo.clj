(ns memo.demos.demo-ultimatum-test
  "memo's demo/demo-ultimatum.ipynb: the ultimatum game, with softmax
  players."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(def Proposal (mapv #(/ % 99.0) (range 100)))   ; np.linspace(0, 1, 100)
(defenum Decision Accept Reject)

(defn payout-offerer [prop dec] (nth [(- 1 prop) 0] (long dec)))
(defn payout-receiver [prop dec] (nth [prop 0] (long dec)))

(defmemo receiver [[prop Proposal] [dec Decision]] []
  "The probability that the receiver accepts a proposal."
  (receiver :knows prop)
  (receiver :chooses [dec Decision] :wpp (exp (* 50.0 (payout-receiver prop dec))))
  (return (Pr (= receiver.dec dec))))

(defmemo offerer [[prop Proposal]] []
  "The probability that the offerer proposes a proposal."
  (offerer :chooses [prop Proposal] :wpp (exp (* 50.0 (imagine
                                                       (receiver :knows prop)
                                                       (receiver :chooses [dec Decision] :wpp (receiver [prop dec]))
                                                       (E (payout-offerer prop receiver.dec))))))
  (return (Pr (= offerer.prop prop))))

(deftest demo-ultimatum
  (ts/check-demo 'memo.demos.demo-ultimatum-test "demo-ultimatum"))
