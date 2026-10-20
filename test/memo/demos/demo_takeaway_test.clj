(ns memo.demos.demo-takeaway-test
  "memo's demo/demo-takeaway.ipynb: the game of take-away, and a lab that
  infers whether a player thinks or takes a shortcut. The demo's model EU
  is expected-utility here: (EU ...) is memo's operator in memo.clj's
  syntax, where Python memo tells EU[g] from a call EU[n](t) by the call.
  quick weighs moves by cost @ π(t), cost reflection, which is M4."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(def N (vec (range 21)))
(defenum Move [ONE 1] [TWO 2])

(defmemo ^:cache expected-utility [[n N]] [t]
  (player :knows n)
  (return (if (= n 0)
            -1
            (if (<= t 0)
              0
              (imagine
               (player :chooses [m Move] :wpp (* (>= n m) (π [n m] t)))
               (player :chooses [n_ N] :wpp (= n_ (- n m)))
               (- (E (expected-utility [player.n_] (- t 1)))))))))

(defmemo ^:cache π [[n N] [m Move]] [t]
  (player :knows n)
  (player :chooses [m Move] :wpp (* (>= n m) (exp (* 10 (if (<= t 0)
                                                          0
                                                          (imagine
                                                           (env :knows n m)
                                                           (env :chooses [n_ N] :wpp (= n_ (- n m)))
                                                           (- (E (expected-utility [env.n_] (- t 1))))))))))
  (return (Pr (= player.m m))))

(deftest demo-takeaway
  (ts/check-demo 'memo.demos.demo-takeaway-test "demo-takeaway"
                 {:renamed {:EU 'expected-utility}
                  :not-ported {:quick "cost reflection (cost @ π(t)) is M4 (DESIGN.md §1)"}}))
