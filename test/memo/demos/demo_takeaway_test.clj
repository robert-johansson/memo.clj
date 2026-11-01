(ns memo.demos.demo-takeaway-test
  "memo's demo/demo-takeaway.ipynb: the game of take-away, and a lab that
  infers whether a player thinks or takes a shortcut. The demo's model EU
  is expected-utility here: (EU ...) is memo's operator in memo.clj's
  syntax, where Python memo tells EU[g] from a call EU[n](t) by the call.
  quick weighs moves by cost @ π(t), cost reflection (M4, DESIGN.md §14.1):
  Python memo's count is XLA's flops, memo.clj's its own run's, so the table
  is checked with Python's count passed in (quick-python-count, from the
  fixture's :costs), and quick, as written, with memo.clj's count."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is]]
            [memo.core :as memo :refer [defmemo]]
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

(defenum Strategy SHORT THINK)

;; Python memo's cost @ π(t), its XLA flops, recorded by the generator
(def ^:private python-cost-of-π-table
  (let [fixture (edn/read-string (slurp "test/fixtures/demos/demo-takeaway.edn"))
        π-model (first (filter #(= :π (:name %)) (:models fixture)))]
    (into {} (for [{[t] :args time :time} (:costs π-model)] [(long t) time]))))

(defn python-cost-of-π [t]
  (or (get python-cost-of-π-table (long t))
      (throw (ex-info (str "No Python cost of π(" t ") in the fixture") {:t t}))))

;; the lab: does the player think (weigh moves by π) or take a shortcut
;; (weigh them by π scaled by how much thinking π takes)?
(defmemo quick [[m Move] [s Strategy]] [t]
  (lab :chooses [n N] :wpp (= n t))
  (lab :knows s)
  (lab :thinks
    (player :knows n)
    (player :chooses [s Strategy] :wpp 1)
    (player :chooses [m Move] :wpp (if (= s 0) (π [n m] t) (exp (/ (* (π [n m] t) 500000) (cost @(π t)))))))
  (lab :observes player.m :is m)
  (return (E (as lab (Pr (= player.s s))))))

;; the same with Python memo's count in place of cost @ π(t)
(defmemo quick-python-count [[m Move] [s Strategy]] [t]
  (lab :chooses [n N] :wpp (= n t))
  (lab :knows s)
  (lab :thinks
    (player :knows n)
    (player :chooses [s Strategy] :wpp 1)
    (player :chooses [m Move] :wpp (if (= s 0) (π [n m] t) (exp (/ (* (π [n m] t) 500000) (python-cost-of-π t))))))
  (lab :observes player.m :is m)
  (return (E (as lab (Pr (= player.s s))))))

(deftest demo-takeaway
  (ts/check-demo 'memo.demos.demo-takeaway-test "demo-takeaway"
                 {:renamed {:EU 'expected-utility :quick 'quick-python-count}}))

(deftest quick-with-memo-clj's-count
  (doseq [t [1 2 19 20]]
    (let [c (:time (memo/cost π t))]
      (is (pos? c))
      ;; quick as written is quick-python-count with memo.clj's count in Python's place
      (with-redefs [python-cost-of-π (constantly c)]
        (is (ts/close? (ts/host (quick-python-count t)) (ts/host (quick t))) (str "t = " t))))))
