(ns memo.tables-test
  "memo.core/rows (M4, DESIGN.md §14.2): a result as data, Python memo's
  return_pandas without a dataframe library."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
            [memo.domains :refer [defenum defdomain]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(defenum Coin T H)
(def N [0 1 2])
(defdomain S :x 2 :y 2)

(defmemo flip [[c Coin]] [p]
  (alice :chooses [c Coin] :wpp (if (== c 1) p (- 1 p)))
  (return (Pr (= alice.c c))))

(defmemo two [[c Coin] [n N]] []
  (return (+ c n))
  (return (* c n)))

(defmemo half [] [] (return 0.5))

(defmemo over-product [[s S]] [] (return 1))

(defn- message [thunk]
  (try (thunk) nil (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(deftest rows-are-the-table-as-data
  (testing "an enum's members by name, the memo's name to the value"
    (is (= [{:c "T" :flip 0.7} {:c "H" :flip 0.3}]
           (mapv #(update % :flip (fn [v] (/ (Math/round (* 1000 (double v))) 1000.0)))
                 (memo/rows flip (flip 0.3))))))
  (testing "a domain of numbers by value; several returns as a vector, in order"
    (is (= [{:c "T" :n 0 :two [0 0]} {:c "T" :n 1 :two [1 0]} {:c "T" :n 2 :two [2 0]}
            {:c "H" :n 0 :two [1 0]} {:c "H" :n 1 :two [2 1]} {:c "H" :n 2 :two [3 2]}]
           (mapv #(update % :two (fn [vs] (mapv long vs))) (memo/rows two (two))))))
  (testing "no axes: one row"
    (is (= [{:half 0.5}] (memo/rows half (half)))))
  (testing "a product domain by its packed value, as Python memo's _vals"
    (is (= [0 1 2 3] (mapv :s (memo/rows over-product (over-product))))))
  (testing "the same cells, in the same order, as print-table"
    (let [printed (with-out-str (memo/print-table flip (flip 0.3)))]
      (is (= ["T" "H"] (map :c (memo/rows flip (flip 0.3)))))
      (is (< (.indexOf printed "| T") (.indexOf printed "| H")))))
  (testing "a var is the memo"
    (is (= (memo/rows flip (flip 0.3)) (memo/rows #'flip (flip 0.3))))))

(deftest rows-read-no-gradient
  (let [g (array/grad (fn [p] (let [r (flip p)] (memo/rows flip r) (array/sum r))))]
    (is (ts/close? 0.0 (ts/host (g (array/array 0.3)))) "rows inside a gradient: no host read refused")))

(deftest rows-errors-are-print-table's
  (is (= "rows takes a memo and its result" (message #(memo/rows inc [1]))))
  (is (= "rows: the result has shape [2 2 3], but flip's results have shape [2]"
         (message #(memo/rows flip (two)))))
  (is (= "rows: the result is :x, not flip's table" (message #(memo/rows flip :x)))))

(defmemo x [[x N]] [] (return (* 2 x)))

(deftest an-axis-named-as-the-memo-is-refused
  ;; Python memo's return_pandas keeps the axis and drops the values
  (is (= "rows: the memo x has an axis named x, so its values and that axis would share the key :x"
         (message #(memo/rows x (x))))))
