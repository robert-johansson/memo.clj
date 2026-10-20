(ns memo.lower-test
  "The exact lowering's memory. demo-empowerment's tables are 714 MB each in
  float64, and the JVM backend ran out of a 16 GB heap computing it while
  the lowering kept every value to the end of a model and spot-checked a
  vectorized function by building its arguments whole on the host. Its test
  (memo.demos.demo-empowerment-test, ^:slow) fails without either fix; these
  check the pieces."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.lower :as lower]))

(deftest values-are-dropped-after-their-last-use
  (let [instrs [{:i :const :out 'a :value 1.0}
                {:i :const :out 'b :value 2.0}
                {:i :op :op :add :out 'c :args ['a 'b]}
                {:i :const :out 'unused :value 3.0}
                {:i :op :op :mul :out 'd :args ['c 'a]}
                {:i :op :op :mul :out 'e :args ['d 'd]}]
        drops #(mapv set (#'lower/drops instrs %))]
    (testing "each value goes after the instruction that reads it last; one never read, at once"
      (is (= [#{} #{} #{'b} #{'unused} #{'a 'c} #{'d}] (drops ['e]))))
    (testing "results are kept"
      (is (= [#{} #{} #{'b} #{'unused} #{'a} #{'d}] (drops ['e 'c]))))))

(deftest the-spot-check-reads-only-its-samples
  (testing "samples are the elements at those indices, x broadcast to shape"
    (doseq [[x shape] [[(array/reshape (array/arange 6) [2 3]) [4 2 3]]
                       [(array/reshape (array/arange 4) [4 1 1]) [4 2 3]]
                       [(array/arange 3) [4 2 3]]
                       [(array/reshape (array/arange 8) [2 1 4]) [2 3 4]]
                       [(array/reshape (array/arange 6.0) [3 2]) [3 2]]
                       [(array/array [true false]) [3 2]]
                       [2.5 [3 2]]
                       [(array/array 7.0) []]]
            :let [all (#'lower/elements x shape)
                  n (count all)]
            idxs [(vec (range n)) [0 (quot n 2) (dec n)] [(dec n)]]]
      (is (= (mapv all idxs) (#'lower/samples x shape idxs))
          (str "x of shape " (array/shape x) " broadcast to " shape " at " idxs)))))
