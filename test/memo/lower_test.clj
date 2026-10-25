(ns memo.lower-test
  "The exact lowering's memory and its constants.

  Memory: demo-empowerment's tables are 714 MB each in float64, and the JVM
  backend ran out of a 16 GB heap computing it while the lowering kept every
  value to the end of a model and spot-checked a vectorized function by
  building its arguments whole on the host. Its test
  (memo.demos.demo-empowerment-test, ^:slow) fails without either fix; these
  check the pieces.

  Constants: what does not depend on a call is computed once per backend
  (memo.lower/program). A host function's table is not among them, and a
  constant computed under a tracer (the batch tracer, a gradient, MLX's
  compile) must serve later calls outside it."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.lower :as lower]
            [memo.test-support :refer [close? host]]))

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

;; ---------------------------------------------------------------------------
;; Constants

(defenum Size SMALL MEDIUM LARGE)

(defn- softmax [beta xs]
  (let [w (mapv #(Math/exp (* beta %)) xs)
        z (reduce + w)]
    (mapv #(/ % z) w)))

;; scaled reads an atom and calls a helper: a change to either changes its
;; values with nothing in the memo changed
(def scale (atom 1.0))
(defn offset [] 0.0)
(defn scaled [s] (* @scale (+ s (offset))))

(defmemo by-scale [[a Size]] [beta]
  (agent :chooses [s Size] :wpp (exp (* beta (scaled s))))
  (return (Pr (= agent.s a))))

(deftest host-functions-are-called-at-every-call
  (try
    (is (close? (softmax 1.0 [0 1 2]) (host (by-scale 1.0))))
    (reset! scale 2.0)
    (is (close? (softmax 1.0 [0 2 4]) (host (by-scale 1.0))) "an atom it reads")
    (with-redefs [offset (constantly 1.0)]
      (is (close? (softmax 1.0 [2 4 6]) (host (by-scale 1.0))) "a helper it calls, redefined"))
    (finally (reset! scale 1.0))))

;; One memo per test below, so that its first call is the one under the
;; tracer. Their constants are the domain's values.
(defmemo on-backends [[a Size]] [beta]
  (agent :chooses [s Size] :wpp (exp (* beta s)))
  (return (Pr (= agent.s a))))

(defmemo batched-first [[a Size]] [beta]
  (agent :chooses [s Size] :wpp (exp (* beta s)))
  (return (Pr (= agent.s a))))

(defmemo grad-first [[a Size]] [beta]
  (agent :chooses [s Size] :wpp (exp (* beta s)))
  (return (Pr (= agent.s a))))

(defmemo compiled-first [[a Size]] [beta]
  (agent :chooses [s Size] :wpp (exp (* beta s)))
  (return (Pr (= agent.s a))))

(deftest constants-are-per-backend
  ;; the JVM backend and the one under test (MLX's, under :mlx), alternately
  (let [bs (distinct [(array/backend-named :jvm) array/*backend*])]
    (doseq [b (concat bs bs)]
      (array/with-backend b
        (let [t (on-backends (array/array 1.0))]
          (is (= (array/backend t) b))
          (is (close? (softmax 1.0 [0 1 2]) (host t))))))))

(deftest constants-made-under-a-tracer-serve-later-calls
  (testing "the batch tracer"
    (is (close? (mapv #(softmax % [0 1 2]) [0.5 1.5])
                (host ((array/vmap batched-first) (array/array [0.5 1.5])))))
    (is (close? (softmax 1.0 [0 1 2]) (host (batched-first (array/array 1.0))))))
  (testing "a gradient: d E[s] / d beta is s's variance"
    (let [p (softmax 1.0 [0 1 2])
          m (reduce + (map * p [0 1 2]))]
      (is (close? (reduce + (map #(* %1 (Math/pow (- %2 m) 2)) p [0 1 2]))
                  (host ((array/grad #(array/sum (array/* (grad-first %) (array/array [0.0 1.0 2.0]))))
                         (array/array 1.0))))))
    (is (close? (softmax 0.5 [0 1 2]) (host (grad-first (array/array 0.5))))))
  (testing "array/compile (on MLX, a trace)"
    (is (close? (softmax 1.0 [0 1 2]) (host ((array/compile compiled-first) (array/array 1.0)))))
    (is (close? (softmax 0.5 [0 1 2]) (host (compiled-first (array/array 0.5)))))))
