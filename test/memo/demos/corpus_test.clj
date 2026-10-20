(ns memo.demos.corpus-test
  "The parity corpus, part 3: memo's demos (memo-mcp/memo/demo), each run as
  written by test/fixtures/generate.py, its models' calls recorded in
  test/fixtures/demos/<demo>.edn. Every demo has a namespace of twins,
  memo.demos.<demo>-test, or a reason it has none."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.test-support :as ts]))

(def not-ported
  "Demos whose models cannot run in memo.clj, and why."
  {"demo-7segment" (str "its models call a convolutional network trained with torch, torchvision "
                        "(an EMNIST download), flax and optax as host functions")
   "demo-liu" "its models call jax2d, a JAX rigid-body physics engine, as host functions"
   "demo-physics" "its models call jax2d, a JAX rigid-body physics engine, as host functions"})

(def demos
  (sort (for [f (.list (io/file "test/fixtures/demos")) :when (str/ends-with? f ".edn")]
          (str/replace f #"\.edn$" ""))))

(defn- twin-ns [demo] (symbol (str "memo.demos." (str/lower-case demo) "-test")))

(deftest every-demo-is-accounted-for
  (is (= 36 (count demos)) "memo 1.3.2's 36 demos")
  (doseq [demo demos]
    (testing demo
      (let [fx (ts/fixture (str "demos/" demo ".edn"))]
        (if-let [why (not-ported demo)]
          (do (is (string? why))
              (is (= "stopped" (:outcome fx)) "it cannot run here in Python either")
              (is (empty? (:models fx))))
          (do (is (= "ran" (:outcome fx)) (str "the demo ran to its end in Python: " (:error fx)))
              (is (some? (try (require (twin-ns demo)) (find-ns (twin-ns demo)) (catch Exception _ nil)))
                  (str (twin-ns demo) " has its twins"))))))))
