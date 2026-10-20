(ns memo.demos.demo-fib-test
  "memo's demo/demo-fib.py: Fibonacci by recursion on a static parameter. The
  demo's debug_print_compiled and debug_trace options print; memo.clj's
  program prints the IR, and it has no trace."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def Unit [0])

(defmemo ^:cache fib [[a Unit]] [n]
  (return (if (< n 2) 1 (+ (fib [a] (- n 1)) (fib [a] (- n 2))))))

(deftest demo-fib
  (ts/check-demo 'memo.demos.demo-fib-test "demo-fib"))
