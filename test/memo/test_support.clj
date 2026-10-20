(ns memo.test-support
  "Helpers for memo.clj's tests: reference fixtures from Python memo, numeric
  comparison at the oracle's precision, and compile-error messages."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [gen.array :as array]
            [memo.errors :as err]))

(defn fixture
  "A fixture file from test/fixtures, written by test/fixtures/generate.py."
  [name]
  (edn/read-string (slurp (io/file "test/fixtures" name))))

(defn tol
  "The tolerance of a comparison with the oracle, which computes in float32.
  The tested backend may be float32 too (MLX)."
  [expected]
  (+ 1e-6 (* 2e-6 (Math/abs (double expected)))))

(defn close?
  "Whether two values (numbers or nested vectors of them) agree within the
  float32 tolerance, NaN matching NaN."
  [expected actual]
  (cond
    (and (sequential? expected) (sequential? actual))
    (and (= (count expected) (count actual))
         (every? true? (map close? expected actual)))

    (and (number? expected) (number? actual))
    (let [e (double expected) a (double actual)]
      (or (and (Double/isNaN e) (Double/isNaN a))
          (= e a)
          (<= (Math/abs (- e a)) (tol e))))

    :else false))

(defn host
  "A result read on the host: a number, or nested vectors, with booleans as
  1.0 and 0.0 (as the fixtures record Python memo's bool results)."
  [x]
  (let [num (fn num [v]
              (cond (boolean? v) (if v 1.0 0.0)
                    (sequential? v) (mapv num v)
                    :else v))]
    (num (array/->clj x))))

(defn compile-error
  "The memo error from evaluating form (a defmemo) in namespace ns, or nil
  when it compiles."
  [ns form]
  (binding [*ns* (the-ns ns)
            err/*warnings?* false]
    (try
      (eval form)
      nil
      (catch Throwable e
        (or (err/root-memo-error e)
            (throw e))))))
