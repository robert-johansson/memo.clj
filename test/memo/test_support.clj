(ns memo.test-support
  "Helpers for memo.clj's tests: reference fixtures from Python memo, numeric
  comparison at the oracle's precision, and compile-error messages."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [is testing]]
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

(defn- model [ns k] @(ns-resolve ns (symbol (name k))))

(defn check-cases
  "Each fixture case: the model of that name in ns, called with its args,
  has Python memo's shape and values. A case that throws fails on its own."
  [ns fixture]
  (doseq [{:keys [model args shape value]} (:cases fixture)]
    (testing (str (name model) " " args)
      (let [result (try (apply (memo.test-support/model ns model) args) (catch Exception e e))]
        (if (instance? Exception result)
          (is false (str "threw: " (ex-message result)))
          (do (is (= shape (array/shape result)))
              (is (close? value (host result))
                  (str "python " (pr-str value) "\nmemo.clj " (pr-str (host result))))))))))

(defn gradient
  "d/d(first argument) of sum(weights * f(args)), by gen.array's grad."
  [f weights args]
  (let [w (array/array weights)]
    (host ((array/grad (fn [a0] (array/sum (array/* w (apply f a0 (rest args)))))) (first args)))))

(defn grad-close?
  "Gradients agree: jax.grad computes in float32. A NaN matches a NaN (Python
  memo's NaN gradients are matched, not repaired; DESIGN.md §2.4)."
  [expected actual]
  (if (Double/isNaN (double expected))
    (Double/isNaN (double actual))
    (< (Math/abs (- (double expected) (double actual))) (+ 1e-5 (* 1e-4 (Math/abs (double expected)))))))

(defn check-grads
  "Each fixture gradient: memo.clj's matches jax.grad's."
  [ns fixture]
  (doseq [{:keys [model args weights value]} (:grads fixture)]
    (testing (str (name model) " " args)
      (let [g (gradient (memo.test-support/model ns model) weights args)]
        (is (grad-close? value g) (str "jax " value " memo.clj " g))))))

(defn check-ledger
  "Each upstream-ledger entry (DESIGN.md §9): memo.clj's gradient is the
  correct one, where Python memo's is not."
  [ns fixture]
  (doseq [{:keys [model args weights python correct why]} (:ledger fixture)]
    (testing (str (name model) " " args ": " why)
      (let [g (gradient (memo.test-support/model ns model) weights args)]
        (is (grad-close? correct g) (str "correct " correct " memo.clj " g))
        (is (not (grad-close? python correct)) "the ledger records a real difference")))))

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
