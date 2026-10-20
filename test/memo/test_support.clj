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

(def ^:dynamic *atol*
  "An absolute tolerance added to tol's, for results that cancel: a small
  difference of large float32 values carries their rounding."
  0.0)

(defn close?
  "Whether two values (numbers or nested vectors of them) agree within the
  float32 tolerance (plus *atol*), NaN matching NaN."
  [expected actual]
  (cond
    (and (sequential? expected) (sequential? actual))
    (and (= (count expected) (count actual))
         (every? true? (map close? expected actual)))

    (and (number? expected) (number? actual))
    (let [e (double expected) a (double actual)]
      (or (and (Double/isNaN e) (Double/isNaN a))
          (= e a)
          (<= (Math/abs (- e a)) (+ *atol* (tol e)))))

    :else false))

(defn host
  "A result read on the host: a number, or nested vectors, with booleans as
  1.0 and 0.0 (as the fixtures record Python memo's bool results)."
  [x]
  (let [num (fn num [v]
              (cond (boolean? v) (if v 1.0 0.0)
                    (sequential? v) (mapv num v)
                    :else v))]
    (num (if (or (number? x) (boolean? x) (sequential? x)) x (array/->clj x)))))

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

(defn matches-digest?
  "A large result against the digest a fixture records in place of its
  values: their sum, and samples at flat indices."
  [{:keys [sum samples]} result]
  (let [flat (vec (flatten (host result)))]
    (and (close? sum (reduce + flat))
         (every? (fn [[i v]] (close? v (nth flat i))) samples))))

(defn deep
  "(thunk) on a thread with a 1 GB stack: a model that recurses 1000 levels
  (demo-mdp's Q(1000)) nests its calls that deep, as Python memo needed a
  higher recursion limit for it. Rethrows what thunk throws. An
  OutOfMemoryError can escape the thread's own handling when the heap is
  full; the thread's uncaught-exception handler delivers it then, so a
  caller never waits forever."
  [thunk]
  (let [result (promise)
        t (Thread. nil #(deliver result (try [:ok (thunk)] (catch Throwable e [:err e])))
                   "memo-deep" (* 1024 1024 1024))]
    (.setUncaughtExceptionHandler t (reify Thread$UncaughtExceptionHandler
                                      (uncaughtException [_ _ e] (deliver result [:err e]))))
    (.start t)
    (let [[k v] @result]
      (if (= k :ok) v (throw v)))))

(defn check-call
  "f called as Python memo was (positional args, then keyword args in a
  trailing map) has its shape, and its values or its digest. tie?, given,
  is a predicate of the call's args and a cell's flat index: such cells
  (exact ties of :to-maximize, which float32 and float64 break apart
  differently) may differ, and only those."
  ([f c] (check-call f c nil))
  ([f {:keys [args kwargs shape value digest]} tie?]
   (testing (str (pr-str args) (when (seq kwargs) (str " " (pr-str kwargs))))
     (let [r (try (deep #(apply f (cond-> (vec args) (seq kwargs) (conj kwargs)))) (catch Throwable e e))]
       (if (instance? Throwable r)
         (is false (str "threw: " (ex-message r)))
         (do (is (= shape (array/shape r)))
             (cond
               digest
               (is (matches-digest? digest r) "the digest")

               tie?
               (let [differ (keep-indexed (fn [i [e a]] (when-not (close? e a) i))
                                          (map vector (flatten value) (flatten (host r))))]
                 (is (every? #(tie? args %) differ)
                     (str "cells that differ and are not ties: " (vec (remove #(tie? args %) differ)))))

               :else
               (is (close? value (host r))
                   (str "python " (pr-str value) "\nmemo.clj " (pr-str (host r)))))))))))

(declare compile-error)

(defn check-demo
  "Every model of memo's demo (test/fixtures/demos/<demo>.edn) against its
  twin of the same name in ns: each call the demo made, or, for a model the
  demo shows must not compile, the form in :compile-forms, whose memo error
  has Python memo's message (or the one in :messages). :not-ported maps a
  model's name to the reason it has no twin; :atol widens the tolerance
  (*atol*) where results cancel; :ties maps a model's name to a tie?
  predicate (check-call); :renamed maps a model's name to its twin's, where
  the demo's name cannot be memo.clj's. Returns the fixture."
  ([ns demo] (check-demo ns demo {}))
  ([ns demo {:keys [compile-forms messages not-ported atol ties renamed] :or {atol 0.0}}]
   (binding [*atol* atol]
   (let [fx (fixture (str "demos/" demo ".edn"))]
     (doseq [{k :name :keys [calls outcome message]} (:models fx)]
       (testing (name k)
         (cond
           (get not-ported k) (is (string? (get not-ported k)))
           (= "ce" outcome) (is (= (get messages k message)
                                   (some-> (compile-error ns (get compile-forms k)) ex-message)))
           :else (if-let [f (some-> (ns-resolve ns (get renamed k (symbol (name k)))) deref)]
                   (doseq [c calls] (check-call f c (get ties k)))
                   (is false (str (name k) " has no twin"))))))
     (is (every? (set (map :name (:models fx)))
                 (concat (keys not-ported) (keys compile-forms) (keys messages) (keys renamed)))
         "a form, message or reason for a model the demo does not have")
     fx))))

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
  "Each upstream-ledger entry (DESIGN.md §9): memo.clj's gradient (entries
  with :weights) or value is the correct one, where Python memo's is not (a
  string :python is the error Python memo raised)."
  [ns fixture]
  (doseq [{:keys [model args weights python correct why]} (:ledger fixture)]
    (testing (str (name model) " " args ": " why)
      (let [f (memo.test-support/model ns model)]
        (if weights
          (let [g (gradient f weights args)]
            (is (grad-close? correct g) (str "correct " correct " memo.clj " g))
            (is (not (grad-close? python correct)) "the ledger records a real difference"))
          (let [v (try (host (apply f args)) (catch Exception e e))]
            (if (instance? Exception v)
              (is false (str "threw: " (ex-message v)))
              (is (close? correct v) (str "correct " (pr-str correct) " memo.clj " (pr-str v))))
            (is (or (string? python) (not (close? python correct)))
                "the ledger records a real difference")))))))

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
