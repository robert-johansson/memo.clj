(ns memo.acceptance-test
  "The first acceptance pass's findings (bean memoclj-ai05), each with a test
  that fails without its fix. Every model here named t<k>-... has a twin of
  the same name in test/fixtures/generate.py, for finding T<k>, and
  acceptance.edn holds Python memo's results, errors and gradients. Where
  Python memo is wrong, memo.clj's own checks say what it does instead."
  (:require [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :as domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(def X123 [1 2 3])
(def Signed [-2 -1 0 1 2])

(defn- thrown
  "What (thunk) throws, or nil."
  [thunk]
  (try (thunk) nil (catch Exception e e)))

;; T1: ** on literals and parameters is Python's

(defmemo t1-lit-neg [] [] (return (** 2 -2)))
(defmemo t1-lit-pos [] [] (return (** 2 3)))
(defmemo t1-param-neg [] [a] (return (** a -1)))
(defmemo t1-param-exp [] [a] (return (** 2 a)))
(defmemo t1-float-exp [[x X123]] [] (return (** x -1.0)))
(defmemo t1-int-pos [[x X123]] [] (return (** x 2)))
(defmemo t1-zero-neg [] [] (return (** 0 -1)))

;; T2: the gradient of abs at 0 is JAX's, 1

(defmemo t2-abs [] [mu] (return (abs mu)))

(defmemo t2-near [] [mu]
  (alice :chooses [x X] :wpp (exp (- (abs (- x mu)))))
  (return (E alice.x)))

(defmemo t2-abs-int [[s Signed]] [] (return (abs s)))

;; T3: a member with an earlier member's value is its alias (IntEnum)

(defenum Dupe [P 1] [Q 1] [R 2])

(defmemo t3-dupe-mean [] []
  (alice :chooses [x Dupe] :uniformly)
  (return (E alice.x)))

(defmemo t3-dupe-to-be [[x Dupe]] []
  (alice :chooses [y Dupe] :to-be Dupe.Q)
  (return (Pr (= alice.y x))))

;; T4: a host function's Ratio or big integer is a Python number

(defn t4-half [v] (/ v 2))
(defn t4-triple [v] (* v 3N))

(defmemo t4-host-ratio [[x X]] [] (return (t4-half x)))
(defmemo t4-static-ratio [] [] (return (t4-half 3)))
(defmemo t4-inline-ratio [] [] (return #{(/ 3 2)}))
(defmemo t4-host-bigint [[x X]] [] (return (t4-triple x)))

;; against Python memo

(def fixture (ts/fixture "acceptance.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.acceptance-test fixture))

(deftest gradients-match-jax
  (ts/check-grads 'memo.acceptance-test fixture))

(deftest runtime-errors-match-python-memo
  (doseq [[k message] (:runtime-errors fixture)]
    (testing (name k)
      (let [e (thrown @(ns-resolve 'memo.acceptance-test (symbol (name k))))]
        (is (err/memo-error? e) "compiles, and fails when called")
        (is (= message (some-> e ex-message)))))))

;; where Python memo is wrong

(defmemo t1-int-neg [[x X123]] [] (return (** x -1)))
(defmemo t1-int-param [[x X123]] [a] (return (** x a)))
(defmemo t1-int-exp-neg [[s Signed]] [] (return (** 2 s)))

(defmemo t1-expect-neg [] []
  (alice :chooses [y X123] :wpp 1)
  (return (E (** alice.y -1))))

(defmemo t1-complex [] [] (return (** -8 0.5)))

(deftest integer-arrays-to-negative-powers-are-loud
  ;; Python memo returns garbage under jit (x ** -1 over [1 2 3] is
  ;; [1 0 2111105451]), and eager JAX rejects them
  (doseq [[what thunk] [["x ** -1" #(t1-int-neg)]
                        ["x ** a, a = -1" #(t1-int-param -1)]
                        ["2 ** s over negative s" #(t1-int-exp-neg)]
                        ["E[y ** -1]" #(t1-expect-neg)]]]
    (testing what
      (let [e (thrown thunk)]
        (is (err/memo-error? e))
        (is (= "Integers cannot be raised to negative powers" (some-> e ex-message))))))
  (testing "to a non-negative power, an integer array stays an integer array"
    (is (ts/close? [1 4 9] (ts/host (t1-int-param 2))))))

(deftest a-negative-number-to-a-fractional-power-is-loud
  ;; Python computes it as a complex number
  (let [e (thrown #(t1-complex))]
    (is (err/memo-error? e))
    (is (re-find #"is complex" (str (some-> e ex-message))))))

(deftest enum-aliases
  (is (= '[P R] (:members Dupe)))
  (is (= [1 2] (domains/values Dupe)))
  (is (= ["P" "R"] (domains/labels Dupe)))
  (is (= 1 (Dupe :Q) (Dupe :P))))

(defn t4-big [v] (* v 10000000000000000000N))

(defmemo t4-host-overflow [[x X]] [] (return (t4-big x)))
(defmemo t4-static-overflow [] [] (return (t4-big 3)))
(defmemo t4-inline-overflow [] [] (return #{(* 3 10000000000000000000N)}))

(deftest host-results-are-pythons-numbers
  (testing "a Ratio is a double (a Ratio is close to its double, so the fixture cannot tell)"
    (is (= 1.5 (ts/host (t4-static-ratio))))
    (is (= 1.5 (ts/host (t4-inline-ratio)))))
  (doseq [[what thunk] [["tabulated" #(t4-host-overflow)]
                        ["static" #(t4-static-overflow)]
                        ["inline" #(t4-inline-overflow)]]]
    (testing (str "beyond 64 bits, loud: " what)
      (let [e (thrown thunk)]
        (is (err/memo-error? e))
        (is (re-find #"an integer beyond 64 bits" (str (some-> e ex-message))))))))
