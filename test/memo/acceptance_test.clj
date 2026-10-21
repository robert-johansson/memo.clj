(ns memo.acceptance-test
  "The first acceptance pass's findings (bean memoclj-ai05), each with a test
  that fails without its fix. Every model here named t<k>-... has a twin of
  the same name in test/fixtures/generate.py, for finding T<k>, and
  acceptance.edn holds Python memo's results, errors and gradients. Where
  Python memo is wrong, memo.clj's own checks say what it does instead."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
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

;; T5: and, or, xor, not and Pr with several conditions on 0/1 integers;
;; the result is an integer unless every operand is a boolean

(defmemo t5-pr-two [] []
  (alice :chooses [p Bool] :uniformly)
  (alice :chooses [q Bool] :uniformly)
  (return (Pr alice.p alice.q)))

(defmemo t5-and-table [[p Bool] [q Bool]] [] (return (and p q)))
(defmemo t5-or-table [[p Bool] [q Bool]] [] (return (or p q)))
(defmemo t5-xor-table [[p Bool] [q Bool]] [] (return (xor p q)))
(defmemo t5-not-table [[p Bool]] [] (return (not p)))
(defmemo t5-and-sum [[p Bool] [q Bool]] [] (return (+ (and p q) (and p q))))
(defmemo t5-and-literal-sum [[p Bool]] [] (return (+ (and (= p 1) 1) (and (= p 1) 1))))
(defmemo t5-xor-mixed [[p Bool] [q Bool]] [] (return (xor p (= q 1))))

;; T6: the cast binds top-level statements and the agents expressions name

(defmemo t6-cast-inner [] []
  (cast [alice bob])
  (alice :thinks (carol :chooses [x X] :wpp 1))
  (return 0))

(defmemo t6-cast-imagine [] []
  (cast [alice])
  (alice :chooses [x X] :wpp 1)
  (return (as alice (imagine (carol :chooses [y X] :wpp 1) 1.0))))

;; T7: knows with no names is no statement

(defmemo t7-knows-none [] []
  (alice :chooses [x X] :wpp 1)
  (bob :knows)
  (return (E alice.x)))

;; T8: after alice observes carol.x, a mind in alice's whose beliefs vary
;; with carol.x cannot be read; the rest can

(defmemo t8-retag-alice-only [[y X]] []
  (alice :thinks
    (carol :chooses [x X] :wpp 1)
    (bob :thinks (carol :chooses [x X] :wpp 1))
    (bob :observes carol.x :is carol.x))
  (alice :observes carol.x :is y)
  (return (as alice (E carol.x))))

(defmemo t8-retag-no-read [[y X]] []
  (alice :thinks
    (carol :chooses [x X] :wpp 1)
    (bob :thinks (carol :chooses [x X] :wpp 1))
    (bob :observes carol.x :is carol.x))
  (alice :observes carol.x :is y)
  (return 1.0))

(defmemo t8-retag-knows [[y X]] []
  (alice :thinks (carol :chooses [x X] :wpp 1) (bob :knows carol.x))
  (alice :observes carol.x :is y)
  (return (as alice (E (as bob carol.x)))))

;; T9: a forgotten choice's copies in other minds cannot be read; unread,
;; they are harmless

(defmemo t9-forget-known-unread [] []
  (alice :chooses [x X] :wpp (+ 1 x))
  (bob :knows alice.x)
  (alice :forgets-about x)
  (return 1.0))

(defmemo t9-forget-observed-unread [] []
  (alice :chooses [x X] :wpp (+ 1 x))
  (bob :thinks (alice :chooses [x X] :wpp 1))
  (bob :observes alice.x :is alice.x)
  (alice :forgets-about x)
  (return 1.0))

(defmemo t9-forget-alone [] []
  (alice :chooses [x X] :wpp (+ 1 x))
  (alice :chooses [y X] :to-be x)
  (alice :forgets-about x)
  (return (E alice.y)))

;; T10: a vectorized function gets an array parameter as its array

(def Two [0 1])

(defn ^:memo/vectorized t10-vtake [table i] (array/take table (array/astype i :int32)))

(defmemo t10-lookup [[k Two]] [^:array table]
  (alice :chooses [x X] :wpp (exp (t10-vtake table x)))
  (return (E alice.x)))

;; against Python memo

(def fixture (ts/fixture "acceptance.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.acceptance-test fixture))

(deftest gradients-match-jax
  (ts/check-grads 'memo.acceptance-test fixture))

(def error-forms
  {:t6-cast-expr-err
   '(defmemo t6-cast-expr-err [] []
      (cast [alice])
      (alice :thinks (carol :chooses [y X] :wpp 1))
      (return (as alice (E carol.y))))
   :t21-wants-no-goal
   '(defmemo t21-wants-no-goal [] []
      (alice :wants g)
      (return 1))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.acceptance-test (error-forms k)) ex-message))))))

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

(defmemo t5-and-two [[x X] [q Bool]] [] (return (and x q)))

(defmemo t5-pr-two-of-three [] []
  (alice :chooses [x X] :uniformly)
  (alice :chooses [q Bool] :uniformly)
  (return (Pr alice.x alice.q)))

(deftest logical-operators-on-other-integers-are-loud
  ;; Python memo computes them bitwise there (2 & 1 is 0)
  (let [e (thrown #(t5-and-two))]
    (is (err/memo-error? e))
    (is (= "memo.clj computes and on booleans and on 0 and 1 only" (some-> e ex-message))))
  (testing "the message names the construct written"
    (is (= "memo.clj computes Pr with several conditions on booleans and on 0 and 1 only"
           (some-> (thrown #(t5-pr-two-of-three)) ex-message)))))

(defn- compile-message
  "The message of the memo error form gives when it is defined, or nil."
  [form]
  (some-> (ts/compile-error 'memo.acceptance-test form) ex-message))

(deftest reading-a-mind-whose-observed-value-moved-is-a-compile-error
  ;; Python memo: a shape error when the model is called
  (is (= "Observation against an already-used value"
         (compile-message '(defmemo t8-retag [[y X]] []
                             (alice :thinks
                               (carol :chooses [x X] :wpp 1)
                               (bob :thinks (carol :chooses [x X] :wpp 1))
                               (bob :observes carol.x :is carol.x))
                             (alice :observes carol.x :is y)
                             (return (as alice (E (as bob carol.x))))))))
  (testing "bob's beliefs, read without the stale copy"
    (is (= "Observation against an already-used value"
           (compile-message '(defmemo t8-retag-beliefs [[y X]] []
                               (alice :thinks
                                 (carol :chooses [x X] :wpp 1)
                                 (bob :thinks (carol :chooses [x X] :wpp 1)
                                              (carol :chooses [z X] :wpp (= z x)))
                                 (bob :observes carol.x :is carol.x))
                               (alice :observes carol.x :is y)
                               (return (as alice (E (as bob (E carol.z)))))))))))

(deftest reading-a-forgotten-choice-is-a-compile-error
  ;; Python memo: a KeyError, ('alice', 'x'), when the model is defined
  (doseq [[what form]
          [["known" '(defmemo t9-known [] []
                       (alice :chooses [x X] :wpp (+ 1 x))
                       (bob :knows alice.x)
                       (alice :forgets-about x)
                       (return (E (as bob alice.x))))]
           ["observed" '(defmemo t9-observed [] []
                          (alice :chooses [x X] :wpp (+ 1 x))
                          (bob :thinks (alice :chooses [x X] :wpp 1))
                          (bob :observes alice.x :is alice.x)
                          (alice :forgets-about x)
                          (return (E (as bob (E alice.x)))))]
           ["beliefs only" '(defmemo t9-beliefs [] []
                              (alice :chooses [x X] :wpp (+ 1 x))
                              (bob :thinks (alice :chooses [x X] :wpp 1)
                                           (alice :chooses [y X] :wpp (= y x)))
                              (bob :observes alice.x :is alice.x)
                              (alice :forgets-about x)
                              (return (E (as bob (E alice.y)))))]
           ["in a mind" '(defmemo t9-mind [] []
                           (carol :thinks
                             (alice :chooses [x X] :wpp (+ 1 x))
                             (bob :knows alice.x)
                             (alice :forgets-about x))
                           (return (E (as carol (E (as bob alice.x))))))]]]
    (testing what
      (is (= "alice.x was forgotten" (compile-message form))))))

(defmemo t11-inline-scaled [] [a] (return #{(* 2 a)}))
(defmemo t11-inline-array [] [a] (return #{(array/* 2 a)}))

(deftest an-inline-form-under-differentiation
  (is (ts/close? 1.0 (ts/host (t11-inline-scaled 0.5))))
  (testing "with clojure.core's operators: a memo error naming the form"
    (let [e (thrown #(ts/gradient t11-inline-scaled 1.0 [0.5]))]
      (is (err/memo-error? e))
      (is (str/includes? (str (some-> e ex-message)) "#{(* 2 a)}"))))
  (testing "with gen.array's: differentiated"
    (is (ts/grad-close? 2.0 (ts/gradient t11-inline-array 1.0 [0.5])))))

(defn ^:memo/vectorized t12-exp [x] (Math/exp x))

(defmemo t12-vectorized-throws [] []
  (alice :chooses [x X] :wpp (t12-exp x))
  (return (E alice.x)))

(deftest a-vectorized-function-that-throws
  (let [e (thrown #(t12-vectorized-throws))]
    (is (err/memo-error? e))
    (is (str/starts-with? (str (some-> e ex-message))
                          "The vectorized function t12-exp failed on whole arrays"))))

(def L [1 2])

(defmemo t13-inline-vector [] [] (return #{L}))

(deftest an-inline-form-that-is-not-a-number
  (let [e (thrown #(t13-inline-vector))]
    (is (err/memo-error? e))
    (is (= "The inline form #{L} gave [1 2], not a number" (some-> e ex-message)))))

(def NotDom 5)
(def SetDom #{0 1})

(deftest what-is-not-a-domain
  (testing "a number (Python memo's {5}; decision D5)"
    (is (= "NotDom is not a domain: 5"
           (compile-message '(defmemo t14-number [] []
                               (alice :chooses [x NotDom] :wpp 1)
                               (return (E alice.x)))))))
  (testing "a set"
    (is (= "SetDom is not a domain: #{0 1}"
           (compile-message '(defmemo t14-set [] []
                               (alice :chooses [x SetDom] :wpp 1)
                               (return (E alice.x)))))))
  (testing "nothing at all"
    (is (= "Unknown domain Nope"
           (compile-message '(defmemo t14-unknown [[x Nope]] [] (return x))))))
  (testing "a parameter named like a domain (decision D3)"
    (is (= "The name `X` is already used for a static parameter and cannot be used as a domain."
           (compile-message '(defmemo t14-shadow [[x X]] [X] (return (* x X))))))))

(deftest print-table-checks-its-arguments
  (testing "a memo's var is the memo"
    (is (str/includes? (with-out-str (memo/print-table #'t2-abs-int (t2-abs-int))) "t2-abs-int")))
  (testing "a result of another shape"
    (let [e (thrown #(memo/print-table t1-float-exp (t2-abs-int)))]
      (is (err/memo-error? e))
      (is (str/starts-with? (str (some-> e ex-message)) "print-table: the result has shape [5]")))))

(defmemo t16-forever [[x X]] [t] (return (t16-forever [x] (+ t 1))))

(deftest runaway-recursion
  ;; Python memo: a RecursionError
  (let [e (thrown #(t16-forever 0))]
    (is (err/memo-error? e))
    (is (= "Maximum recursion depth exceeded in memo t16-forever" (some-> e ex-message)))))

(deftest format-error-renders-the-memo-error-inside
  (let [e (binding [*ns* (the-ns 'memo.acceptance-test)]
            (try (eval '(defmemo t17-bad [] [] (return alice.nope))) nil (catch Exception e e)))]
    (is (not (err/memo-error? e)) "Clojure wraps the error of a defmemo that fails to compile")
    (is (= (err/format-error (err/root-memo-error e)) (err/format-error e)))))

;; error messages (T18-T24)

(defmemo t19-ok1 [[x X]] [a] (return (* x a)))
(defmemo t19-ok2 [[x X] [y X]] [] (return (+ x y)))

(deftest error-messages
  (testing "T18 a malformed KL names KL (Python memo reads it as a query of an agent named KL)"
    (doseq [form ['(defmemo t18-expr [] []
                     (a :chooses [x X] :wpp 1) (b :chooses [x X] :wpp 1)
                     (return (KL (+ a.x 1) b.x)))
                  '(defmemo t18-lits [] [] (return (KL 3 4)))]]
      (is (= "Unexpected variable in (KL ...)" (compile-message form)))))
  (testing "T19 axes for a memo defined before: Python memo's messages, when it is defined"
    (is (= "Too many arguments to memo call!"
           (compile-message '(defmemo t19-many [[x X]] [] (return (t19-ok1 [x x] 1.0))))))
    (is (= "Not enough arguments to memo call!"
           (compile-message '(defmemo t19-few [[x X]] [] (return (t19-ok2 [x])))))))
  (testing "T20 a parameter that is not a number says what it is"
    (is (= "Parameter a was not a numeric scalar, but \"hello\", a String."
           (some-> (thrown #(t1-param-neg "hello")) ex-message)))
    (is (= "Parameter a was not a numeric scalar, but :a, a Keyword."
           (some-> (thrown #(t1-param-neg :a)) ex-message)))
    (is (str/starts-with? (str (some-> (thrown #(t1-param-neg nil)) ex-message))
                          "Parameter a was not a numeric scalar, but rather an array.")
        "nil: Python memo's message for None"))
  (testing "T21 an imagine ending in a statement"
    (is (= "Invalid statement at end of imagine expression"
           (compile-message '(defmemo t21-imagine [] []
                               (return (imagine (alice :chooses [x X] :wpp 1))))))))
  (testing "T22 a frame made for EU is named as Python memo names it, without $"
    (is (re-matches #"Unknown choice future_alice_\d+\.zz"
                    (str (compile-message '(defmemo t22-goal [] []
                                             (alice :wants g zz)
                                             (alice :chooses [x X] :to-maximize (EU g))
                                             (return 1))))))))

(deftest warnings-say-where
  ;; T23: Python memo prints the source line
  (let [w (java.io.StringWriter.)]
    (binding [*err* w
              err/*warnings?* true
              *ns* (the-ns 'memo.acceptance-test)]
      (eval '(defmemo t23-redundant [] []
               (alice :chooses [x X] :wpp 1)
               (bob :knows alice.x)
               (return (E (as bob (E alice.x)))))))
    (is (str/includes? (str w) "Redundant expectation"))
    (is (str/includes? (str w) "in: defmemo t23-redundant"))))

(def inspect-forms
  {:t24-insp1 '(defmemo t24-insp1 [[q X]] []
                 (alice :knows q)
                 (alice :thinks
                   (bob :chooses [x X] :wpp 1)
                   (bob :thinks (carol :chooses [y X] :wpp 1))
                   (bob :observes carol.y :is bob.x)
                   (bob :inspect))
                 (alice :observes bob.x :is q)
                 (alice :inspect)
                 (carol :inspect)
                 (return 1))
   :t24-insp2 '(defmemo t24-insp2 [] []
                 (child :has-theory :child)
                 (child :thinks (sally :chooses [x X] :wpp 1))
                 (child :inspect)
                 (return 1))})

(deftest inspect-prints-as-python-memo
  ;; T24: Python memo's order (as choices were made) and words; the
  ;; "** Tracing" lines name the source, a file there, the defmemo here
  (doseq [[k text] (:inspect fixture)]
    (testing (name k)
      (let [ours (with-out-str (binding [*ns* (the-ns 'memo.acceptance-test)] (eval (inspect-forms k))))
            lines #(remove (fn [l] (str/starts-with? l "** Tracing")) (str/split-lines %))]
        (is (= (lines text) (lines ours)))))))

(deftest a-namespaces-own-bool-takes-its-place
  ;; decision D2: Python memo's generated code imports memo's Bool, which
  ;; silently wins over a model's own (there a uniform choice gives 1/2)
  (let [ns (create-ns 'memo.acceptance-own-bool)]
    (try
      (binding [*ns* ns]
        (refer-clojure)
        (eval '(require '[memo.core :refer [defmemo]] '[memo.domains :refer [defenum]]))
        (eval '(defenum Bool NO MAYBE YES))
        (eval '(defmemo three-way [] []
                 (alice :chooses [b Bool] :uniformly)
                 (return (Pr (= alice.b 1)))))
        (is (ts/close? (/ 1.0 3) (ts/host ((resolve 'three-way))))))
      (finally (remove-ns 'memo.acceptance-own-bool)))))
