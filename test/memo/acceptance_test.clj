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
            [memo.domains :as domains :refer [defdomain defenum]]
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

;; the second acceptance check (memoclj-a46m)

(defmemo s2-bool-sum [] [a] (return (+ (> a 0) (> a 0))))
(defmemo s2-bool-neg [] [a] (return (- (> a 0))))

(defmemo s2-bool-expect [] [a]
  (alice :chooses [x X] :wpp 1)
  (return (E (* alice.x (+ (> a 0) (> a 0))))))

(defmemo s3-pow0 [] [a]
  (alice :chooses [x X] :wpp 1)
  (return (E (** (* a alice.x) 0))))

(defenum Mixed [RED 1] GREEN BLUE)

(defmemo s4-mixed [[c Mixed]] []
  (alice :chooses [x Mixed] :uniformly)
  (return (Pr (= alice.x c))))

(defmemo e8-xor3 [[p Bool] [q Bool] [r Bool]] [] (return (xor p q r)))
(defmemo d2-div0 [] [a] (return (/ a 0)))
(defmemo d2-mod0 [] [a] (return (% a 0)))

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
      (let [args (get {:d2-div0 [1.0] :d2-mod0 [1.0]} k [])  ; generate.py's calls
            e (thrown #(apply @(ns-resolve 'memo.acceptance-test (symbol (name k))) args))]
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

;; the rerun of the pass's models (T32): raw errors there before the fixes

(declare Later)
(defdomain Cell2 :x 2 :y 3)

(deftest what-the-rerun-found
  (testing "a defdomain field of size 1 (Python memo: an assertion)"
    (is (= "defdomain Bad: each field is a keyword and a size above 1, not :x 1"
           (some-> (thrown #(binding [*ns* (the-ns 'memo.acceptance-test)]
                              (eval '(defdomain Bad :x 1))))
                   err/root-memo-error ex-message))))
  (testing "packing a member with a field missing"
    (let [e (thrown #(Cell2 {:x 1}))]
      (is (err/memo-error? e))
      (is (= "Domain Cell2 packs exactly its fields [:x :y]" (some-> e ex-message)))))
  (testing "a domain declared but without a value"
    (is (= "Unknown domain Later"
           (compile-message '(defmemo t32-later [] [] (alice :chooses [x Later] :wpp 1) (return 1))))))
  (testing "a default that uses another parameter (Python memo: a NameError)"
    (is (= "The default of parameter b refers to the parameter a"
           (compile-message '(defmemo t32-default [] [a (b a)] (return b))))))
  (testing "an empty Pr has a hint"
    (is (some? (some-> (ts/compile-error 'memo.acceptance-test '(defmemo t32-pr [] [] (return (Pr))))
                       ex-data :memo/hint)))))

(defn- in-namespace
  "body's forms evaluated, in order, in a fresh namespace with memo.clj's
  forms referred; the last's value. The namespace is removed after."
  [forms]
  (let [nm (gensym "memo.acceptance-scratch")
        ns (create-ns nm)]
    (try
      (binding [*ns* ns]
        (refer-clojure)
        (eval '(require '[memo.core :refer [defmemo]] '[memo.domains :refer [defenum]]
                        '[gen.array :as array]))
        (reduce (fn [_ f] (eval f)) nil forms))
      (finally (remove-ns nm)))))

(deftest the-second-check
  (testing "S1 a memo call checks domains by value, not by name"
    (let [e (thrown #(in-namespace
                      '[(def X [2 1 0])
                        (defmemo caller [[x X]] [] (return (memo.acceptance-test/t19-ok1 [x] 10.0)))
                        (caller)]))]
      (is (err/memo-error? e))
      (is (= "Domain mismatch in memo call argument 1: X != X." (some-> e ex-message)))
      (is (some? (:memo/hint (ex-data e))) "E9: with a hint"))
    (is (ts/close? [0.0 10.0 20.0]
                   (ts/host (in-namespace
                             '[(def Y [0 1 2])
                               (defmemo caller [[y Y]] [] (return (memo.acceptance-test/t19-ok1 [y] 10.0)))
                               (caller)])))
        "the same values under another name"))
  (testing "S5 domain values beyond 32 bits, on a backend with 32-bit integers"
    (let [r (try (ts/host (in-namespace '[(def Big [0 10000000000])
                                         (defmemo big [] []
                                           (alice :chooses [x Big] :wpp 1)
                                           (return (E (/ alice.x 10000000000))))
                                         (big)]))
                 (catch Exception e e))]
      (if (= :int32 (gen.array/dtype (gen.array/array [1])))
        (is (= "Domain value 10000000000 is out of bounds for int32" (some-> (err/root-memo-error r) ex-message)))
        (is (ts/close? 0.5 r)))))
  (testing "S6 repeated domain values warn"
    (let [w (java.io.StringWriter.)]
      (binding [*err* w err/*warnings?* true]
        (in-namespace '[(def Dup [1 1 2])
                        (defmemo dup [[y Dup]] [] (alice :chooses [x Dup] :wpp 1) (return (Pr (= alice.x y))))]))
      (is (str/includes? (str w) "Domain Dup has repeated values"))))
  (testing "C1 parameters not in a vector"
    (is (= "Malformed parameters in defmemo c1" (compile-message '(defmemo c1 [] a (return a))))))
  (testing "C2 an axis or a choice in #{...}"
    (is (= "Cannot reference x inside the inline escape hatch #{...}"
           (compile-message '(defmemo c2 [[x X]] [] (return #{x})))))
    (is (= "Cannot reference y inside the inline escape hatch #{...}"
           (compile-message '(defmemo c2b [] [] (alice :chooses [y X] :wpp 1) (return (as alice #{y})))))))
  (testing "C3 a destructuring parameter"
    (is (= "The default of parameter a, b, is not defined"
           (compile-message '(defmemo c3 [] [[a b]] (return a))))))
  (testing "C4 a vectorized function returning nil"
    (is (= "The vectorized function c4-nil returned nil, not an array"
           (some-> (thrown #(in-namespace '[(def X [0 1 2])
                                            (defn ^:memo/vectorized c4-nil [x] nil)
                                            (defmemo c4 [] [] (alice :chooses [x X] :wpp (c4-nil x))
                                              (return (E alice.x)))
                                            (c4)]))
                   ex-message))))
  (testing "C5 print-table of nil"
    (is (str/starts-with? (str (some-> (thrown #(memo/print-table t2-abs-int nil)) ex-message))
                          "print-table: the result is nil")))
  (testing "E1 format-error names an exception that is not a memo error"
    (is (= "error (not a memo error): RuntimeException: plain"
           (err/format-error (RuntimeException. "plain")))))
  (testing "E2 an unknown variant"
    (is (= "Unknown variant" (compile-message '(defmemo ^{:variant :martian} e2 [] [] (return 1))))))
  (testing "E3 a repeated axis or parameter (Python memo: a SyntaxError)"
    (is (= "Duplicate axis x in defmemo e3" (compile-message '(defmemo e3 [[x X] [x X]] [] (return x)))))
    (is (= "Duplicate parameter a in defmemo e3b" (compile-message '(defmemo e3b [] [a a] (return a))))))
  (testing "E4 a call of a memo defined before, checked when the caller is"
    (is (= "Missing parameter a in call to memo t19-ok1"
           (compile-message '(defmemo e4 [[x X]] [] (return (t19-ok1 [x]))))))
    (is (= "Too many arguments to memo t19-ok1 (it takes 1)"
           (compile-message '(defmemo e4b [[x X]] [] (return (t19-ok1 [x] 1 2))))))
    (is (= "memo t19-ok1 has no parameter d"
           (compile-message '(defmemo e4c [[x X]] [] (return (t19-ok1 [x] 1 :d 2)))))))
  (testing "E5 a memo with axes called without them"
    (is (= "t19-ok1 is a memo with axes, called without them"
           (compile-message '(defmemo e5 [[x X]] [] (return (t19-ok1 x 1.0)))))))
  (testing "E6 a Java method"
    (is (str/includes? (str (some-> (ts/compile-error 'memo.acceptance-test
                                                      '(defmemo e6 [[x X]] [] (return (Math/sqrt x))))
                                    ex-data :memo/hint))
                       "Java method")))
  (testing "E7 no class names from a ClassCastException"
    (is (= "The inline form #{(* 2 a)} failed: a gen.array value reached a function that takes numbers"
           (some-> (thrown #(ts/gradient t11-inline-scaled 1.0 [0.5])) ex-message))))
  (testing "D3 a parameter beyond 64 bits"
    (is (= "Parameter a was 100000000000000000000, an integer beyond 64 bits"
           (some-> (thrown #(t1-param-neg 100000000000000000000N)) ex-message)))))
