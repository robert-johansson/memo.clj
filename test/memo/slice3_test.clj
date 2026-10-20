(ns memo.slice3-test
  "M1, slice 3: memo calls, path conditions, recursion (RSA from memo's demo).
  Every model here has a twin of the same name in test/fixtures/generate.py,
  and slice3.edn holds Python memo's results, errors and gradients."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(defenum Coin HEADS TAILS)
(defenum U [GREEN 1] [PINK 2] [SQUARE 4] [ROUND 8])
(defenum R [GREEN_SQUARE 5] [GREEN_CIRCLE 9] [PINK_CIRCLE 10])
(def Unit [0])

;; demo-rsa.py, with its host function denotes(u, r) = (u & r) != 0 written in
;; memo's arithmetic (every u is a power of two)
(defmemo L [[u U] [r R]] [beta t]
  (listener :thinks
    (speaker :given [r R] :wpp 1)
    (speaker :chooses [u U] :wpp (* (= (mod (/ (- r (mod r u)) u) 2) 1)
                                    (if (= t 0) 1 (exp (* beta (L [u r] beta (- t 1))))))))
  (listener :observes speaker.u :is u)
  (listener :chooses [r R] :wpp (Pr (= speaker.r r)))
  (return (Pr (= listener.r r))))

(defmemo L0 [[u U] [r R]] []
  (listener :knows u)
  (listener :chooses [r R] :wpp (= (mod (/ (- r (mod r u)) u) 2) 1))
  (return (Pr (= listener.r r))))

(defmemo S1 [[r R] [u U]] [alpha]
  (speaker :knows r)
  (speaker :chooses [u U] :wpp (exp (* alpha (log (L0 [u r])))))
  (return (Pr (= speaker.u u))))

;; S1 with its weights as L0 ** alpha: the same values, and a finite gradient
;; where (exp (* alpha (log L0))) has NaN, in Python memo and here alike
(defmemo S1-pow [[r R] [u U]] [alpha]
  (speaker :knows r)
  (speaker :chooses [u U] :wpp (** (L0 [u r]) alpha))
  (return (Pr (= speaker.u u))))

(defmemo L1 [[u U] [r R]] [alpha]
  (listener :thinks
    (speaker :given [r R] :wpp 1)
    (speaker :chooses [u U] :wpp (S1 [r u] alpha)))
  (listener :observes speaker.u :is u)
  (listener :chooses [r R] :wpp (Pr (= speaker.r r)))
  (return (Pr (= listener.r r))))

(defmemo test- [[x X]] [t]
  (return x))

(defmemo memo-call-ellipsis [] [(t 0)]
  (alice :chooses [x X] :wpp (test- [x] ...))
  (return (E alice.x)))

(defmemo recursive-kwarg [[x X]] [a (b 0)]
  (return (if (> a 0) (recursive-kwarg [x] :a (- a 1) :b (+ b 1)) b)))

(defmemo kwarg [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (recursive-kwarg [alice.x] :a 3))))

(defmemo kwarg-mixed [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (recursive-kwarg [alice.x] 3 :b 10))))

(defmemo multi-return-1 [[x X]] []
  (return 1)
  (return 2))

(defmemo multi-return-2 [[x X]] []
  (return (multi-return-1 1 [x])))

(defmemo observes-call [[x X]] []
  (a :thinks (b :chooses [x X] :wpp 1))
  (a :observes b.x :is x)
  (return (as a (as b (test- [x] 0)))))

(defmemo table2 [[x X] [c Coin]] []
  (return (+ (* x 10) c)))

(defmemo call-two-agents [] []
  (alice :chooses [x X] :wpp 1)
  (bob :chooses [c Coin] :wpp (+ c 1))
  (return (E (table2 [alice.x bob.c]))))

(defmemo call-reversed [[c Coin]] []
  (alice :knows c)
  (alice :chooses [x X] :wpp (+ (table2 [x c]) 1))
  (return (E alice.x)))

(defmemo fib [[a Unit]] [n]
  (return (if (< n 2) 1 (+ (fib [a] (- n 1)) (fib [a] (- n 2))))))

(defmemo multi-return-3 [[x X]] []
  (return (multi-return-1 [x])))

(defmemo multi-return-4 [[x X]] []
  (return (multi-return-1 3 [x])))

(defmemo multi-return-5 [[x X]] []
  (return (test- 0 [x] 0)))

(defmemo call-domain-mismatch [[c Coin]] []
  (return (test- [c] 0)))

(def fixture (ts/fixture "slice3.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.slice3-test fixture))

(def error-forms
  {:call-not-static
   '(defmemo call-not-static [] []
      (alice :chooses [x X] :wpp (test- [x] x))
      (return 1))
   :call-kwarg-not-static
   '(defmemo call-kwarg-not-static [] []
      (alice :chooses [x X] :wpp (test- [x] :t x))
      (return 1))
   :call-unknown-choice
   '(defmemo call-unknown-choice [] []
      (return (test- [z] 0)))
   :call-bad-axis
   '(defmemo call-bad-axis [[x X]] []
      (return (test- [(+ x 1)] 0)))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.slice3-test (error-forms k)) ex-message))))))

(def runtime-messages
  "Where memo.clj's runtime message differs from Python memo's on purpose: it
  shows memo.clj's syntax, and Python's has a doubled word."
  {:multi-return-3 (str "You are calling a memo model that has multiple return values, but "
                        "you have not specified which return value you want. Try writing "
                        "(model 0 [...] ...) to get the first return value.")
   :multi-return-5 (str "You are calling a memo model with only one return value, so you do "
                        "not have to specify which return value you want.")})

(deftest runtime-errors-match-python-memo
  (doseq [[k message] (:runtime-errors fixture)]
    (testing (name k)
      (let [e (try (@(ns-resolve 'memo.slice3-test (symbol (name k)))) nil (catch clojure.lang.ExceptionInfo e e))]
        (is (err/memo-error? e) "compiles, and fails when called")
        (is (= (get runtime-messages k message) (some-> e ex-message)))))))

(deftest gradients-match-jax
  (ts/check-grads 'memo.slice3-test fixture))

(deftest calls
  (testing "keyword arguments, and a call that gives one both ways"
    (is (ts/close? [2.0 2.0 2.0] (ts/host (recursive-kwarg 2))))
    (is (ts/close? [7.0 7.0 7.0] (ts/host (recursive-kwarg {:a 2 :b 5}))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"passed both positionally and by name"
                          (recursive-kwarg 2 {:a 3}))))
  (testing "a recursion's depth is a path condition, read on the host"
    (is (= 89.0 (double (first (ts/host (fib 10))))))))
