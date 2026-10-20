(ns memo.slice4-test
  "M1, slice 4: host functions and the inline escape hatch. Every model here
  has a twin of the same name in test/fixtures/generate.py, and slice4.edn
  holds Python memo's results, gradients and its upstream-ledger entries."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(def N5 5)
(def N11 (vec (range 11)))
(def B2 [0 1])
(defenum U [GREEN 1] [PINK 2] [SQUARE 4] [ROUND 8])
(defenum R [GREEN_SQUARE 5] [GREEN_CIRCLE 9] [PINK_CIRCLE 10])
(defenum Loc BOX BASKET)
(defenum Action ACT_STAY ACT_MOVE)
(defenum Obs [OBS_NONE -1] [OBS_STAY 0] [OBS_MOVE 1])

(defmemo test- [[x X]] [t]
  (return x))

;; host functions on host numbers: plain Clojure

(defn plus-one [n] (+ n 1))
(defn returns-scalar [x] (+ (Math/cos x) (nth [0 1 2] x)))
(defn returns-scalar-no-arg [] (Math/cos 3.14))
(defn takes-nonstatic-arg [x z y] (* (nth z x) y))
(defn norm-pdf [x {:keys [loc scale]}]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x loc) scale) 2))) (* scale (Math/sqrt (* 2 Math/PI)))))
(defn kw-only-func [{:keys [a b]}] (+ a b))
(defn denotes [u r] (not= 0 (bit-and u r)))
(defn do-action [l a] (get-in [[0 1] [1 0]] [a l]))
(defn returns-nonscalar0 [] (array/array [0 1]))
(defn returns-nonscalar1 [_] (array/array [0 1]))

;; host functions on whole arrays: gen.array's operators, so differentiable

(defn ^:memo/vectorized gauss [x mu] (array/exp (array/negate (array/square (array/- x mu)))))
(defn ^:memo/vectorized sq [a] (array/* a a))

(defmemo inline-const [] []
  (return #{N5}))

(defmemo inline-call [] []
  (return (plus-one #{N5})))

(defmemo inline-memo [[x X]] []
  (return (test- [x] #{N5})))

(defmemo inline-param [] [p]
  (return (+ #{(* p 10)} 1)))

(defmemo ffi-ok [] []
  (alice :chooses [x X] :wpp 1)
  (return (+ (E (returns-scalar alice.x)) 12)))

(defmemo ffi-ok-no-arg [] []
  (return (+ (returns-scalar-no-arg) 15)))

(defmemo ffi-static [] []
  (alice :chooses [x X] :wpp (takes-nonstatic-arg x #{[0 1 2 3 4]} 3.0))
  (return (E alice.x)))

(defmemo ffi-kwargs [] []
  (agent :chooses [n N11] :wpp (norm-pdf n :scale 1 :loc 5))
  (return (E agent.n)))

(defmemo ffi-kwargs-only [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (kw-only-func :a alice.x :b 1.0))))

;; demo-rsa.py as written, with its host function
(defmemo L-ffi [[u U] [r R]] [beta t]
  (listener :thinks
    (speaker :given [r R] :wpp 1)
    (speaker :chooses [u U] :wpp (* (denotes u r)
                                    (if (= t 0) 1 (exp (* beta (L-ffi [u r] beta (- t 1))))))))
  (listener :observes speaker.u :is u)
  (listener :chooses [r R] :wpp (Pr (= speaker.r r)))
  (return (Pr (= listener.r r))))

;; demo-sally-anne.ipynb as written
(defmemo sally-anne-ffi [[marble_pos_t0 Loc] [obs Obs] [where_look Loc]] []
  (child :knows marble_pos_t0 obs where_look)
  (child :thinks
    (sally :knows marble_pos_t0)
    (sally :thinks
      (anne :knows marble_pos_t0)
      (anne :chooses [a Action] :wpp (if (= a Action.ACT_MOVE) 0.01 0.99))
      (anne :chooses [marble_pos_t1 Loc] :wpp (= (do-action marble_pos_t0 a) marble_pos_t1))
      (anne :chooses [o Obs] :wpp (if (or (= o Obs.OBS_NONE) (= o a)) 1 0)))
    (sally :observes anne.o :is obs)
    (sally :chooses [where_look Loc] :wpp (Pr (= anne.marble_pos_t1 where_look))))
  (return (as child (Pr (= sally.where_look where_look)))))

(defmemo ffi-param-grad [] [mu]
  (alice :chooses [n N11] :wpp (gauss n mu))
  (return (E alice.n)))

(defmemo g-static-ffi [] [a]
  (alice :chooses [x B2] :wpp (exp (* (sq a) x)))
  (return (Pr (= alice.x 1))))

(defmemo h [] [a]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (E alice.x)))

(defmemo calls-h [] [a]
  (return (* (h a) 2)))

(defmemo ffi-scalar0 [] []
  (return (returns-nonscalar0)))

(defmemo ffi-scalar1 [] []
  (return (returns-nonscalar1 1.0)))

(def fixture (ts/fixture "slice4.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.slice4-test fixture))

(deftest gradients-match-jax
  (ts/check-grads 'memo.slice4-test fixture))

(deftest gradients-python-memo-gets-wrong
  (ts/check-ledger 'memo.slice4-test fixture))

(deftest not-scalar-in-scalar-out
  ;; Python memo raises these when called, naming its own functions
  (doseq [[k f] [[:ffi-scalar0 ffi-scalar0] [:ffi-scalar1 ffi-scalar1]]]
    (testing (name k)
      (is (some? (get-in fixture [:runtime-errors k])))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"^The function returns-nonscalar[01]\(\.\.\.\) is not scalar-in-scalar-out"
                            (f))))))

(defn gauss-host [x mu] (Math/exp (- (Math/pow (- x mu) 2))))

(defmemo host-param [] [mu]
  (alice :chooses [n N11] :wpp (gauss-host n mu))
  (return (E alice.n)))

(deftest host-functions-and-gradients
  (testing "a host function on host numbers gives the values of its vectorized twin"
    (is (ts/close? (ts/host (ffi-param-grad 4.3)) (ts/host (host-param 4.3)))))
  (testing "but cannot be differentiated: a loud error, not Python memo's silent zero"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Cannot differentiate through the host function gauss-host"
                          ((array/grad host-param) 4.3)))))

(defn picks [x] (nth [1.0 2.0] x))

(defmemo picks-out-of-range [] []
  (alice :chooses [x X] :wpp (picks x))
  (return (E alice.x)))

(deftest a-failing-host-function-is-a-memo-error
  (let [e (try (picks-out-of-range) nil (catch Exception e e))]
    (is (err/memo-error? e))
    (is (= "The host function picks failed on [2]: IndexOutOfBoundsException" (ex-message e)))
    (is (instance? IndexOutOfBoundsException (ex-cause e)))))

(defn ^:memo/vectorized step-if [x] (if (array/< x 2) 1.0 0.0))
(defn ^:memo/vectorized step-where [x] (array/where (array/< x 2) 1.0 0.0))

(defmemo uses-step-if [] []
  (alice :chooses [x X] :wpp (step-if x))
  (return (E alice.x)))

(defmemo uses-step-where [] []
  (alice :chooses [x X] :wpp (step-where x))
  (return (E alice.x)))

(deftest vectorized-functions-are-spot-checked
  (testing "an if inside a vectorized function takes one branch for every element: loud, not silent"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"vectorized function step-if gives other values on whole arrays"
                          (uses-step-if))))
  (testing "where is the way to branch there"
    ;; (a differentiated call is checked too, on primal values: gauss in
    ;; gradients-match-jax)
    (is (ts/close? 0.5 (ts/host (uses-step-where))))))

(defmemo rec-no-axes [] [t]
  (alice :chooses [x X] :wpp 1)
  (return (if (> t 0) (+ (rec-no-axes (- t 1)) (E alice.x)) 0)))

(deftest divergences-from-python-memo
  (testing "a memo without axes recurses (Python memo recurses forever: its host calls are not guarded)"
    (is (= 2.0 (double (ts/host (rec-no-axes 2)))))))

(deftest unknown-function
  (is (= "Unknown function nonesuch"
         (ex-message (ts/compile-error 'memo.slice4-test
                                       '(defmemo calls-nonesuch [] []
                                          (return (nonesuch 1))))))))
