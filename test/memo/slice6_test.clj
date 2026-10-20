(ns memo.slice6-test
  "M1, slice 6: H, KL, forgets-about, inspect, cast, array parameters and the
  cache. Every model here has a twin of the same name in
  test/fixtures/generate.py, and slice6.edn holds Python memo's results."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(def X [0 1 2])
(defenum Coin HEADS TAILS)
(defenum Bool NO YES)

(def Rkl (mapv #(+ -10.0 (* 20.0 (/ % 999.0))) (range 1000)))

(defn normpdf [x mu sigma]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x mu) sigma) 2))) (* sigma (Math/sqrt (* 2 Math/PI)))))

(defmemo kl [] []
  (alice :chooses [p Rkl] :wpp (normpdf p 0 1))
  (alice :chooses [q Rkl] :wpp (normpdf q 0 2))
  (return (KL alice.p alice.q)))

(defmemo kl-victor [[x X]] []
  (bob :thinks (alice :given [x X] :wpp 1) (env :chooses [x X] :wpp 1))
  (return (as bob (KL alice.x env.x))))

(defmemo entropy-uniform [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (return (as alice (H bob.x))))

(defmemo entropy-joint [] [beta]
  (alice :thinks (bob :chooses [x X] :wpp (exp (* beta x))) (bob :chooses [y X] :wpp (>= y x)))
  (return (as alice (H bob.x bob.y))))

(defmemo entropy-marginal [] []
  (alice :thinks (bob :chooses [x X] :wpp 1) (bob :chooses [y X] :wpp (= y x)))
  (return (as alice (H bob.y))))

(def is-prime [0 0 1 1 0 1 0 1 0 0 0 1 0])
(def Ne [1 2 3 4 5 6])
(def Qe [0 1 2 3])
(def Ae [0 1])

(defn respond [q a n]
  (= a (case (long q)
         0 (if (= n 7) 1 0)
         1 (if (> n 6) 1 0)
         2 (if (even? n) 1 0)
         3 (nth is-prime n))))

;; demo-eig.ipynb
(defmemo eig [[q Qe]] []
  (alice :knows q)
  (alice :thinks
    (bob :chooses [n_red Ne] :wpp 1)
    (bob :chooses [n_blu Ne] :wpp 1)
    (bob :knows q)
    (bob :chooses [a Ae] :wpp (respond q a (+ n_red n_blu))))
  (alice :snapshots-self-as future_self)
  (return (as alice (imagine
                     (future_self :observes bob.a :is bob.a)
                     (- (H bob.n_red bob.n_blu) (E (as future_self (H bob.n_red bob.n_blu))))))))

(defmemo forgets-about-1 [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y X] :wpp (+ x y))
  (alice :forgets-about x)
  (return (E alice.y)))

(defmemo forgets-about-multiple [] []
  (alice :chooses [x X] :uniformly)
  (alice :chooses [y X] :uniformly)
  (alice :forgets-about x y)
  (alice :chooses [z Bool] :uniformly)
  (return (E alice.z)))

(defmemo with-cast [] []
  (cast [alice bob])
  (alice :chooses [x X] :wpp 1)
  (bob :knows alice.x)
  (return (E (as bob alice.x))))

(defn g-sum [n] (reduce + n))
(defn arr-at [arr i] (nth arr i))
(defn exotic-kwarg-func [x {:keys [arr scale] :or {scale 1.0}}] (* (nth arr x) scale))

(defmemo exotic-param [] [(^:array x X)]
  (return (+ (g-sum x) (arr-at x 0))))

(defmemo exotic-param-kwarg [] [(^:array arr [1 2 3])]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x :arr arr :scale 2.0))
  (return (E alice.x)))

(defmemo exotic-kwarg-recursive [[x X]] [k ^:array arr]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x :arr arr :scale 1.0))
  (return (if (> k 0) (exotic-kwarg-recursive [x] :k (- k 1) :arr arr) 1)))

(defmemo exotic-kwarg [] [(^:array arr [1 2 3])]
  (alice :chooses [x X] :wpp 1)
  (return (E (exotic-kwarg-recursive [alice.x] :k 2 :arr arr))))

(defenum DayAction SH GO)
(defenum NightAction Retire Party)
(defenum NightType Responsible Fun)
(def Pp (mapv #(/ % 100.0) (range 101)))

(defn one-night-payoff [night-action] (nth [1.0 2.0] night-action))
(defn one-day-payoff [day-action night-action] (get-in [[0.0 0.0] [1.0 -1.0]] [day-action night-action]))
(defn ber [k p] (let [p (if (boolean? p) (if p 1.0 0.0) p)] (if (== k 1) p (- 1.0 p))))

;; demo-penelope.ipynb
(defmemo ^{:cache true} day-model [[p Pp] [da DayAction]] [(n 0)]
  (day :knows p da)
  (day :thinks
    (night :knows p)
    (night :given [t NightType] :wpp (ber t p))
    (night :chooses [na NightAction]
           :wpp (if (= t NightType.Responsible) (= na NightAction.Retire) (ber na (night-model 0 [p] n)))))
  (day :snapshots-self-as next_day)
  (return (as day (+ (E (one-day-payoff da night.na))
                     (if (<= n 0)
                       0
                       (imagine
                        (next_day :observes night.na :is night.na)
                        (next_day :chooses [p_ Pp] :to-minimize (abs (- p_ (Pr (= night.t NightType.Fun)))))
                        (next_day :chooses [da_ DayAction] :to-maximize (day-model [p_ da_] (- n 1)))
                        (E (as next_day (day-model [p_ da_] (- n 1))))))))))

(defmemo ^{:cache true} night-model [[p Pp]] [(n 0)]
  (night :knows p)
  (night :wants payoff (+ (one-night-payoff na)
                          (if (<= n 0) 0 (E (if (= day.da_ DayAction.GO) (night-model 1 [p] (- n 1)) 0)))))
  (night :chooses [sigma Pp] :to-maximize (EU payoff))
  (night :chooses [na NightAction] :wpp (ber na sigma))
  (night :thinks
    (day :knows p sigma)
    (day :thinks
      (night :knows p sigma)
      (night :given [t NightType] :wpp (ber t p))
      (night :chooses [na NightAction]
             :wpp (if (= t NightType.Responsible) (= na NightAction.Retire) (ber na sigma))))
    (day :observes night.na :is na)
    (day :chooses [p_ Pp] :to-minimize (abs (- p_ (Pr (= night.t NightType.Fun)))))
    (day :chooses [da_ DayAction] :to-maximize (if (<= n 0) 1 (day-model [p_ da_] (- n 1)))))
  (return (E night.sigma))
  (return (E (as night (EU payoff)))))

(def fixture (ts/fixture "slice6.edn"))

(deftest results-match-python-memo
  (ts/check-cases 'memo.slice6-test fixture))

(deftest gradients-match-jax
  (ts/check-grads 'memo.slice6-test fixture))

(def error-forms
  {:kl-fail-unknown
   '(defmemo kl-fail-unknown [] []
      (alice :chooses [p X] :wpp 1)
      (alice :chooses [q X] :wpp 1)
      (return (KL alice.r alice.q)))
   :kl-fail-known
   '(defmemo kl-fail-known [[r X]] []
      (alice :knows r)
      (alice :chooses [q X] :wpp 1)
      (return (KL alice.r alice.q)))
   :kl-fail-dom
   '(defmemo kl-fail-dom [] []
      (alice :chooses [p X] :wpp 1)
      (alice :chooses [q Coin] :wpp 1)
      (return (KL alice.p alice.q)))
   :entropy-unknown
   '(defmemo entropy-unknown [] []
      (return (as alice (H bob.x))))
   :entropy-known
   '(defmemo entropy-known [] []
      (bob :chooses [x X] :wpp 1)
      (alice :knows bob.x)
      (return (as alice (H bob.x))))
   :forgets-scope-parent
   '(defmemo forgets-scope-parent [] []
      (alice :chooses [x X] :wpp 1)
      (alice :forgets-about x)
      (return (E alice.x)))
   :forgets-scope-self
   '(defmemo forgets-scope-self [] []
      (alice :chooses [x X] :wpp 1)
      (alice :forgets-about x)
      (alice :chooses [y X] :wpp x)
      (return 0))
   :forgets-not-own
   '(defmemo forgets-not-own [[x X]] []
      (alice :knows x)
      (alice :forgets-about x)
      (return 0))
   :cast-err
   '(defmemo cast-err [] []
      (cast [alice])
      (alice :chooses [x X] :wpp 1)
      (bob :chooses [y X] :wpp 1)
      (return 1))
   :exotic-misused
   '(defmemo exotic-misused [] [^:array arr]
      (return arr))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.slice6-test (error-forms k)) ex-message))))))

(deftest kl-against-a-zero-probability
  ;; log(p/0) is +inf, and nan_to_num makes it the dtype's largest value, as
  ;; in Python memo; only the dtype differs (DESIGN.md §2.4, §13.8)
  (let [f (do (ts/compile-error 'memo.slice6-test
                                '(defmemo kl-zero [] []
                                   (alice :chooses [p Bool] :wpp 1)
                                   (alice :chooses [q Bool] :wpp (= q 0))
                                   (return (KL alice.p alice.q))))
              @(ns-resolve 'memo.slice6-test 'kl-zero))
        v (double (ts/host (f)))]
    (is (and (Double/isFinite v) (> v 1e37)))))

(deftest inspect-prints-what-an-agent-tracks
  (let [out (with-out-str
              (ts/compile-error 'memo.slice6-test
                                '(defmemo inspected [] []
                                   (alice :thinks (bob :chooses [x X] :wpp 1))
                                   (alice :inspect)
                                   (return 1))))]
    (is (str/includes? out "** Tracing alice"))
    (is (str/includes? out "bob.x: X (uncertain)"))
    (is (str/includes? out "alice is being modeled by observer"))))

(def calls (atom 0))
(defn counted [x] (swap! calls inc) x)

(defmemo ^{:cache true} cached-model [] [a]
  (alice :chooses [x X] :wpp (counted (+ x a)))
  (return (E alice.x)))

(defmemo ^{:cache true} cached-soft [] [a]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (E alice.x)))

(deftest the-cache
  (reset! calls 0)
  (let [v (ts/host (cached-model 1.0))]
    (is (= v (ts/host (cached-model 1.0))))
    (is (= 3 @calls) "the second call is served from the cache")
    (ts/compile-error 'memo.slice6-test '(defmemo unrelated [] [] (return 1)))
    (cached-model 1.0)
    (is (= 6 @calls) "any memo's (re)definition drops the cache: a callee may have changed"))
  (testing "a value being differentiated is never served from the cache"
    (cached-soft 1.0)
    ;; d/da E[x] is Var[x] here; a cached number would make it 0
    (is (> (ts/host ((array/grad cached-soft) 1.0)) 0.3))))

(defn arr-total [a] (array/sum a))

(defmemo exotic-arr [] [^:array arr]
  (return (arr-total arr)))

(deftest array-parameters
  (is (ts/close? 3.0 (ts/host (exotic-param [0 1 2]))) "a vector")
  (is (ts/close? 6.0 (ts/host (exotic-arr (array/array [1 2 3])))) "a gen.array array, passed whole")
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"was not an array, despite being annotated"
                        (exotic-param 3.0))))
