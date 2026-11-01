(ns memo.testpy-test
  "The parity corpus, part 1: memo 1.3.3's own test suite (demo/test.py),
  every case, under its own name. testpy.edn holds Python memo's outcome for
  each (generate.py runs test.py itself with a recording memo_test); each
  case here has a Clojure twin, a form that must not compile, or a reason it
  has no counterpart in memo.clj."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

;; test.py's module

(def X [0 1 2])
(def Y [0 1 2])
(def N5 5)                                  ; test.py's N, until it becomes np.arange(11)
(def N (vec (range 11)))                    ; test.py's later N
(def Z (vec (range 1000)))
(def Z30 (vec (range 30)))                  ; post-optim-distinctness's: the fixture's :changes
(def R (mapv #(+ -10.0 (* 20.0 (/ % 999.0))) (range 1000)))
(defenum Bool NO YES)

(defn f [n] (array/+ (array/array n) 1))     ; jax.jit n + 1: an array stays an array
(defn g [n] (reduce + n))                    ; n.sum()
(defn normpdf [x mu sigma]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x mu) sigma) 2))) (* sigma (Math/sqrt (* 2 Math/PI)))))
(defn returns-scalar [x] (+ (Math/cos x) (nth [0 1 2] x)))
(defn returns-scalar-no-arg [] (Math/cos 3.14))
(defn returns-nonscalar0 [] (array/array [0 1]))
(defn returns-nonscalar1 [_] (array/array [0 1]))
(defn returns-nonscalar-kwarg [_ {:keys [scale] :or {scale 1.0}}] (array/* (array/array [0 1]) scale))
(defn takes-nonstatic-arg [x z y] (* (nth z x) y))
(defn norm-pdf [x {:keys [loc scale]}]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x loc) scale) 2))) (* scale (Math/sqrt (* 2 Math/PI)))))
(defn kw-only-func [{:keys [a b]}] (+ a b))
(defn array-index [arr i] (nth arr i))

(defn- positional-and-keywords
  "A Clojure function's arguments as Python passes them: positional, then
  keywords in a trailing map."
  [more]
  (if (map? (last more)) [(vec (butlast more)) (last more)] [(vec more) {}]))

(defn exotic-kwarg-func [x & more]                  ; (x, arr, scale=1.0)
  (let [[pos kw] (positional-and-keywords more)
        arr (if (seq pos) (first pos) (:arr kw))]
    (* (nth arr x) (:scale kw 1.0))))

(defn exotic-both-func [x & more]                   ; (x, arr, weights)
  (let [[pos kw] (positional-and-keywords more)
        arr (if (seq pos) (first pos) (:arr kw))
        weights (if (next pos) (second pos) (:weights kw))]
    (* (nth arr x) (nth weights x))))

;; the twins, in test.py's order

(defmemo test- [[x X]] [t]
  (return x))

(defmemo observes-call [[x X]] []
  (a :thinks (b :chooses [x X] :wpp 1))
  (a :observes b.x :is x)
  (return (as a (as b (test- [x] 0)))))

(defmemo observes-other [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp 1)
  (alice :observes bob.x :is charlie.x)
  (return (E (as alice bob.x))))

(defmemo observes-other-imagine [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp 1)
  (alice :observes bob.x :is charlie.x)
  (return (E (as alice (imagine (env :knows bob.x) (+ bob.x 1))))))

(defmemo imagine-ok [] []
  (return (as alice (imagine (bob :chooses [y X] :wpp 1) (E bob.y)))))

(defmemo inline [] []
  (return #{N5}))

(defmemo inline-call [] []
  (return (f #{N5})))

(defmemo inline-memo [[x X]] []
  (return (test- [x] #{N5})))

(defmemo memo-call-ellipsis [] [(t 0)]
  (alice :chooses [x X] :wpp (test- [x] ...))
  (return (E alice.x)))

(defmemo recursive-kwarg [[x X]] [a (b 0)]
  (return (if (> a 0) (recursive-kwarg [x] :a (- a 1) :b (+ b 1)) b)))

(defmemo kwarg [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (recursive-kwarg [alice.x] :a 3))))

;; cost reflection (M4, DESIGN.md §14.1)
(defmemo cost-1 [] []
  (return (cost @(inline))))

(defmemo cost-kwarg [] []
  (alice :chooses [x X] :wpp 1)
  (return (cost @(recursive-kwarg :a 2))))

(defmemo cost-time [] []
  (return (time @(inline))))

(defmemo cost-time-matches-cost [] []
  (return (= (time @(recursive-kwarg :a 2)) (cost @(recursive-kwarg :a 2)))))

(defmemo cost-space [] []
  (alice :chooses [x X] :wpp 1)
  (return (space @(recursive-kwarg :a 2))))

(defmemo cost-size [] []
  (return (size @(inline))))

(defmemo cost-size-kwarg [] []
  (alice :chooses [x X] :wpp 1)
  (return (size @(recursive-kwarg :a 2))))

(defmemo kwarg-mixed [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (recursive-kwarg [alice.x] 3 :b 10))))

(defmemo imagine-knows [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (as alice (imagine (world :knows x) world.x)))))

(defmemo imagine-knows-other [[z X]] []
  (alice :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [z X] :wpp 1))
  (alice :observes bob.z :is z)
  (return (as alice (imagine (world :knows bob.z) (as world bob.z)))))

(defmemo imagine-future-stress [] []
  (alice :chooses [x X] :wpp 1)
  (alice :thinks (bob :chooses [z X] :wpp 1))
  (alice :snapshots-self-as future_alice)
  (return (E (as alice
                 (imagine
                  (world :knows x bob.z)
                  (world :chooses [z X] :wpp 1)
                  (future_alice :chooses [y X] :wpp (+ x y))
                  (future_alice :thinks (world :chooses [z X] :wpp 1))
                  (future_alice :observes world.z :is world.z)
                  (E (+ future_alice.y (as future_alice (+ world.z y)) world.z (as world bob.z))))))))

(defmemo imagine-toplevel [] []
  (return (imagine (alice :chooses [x X] :wpp 1) (E alice.x))))

(defmemo ffi-ok [] []
  (alice :chooses [x X] :wpp 1)
  (return (+ (E (returns-scalar alice.x)) 12)))

(defmemo ffi-ok-no-arg [] []
  (return (+ (returns-scalar-no-arg) 15)))

(defmemo ffi-scalar0 [] []
  (return (returns-nonscalar0)))

(defmemo ffi-scalar1 [] []
  (return (returns-nonscalar1 1.0)))

(defmemo ffi-scalar-kwarg [] []
  (return (returns-nonscalar-kwarg 1.0 :scale 2.0)))

(defmemo observes-const [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :observes-that (= bob.x 0))
  (return (as alice (E bob.x))))

(defmemo observes-const-float [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :observes-event :wpp (/ bob.x 3))
  (return (as alice (E bob.x))))

(defmemo observes-const-void-choose [] []
  (alice :chooses [x X] :wpp 1)
  (alice :observes-event :wpp (/ x 3))
  (return (E alice.x)))

(defmemo observes-const-void [] []
  (alice :observes-event :wpp 3.14)
  (return (as alice 2)))

(defmemo pr-joint [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y X] :wpp 1)
  (return (Pr (= alice.x 0) (= alice.y 0))))

(defmemo choose-many [] []
  (alice :chooses [x X] [y Y] :wpp 1)
  (return (Pr (= alice.x 0) (= alice.y 0))))

(defmemo choose-max [] []
  (alice :chooses [x X] [y Y] :to-maximize (+ x y))
  (return (Pr (= alice.x 0) (= alice.y 0))))

(defmemo choose-min [] []
  (alice :chooses [x X] [y Y] :to-minimize (+ x y))
  (return (Pr (= alice.x 0) (= alice.y 0))))

(defmemo post-optim [[z1 Z] [z2 Z]] []
  (alice :chooses [z1 Z] :wpp 1)
  (alice :chooses [z2 Z] :wpp 1)
  (return (Pr (= z1 alice.z1) (= alice.z2 z2))))

(defmemo post-optim-distinctness [[z Z30]] []
  (alice :chooses [z1 Z30] :wpp 1)
  (alice :chooses [z2 Z30] :wpp 1)
  (return (Pr (= z alice.z1) (= z alice.z2))))

(defmemo post-diag-1 [[x X]] []
  (bob :knows x)
  (bob :chooses [y X] :wpp (+ x y))
  (return (Pr (= bob.y x))))

(defmemo post-diag-2 [[x X]] []
  (bob :thinks (alice :chooses [x X] :wpp 1))
  (bob :observes alice.x :is x)
  (bob :chooses [y X] :wpp (+ alice.x y))
  (return (Pr (= bob.y x))))

(defmemo post-diag-3 [[x X]] []
  (bob :chooses [x X] :wpp 1)
  (bob :chooses [y X] :wpp (+ x y))
  (return (Pr (= bob.y x))))

(defmemo kl [] []
  (alice :chooses [p R] :wpp (normpdf p 0 1))
  (alice :chooses [q R] :wpp (normpdf q 0 2))
  (return (KL alice.p alice.q)))

(defmemo kl-victor [[x X]] []
  (bob :thinks (alice :given [x X] :wpp 1) (env :chooses [x X] :wpp 1))
  (return (as bob (KL alice.x env.x))))

(defmemo predict-no-deps-warning [] []
  (alice :chooses [x X] :to-maximize (Predict x))
  (return 1.0))

(defmemo predict [] []
  (bob :chooses [z X] :wpp 1)
  (bob :thinks
    (alice :chooses [x X] :to-maximize (Predict z))
    (alice :chooses [y X] :to-maximize (Predict z))
    (alice :thinks (world :knows x y) (world :chooses [z X] :wpp (+ z x y)))
    (alice :observes world.z :is z)
    (alice :chooses [z X] :wpp (+ world.z z)))
  (return (E (as bob (E alice.x)))))

(defmemo wants [[x X] [y X]] []
  (bob :thinks
    (alice :wants u (+ x y))
    (alice :wants v (- x y))
    (alice :chooses [x X] :to-maximize (EU u))
    (alice :chooses [y X] :to-maximize (EU v)))
  (bob :knows x y)
  (return (as bob (Pr (= alice.x x) (= alice.y y)))))

(defmemo jianlin [[x X]] []
  (a :thinks (b :chooses [u X] :wpp 1.0))
  (a :observes b.u :is x)
  (a :chooses [v X] :wpp (= b.u v))
  (return (Pr (= a.v x))))

(defmemo ffi-static [] []
  (alice :chooses [x X] :wpp (takes-nonstatic-arg x #{[0 1 2 3 4]} 3.0))
  (return (E alice.x)))

(defmemo ffi-kwargs [] []
  (agent :chooses [n N] :wpp (norm-pdf n :scale 1 :loc 5))
  (return (E agent.n)))

(defmemo ffi-kwargs-only [] []
  (alice :chooses [x X] :wpp 1)
  (return (E (kw-only-func :a alice.x :b 1.0))))

(defmemo exotic-param-2 [] [(^:array x X)]
  (return (f x)))

(defmemo exotic-param-3 [] [(^:array x "not_an_array")]
  (return (f x)))

(defmemo exotic-param-4 [] [(x X)]
  (return (f x)))

(defmemo exotic-param-5 [] [(^:array x X)]
  (return (+ (g x) (array-index x 0))))

(defmemo exotic-param-kwarg-item [] [(^:array arr [1 1 1])]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x :arr arr :scale 2.0))
  (return (E alice.x)))

(defmemo exotic-param-arg-with-kwarg-item [] [(^:array arr [1 1 1])]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x arr :scale 2.0))
  (return (E alice.x)))

(defmemo exotic-param-both-item [] [(^:array arr [1 1 1]) (^:array weights [1 1 1])]
  (alice :chooses [x X] :wpp (exotic-both-func x arr :weights weights))
  (return (E alice.x)))

(defmemo exotic-kwarg-recursive [[x X]] [k ^:array arr]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x arr :scale 1.0))
  (return (if (> k 0) (exotic-kwarg-recursive [x] :k (- k 1) :arr arr) 1)))

(defmemo exotic-kwarg [] [(^:array arr [1 1 1])]
  (alice :chooses [x X] :wpp 1)
  (return (E (exotic-kwarg-recursive [alice.x] :k 2 :arr arr))))

(defmemo exotic-with-regular-recursive [[x X]] [k ^:array arr scale]
  (alice :chooses [x X] :wpp (exotic-kwarg-func x arr :scale scale))
  (return (if (> k 0) (exotic-with-regular-recursive [x] :k (- k 1) :arr arr :scale scale) 1)))

(defmemo exotic-with-regular-kwarg [] [(^:array arr [1 1 1])]
  (alice :chooses [x X] :wpp 1)
  (return (E (exotic-with-regular-recursive [alice.x] :k 2 :arr arr :scale 2.0))))

(defmemo multi-return-1 [[x X]] []
  (return 1)
  (return 2))

(defmemo multi-return-2 [[x X]] []
  (return (multi-return-1 0 [x])))

(defmemo multi-return-3 [[x X]] []
  (return (multi-return-1 [x])))

(defmemo multi-return-4 [[x X]] []
  (return (multi-return-1 3 [x])))

(defmemo multi-return-5 [[x X]] []
  (return (test- 0 [x])))

(defmemo such-that [] []
  (alice :chooses [n N] :such-that (< n 3))
  (return (E alice.n)))

(defmemo uniformly-1 [] []
  (alice :chooses [n N] :uniformly)
  (return (E alice.n)))

(defmemo uniformly-2 [] []
  (alice :chooses [n N] [m N] :uniformly)
  (return (E (- alice.n alice.m))))

(defmemo has-theory-child-2 [] []
  (child :has-theory :child)
  (child :thinks (adult :chooses [x X] :wpp 1))
  (return 0))

(defmemo ^{:variant :child} has-theory-child-4 [] []
  (child :chooses [x X] :uniformly)
  (return 0))

(defmemo to-be-1 [] []
  (alice :chooses [x X] :to-be 2)
  (return (E alice.x)))

(defmemo to-be-2 [] []
  (alice :chooses [x X] [y X] :to-be 2)
  (return (E (+ alice.x alice.y))))

(defmemo to-be-3 [] []
  (alice :chooses [x X] :to-be x)
  (return (E alice.x)))

(defmemo to-be-4 [] []
  (alice :chooses [x X] [y Y] :to-be (+ x y))
  (return (E (+ alice.x alice.y))))

(defmemo forgets-about-1 [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (+ x y))
  (alice :forgets-about x)
  (return (E alice.y)))

(defmemo forgets-about-multiple [] []
  (alice :chooses [x X] :uniformly)
  (alice :chooses [y Y] :uniformly)
  (alice :forgets-about x y)
  (alice :chooses [z Bool] :uniformly)
  (return (E alice.z)))

(defmemo var-redundant-known [] []
  (alice :chooses [coffee Bool] :uniformly)
  (bob :knows alice.coffee)
  (return (E (as bob (Var (= alice.coffee 1))))))

;; kach/memo issue 126, fixed in 1.3.3: a copy is unaffected by later observations

(defmemo snapshot-unaffected-by-later-observe [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp (= x 2))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is charlie.x)
  (return (as alice (as past_alice (E bob.x)))))

(defmemo snapshot-unaffected-by-later-observe-two-vars [] []
  (alice :thinks (bob :chooses [x X] :wpp 1) (bob :chooses [y X] :wpp (= y x)))
  (charlie :chooses [x X] :wpp (= x 2))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is charlie.x)
  (return (as alice (as past_alice (E (+ bob.x bob.y))))))

(defmemo snapshot-unaffected-by-later-observe-issue [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E bob.x)))))

(defmemo snapshot-unaffected-by-later-observe-two-vars-issue [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1) (bob :chooses [y X] :wpp (= y x)))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E (+ bob.x bob.y))))))

(defmemo snapshot-observing-self-unaffected [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (charlie :chooses [x X] :wpp (= x 2))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is charlie.x)
  (return (E (as alice (+ (as past_alice (E bob.x)) bob.x 1)))))

(defmemo imagine-observe-unaffected-outside [] []
  (alice :chooses [z X] :wpp (= z 2))
  (alice :thinks (carol :thinks (bob :chooses [x X] :wpp 1)))
  (return (E (as alice (+ (imagine (carol :observes bob.x :is z)
                                   (as carol bob.x))
                          (as carol (E bob.x)))))))

;; cases that must not compile

(def compile-forms
  {:chooses-multiple
   '(defmemo chooses-multiple [] []
      (bob :chooses [x X] :wpp 1)
      (bob :chooses [x X] :wpp 1)
      (return 1))
   :imagine-unknown-err
   '(defmemo imagine-unknown-err [] []
      (return (as alice (imagine (bob :chooses [y X] :wpp 1) bob.y))))
   :imagine-unknown-err-expect
   '(defmemo imagine-unknown-err-expect [] []
      (return (as alice (E (imagine (bob :chooses [y X] :wpp 1) bob.y)))))
   :imagine-unknown-err-expect-future
   '(defmemo imagine-unknown-err-expect-future [] []
      (return (as alice (E (imagine (future_alice :chooses [y X] :wpp 1) future_alice.y)))))
   :choose-err
   '(defmemo choose-err [] []
      (alice :chooses [x X] [y Y] :to-eat (+ x y))
      (return (Pr (= alice.x 0) (= alice.y 0))))
   :kl-fail-unknown
   '(defmemo kl-fail-unknown [] []
      (alice :chooses [p R] :wpp (normpdf p 0 1))
      (alice :chooses [q R] :wpp (normpdf q 0 2))
      (return (KL alice.r alice.q)))
   :kl-fail-known
   '(defmemo kl-fail-known [[r R]] []
      (alice :knows r)
      (alice :chooses [q R] :wpp (normpdf q 0 2))
      (return (KL alice.r alice.q)))
   :kl-fail-dom
   '(defmemo kl-fail-dom [] []
      (alice :chooses [p Z] :wpp (normpdf p 0 1))
      (alice :chooses [q R] :wpp (normpdf q 0 2))
      (return (KL alice.p alice.q)))
   :predict-fail-knows
   '(defmemo predict-fail-knows [] []
      (bob :chooses [z X] :wpp 1)
      (bob :thinks
        (alice :chooses [x X] :to-maximize (Predict x))
        (alice :knows z))
      (return (E (as bob (E alice.x)))))
   :param-name-conflict
   '(defmemo param-name-conflict [] [x]
      (alice :chooses [x X] :wpp 1)
      (return 1.0))
   :exotic-param-1
   '(defmemo exotic-param-1 [] [(^:array x X)]
      (return x))
   :uniformly-3
   '(defmemo uniformly-3 [] []
      (alice :chooses [n N] [m N] :uniformly :wpp 1)
      (return (E (- alice.n alice.m))))
   :uniformly-4
   '(defmemo uniformly-4 [] []
      (alice :chooses :uniformly)
      (return 1))
   :has-theory-child-1
   '(defmemo has-theory-child-1 [] []
      (child :has-theory :child)
      (child :thinks (adult :thinks (mom :inspect)))
      (return 0))
   :has-theory-child-3
   '(defmemo ^{:variant :child} has-theory-child-3 [] []
      (child :thinks (adult :chooses [x X] :wpp 1))
      (return 0))
   :forgets-about-scope-parent
   '(defmemo forgets-about-scope-parent [] []
      (alice :chooses [x X] :wpp 1)
      (alice :forgets-about x)
      (return (E alice.x)))
   :forgets-about-scope-self
   '(defmemo forgets-about-scope-self [] []
      (alice :chooses [x X] :wpp 1)
      (alice :forgets-about x)
      (alice :chooses [y Y] :wpp x)
      (return 0))
   :forgets-about-not-own-choice
   '(defmemo forgets-about-not-own-choice [[x X]] []
      (alice :knows x)
      (alice :forgets-about x)
      (return 0))})

(def not-applicable
  "Cases with no counterpart in memo.clj, and why."
  (let [kwargs "Python's **kwargs syntax, which memo.clj's syntax has no counterpart for"]
    {:kwargs-unpacking-err kwargs
     :cost-kwargs-unpacking-err kwargs
     :ffi-kwargs-expansion kwargs}))

(def memo-clj-values
  "Where memo.clj's value differs from Python memo's on purpose, a check of
  memo.clj's: time @ and space @ count memo.clj's own run, not XLA's flops
  and scratch bytes (DESIGN.md §14.1; memo.cost-test checks the counts)."
  {;; recursive_kwarg(a=2): three levels of 7 operations, each counted where
   ;; it is called (Python XLA: 12)
   :cost-kwarg #(= 21.0 (double %))
   ;; bytes at the backend's dtypes, so the backend's own number
   :cost-space #(pos? (double %))})

(def memo-clj-messages
  "Where memo.clj's message differs from Python memo's on purpose: it shows
  memo.clj's syntax and names, or drops Python's exception type."
  (let [not-scalar #(str "The function " % "(...) is not scalar-in-scalar-out. memo can only handle "
                         "host functions that take scalars as input and return a single scalar as output.")]
    {:choose-err "unknown keyword argument to chooses: :to-eat"
     :uniformly-3 "Cannot use `:uniformly` together with a keyword argument"
     :ffi-scalar0 (not-scalar "returns-nonscalar0")
     :ffi-scalar1 (not-scalar "returns-nonscalar1")
     :ffi-scalar-kwarg (not-scalar "returns-nonscalar-kwarg")
     :exotic-param-2 (not-scalar "f")
     :exotic-param-3 "Parameter x was not an array, despite being annotated as ^:array x."
     :exotic-param-4 (str "Parameter x was not a numeric scalar, but rather an array. By default, all "
                          "memo parameters must be scalars. Annotate this parameter as ^:array x if you "
                          "really did intend to pass in a non-scalar.")
     :multi-return-3 (str "You are calling a memo model that has multiple return values, but you have "
                          "not specified which return value you want. Try writing (model 0 [...] ...) "
                          "to get the first return value.")
     :multi-return-4 "Your memo model does not have enough return values."
     :multi-return-5 (str "You are calling a memo model with only one return value, so you do not have "
                          "to specify which return value you want.")}))

(def fixture (ts/fixture "testpy.edn"))

(defn- twin [k] (some-> (ns-resolve 'memo.testpy-test (symbol (name k))) deref))

(deftest every-case-of-memos-test-suite
  (is (= ["post_optim_distinctness over Z30 = np.arange(30), not Z"] (:changes fixture)))
  (doseq [{k :name :keys [outcome stage message args shape value digest]} (:cases fixture)]
    (testing (name k)
      (cond
        (not-applicable k)
        (is (string? (not-applicable k)))

        (= "compile" stage)
        (is (= (get memo-clj-messages k message)
               (some-> (ts/compile-error 'memo.testpy-test (compile-forms k)) ex-message)))

        (= "call" stage)
        (let [e (try (apply (twin k) args) nil (catch Exception e e))]
          (is (err/memo-error? e) "compiles, and fails loudly when called")
          (is (= (get memo-clj-messages k message) (some-> e ex-message))))

        :else
        (let [r (try (apply (twin k) args) (catch Exception e e))]
          (if (instance? Exception r)
            (is false (str "threw: " (ex-message r)))
            (do (is (= shape (array/shape r)))
                (cond
                  (memo-clj-values k)
                  (is ((memo-clj-values k) (ts/host r)) (str "memo.clj " (pr-str (ts/host r))))

                  digest
                  (is (ts/matches-digest? digest r))

                  :else
                  (is (ts/close? value (ts/host r))
                      (str "python " (pr-str value) "\nmemo.clj " (pr-str (ts/host r))))))))))))

(deftest the-port-is-complete
  (let [names (set (map :name (:cases fixture)))]
    (is (= 102 (count (:cases fixture))) "test.py's 101 cases, imagine_ok defined twice")
    (doseq [k names]
      (is (or (not-applicable k) (compile-forms k) (twin k)) (str (name k) " has no twin")))
    (is (every? names (concat (keys compile-forms) (keys not-applicable) (keys memo-clj-messages)
                               (keys memo-clj-values)))
        "no twin, form or message for a case test.py does not have")))
