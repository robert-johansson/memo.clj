(ns memo.gen-test
  "M2: memo.gen, a memo as a gen.clj distribution or deterministic
  generative function (DESIGN.md §6). Its laws (§6.3): each distribution
  against its memo's table (normalization, log-prob, frequencies, draws in
  the domain), gradients against finite differences, the zero-cell gradient
  and the NaN guard, the table's checks, batched runs, and gen.clj's GFI
  laws over models that trace memo distributions."
  (:require [clojure.java.basis :as basis]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.batched :as gb]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.gen :as mg]
            [memo.test-support :as ts :refer [close? host]]))

;; gen.laws is under gen.clj's test/, which a dependency does not get: the
;; test/ of the gen.clj in use (the pin's checkout, or the working copy under
;; :dev) goes on the classpath, so the laws are that gen.clj's own. In the
;; same form as the require: each top-level form is evaluated under a new
;; class loader.
(let [root (get-in (basis/current-basis) [:libs 'gen.clj/gen.clj :deps/root])]
  (.addURL ^clojure.lang.DynamicClassLoader (clojure.lang.RT/baseLoader)
           (.toURL (.toURI (io/file root "test"))))
  (require '[gen.laws :as laws]))

(alter-var-root #'err/*warnings?* (constantly false))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(defenum Opt PASS FIX FLEE)
(defenum Access OFF ON)
(def X [10 20 30])

(defn available [o fix-on flee-on] (cond (= o 0) 1 (= o 1) fix-on :else flee-on))
(defn utility [o] (nth [0.2 1.0 0.6] o))

;; GenGDT's mind (src/gengdt/memo_agent.clj): FIX and FLEE each come to mind
;; with probability q, then a softmax choice among what came to mind. At
;; q = 0 its table is [1 0 0].
(defmemo choose [[a Opt]] [beta q]
  (agent :chooses [fixon Access] :wpp (+ (* fixon q) (* (- 1 fixon) (- 1 q))))
  (agent :chooses [fleeon Access] :wpp (+ (* fleeon q) (* (- 1 fleeon) (- 1 q))))
  (agent :chooses [o Opt] :wpp (* (available o fixon fleeon) (exp (* beta (utility o)))))
  (return (Pr (= agent.o a))))

;; FLEE weighted by q: at q = 0, [1 e^beta 0]/(1 + e^beta)
(defmemo flee [[a Opt]] [beta q]
  (agent :chooses [o Opt] :wpp (* (if (= o Opt.FLEE) q 1) (exp (* beta o))))
  (return (Pr (= agent.o a))))

;; a domain whose values are not indices
(defmemo pickx [[x X]] [beta]
  (agent :chooses [x X] :wpp (exp (* beta (/ x 10))))
  (return (Pr (= agent.x x))))

(defmemo joint [[a Opt] [x X]] [beta]
  (agent :chooses [a Opt] :wpp (exp (* beta a)))
  (agent :chooses [x X] :wpp (+ a (/ x 10)))
  (return (Pr (and (= agent.a a) (= agent.x x)))))

;; RSA (slice 3's, from memo's demo-rsa.py)
(defenum U [GREEN 1] [PINK 2] [SQUARE 4] [ROUND 8])
(defenum R [GREEN_SQUARE 5] [GREEN_CIRCLE 9] [PINK_CIRCLE 10])

(defmemo L0 [[u U] [r R]] []
  (listener :knows u)
  (listener :chooses [r R] :wpp (= (mod (/ (- r (mod r u)) u) 2) 1))
  (return (Pr (= listener.r r))))

;; NaN gradient wherever L0 has a zero, with the right values (DESIGN.md §2.4)
(defmemo S1 [[r R] [u U]] [alpha]
  (speaker :knows r)
  (speaker :chooses [u U] :wpp (exp (* alpha (log (L0 [u r])))))
  (return (Pr (= speaker.u u))))

(defmemo S1-pow [[r R] [u U]] [alpha]
  (speaker :knows r)
  (speaker :chooses [u U] :wpp (** (L0 [u r]) alpha))
  (return (Pr (= speaker.u u))))

(defmemo S1-t [[u U] [r R]] [alpha]
  (speaker :knows r)
  (speaker :chooses [u U] :wpp (** (L0 [u r]) alpha))
  (return (Pr (= speaker.u u))))

(defmemo L1 [[u U] [r R]] [alpha]
  (listener :thinks
    (speaker :given [r R] :wpp 1)
    (speaker :chooses [u U] :wpp (S1-pow [r u] alpha)))
  (listener :observes speaker.u :is u)
  (listener :chooses [r R] :wpp (Pr (= speaker.r r)))
  (return (Pr (= listener.r r))))

;; an observation that is impossible: memo's silent all-zero table
(defmemo impossible [[a Opt]] []
  (alice :knows a)
  (alice :thinks (bob :chooses [o Opt] :wpp 1))
  (alice :observes-that (= bob.o 7))
  (return (as alice (Pr (= bob.o a)))))

(defmemo two [[a Opt]] [beta]
  (agent :chooses [o Opt] :wpp (exp (* beta o)))
  (return (Pr (= agent.o a)))
  (return (* 2 (Pr (= agent.o a)))))

(defmemo nan-table [[a Opt]] []
  (agent :chooses [o Opt] :wpp 1)
  (return (/ (- (Pr (= agent.o a)) (Pr (= agent.o a))) (- (Pr (= agent.o a)) (Pr (= agent.o a))))))

(defmemo negative [[a Opt]] []
  (agent :chooses [o Opt] :wpp 1)
  (return (- (Pr (= agent.o a)) 0.5)))

(defmemo no-axes [] [beta]
  (agent :chooses [o Opt] :wpp (exp (* beta o)))
  (return (E agent.o)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- num
  "A host number or vector of them, integers as longs (backends differ in
  the dtype of a draw)."
  [x]
  (cond (sequential? x) (mapv num x)
        (== x (Math/rint (double x))) (long x)
        :else (double x)))

(defn- memo-message
  "The message of the memo error thunk throws, or nil."
  [thunk]
  (try (thunk) nil
       (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(defn- check-dist
  "d against table, its probabilities over support in order: the support,
  log-prob against the log of the table, normalization over the support,
  and 4000 draws, all in the support and each value's frequency within five
  standard errors of its probability."
  [d table support]
  (is (= (num support) (num (dc/support d))))
  (let [lps (mapv #(host (dc/log-prob d %)) support)]
    (is (close? (mapv #(Math/log %) table) lps) "log-prob is the log of the table")
    (is (close? 1.0 (reduce + (map #(Math/exp %) lps))) "normalized over the support"))
  (let [n 4000
        counts (frequencies (num (host (dc/sample d (rng/key 7) [n]))))]
    (is (every? (set (num support)) (keys counts)) "draws are values of the domain")
    (doseq [[s p] (map vector (num support) table)]
      (let [f (/ (double (get counts s 0)) n)
            se (Math/sqrt (/ (* p (- 1 p)) n))]
        (is (<= (Math/abs (- f p)) (+ (* 5 se) 1e-12)) (str "the frequency of " s))))))

(defn- fd-close?
  "A gradient against central finite differences of f at x (h = 1e-3, to
  float32's precision)."
  [g f x]
  (let [h 1e-3
        fd (/ (- (double (host (f (+ x h)))) (double (host (f (- x h))))) (* 2 h))]
    (< (Math/abs (- (double (host g)) fd)) (+ 1e-3 (* 1e-3 (Math/abs fd))))))

;; ---------------------------------------------------------------------------
;; Distributions against their tables
;; ---------------------------------------------------------------------------

(deftest distributions-are-their-tables
  (testing "GenGDT's mind, and its zero cells at q = 0"
    (check-dist ((mg/dist choose) 1.3 0.4) (host (choose 1.3 0.4)) [0 1 2])
    (check-dist ((mg/dist choose) 1.3 0.0) [1.0 0.0 0.0] [0 1 2]))
  (testing "draws are domain values, not indices"
    (check-dist ((mg/dist pickx) 0.7) (host (pickx 0.7)) [10 20 30]))
  (testing "two sampled axes: a draw is the vector of their values"
    (check-dist ((mg/dist joint) 0.5) (vec (flatten (host (joint 0.5))))
                (vec (for [a [0 1 2] x [10 20 30]] [a x]))))
  (testing "a given axis: the conditional distribution, its value first"
    (doseq [[i u] (map-indexed vector [1 2 4 8])]
      (check-dist ((mg/dist L1 {:given '[u]}) [u] 1.5) (nth (host (L1 1.5)) i) [5 9 10]))
    (doseq [[i r] (map-indexed vector [5 9 10])]
      (check-dist ((mg/dist S1-pow {:given '[r]}) [r] 2.0) (nth (host (S1-pow 2.0)) i) [1 2 4 8])
      ;; the given axis after the sampled one
      (check-dist ((mg/dist S1-t {:given '[r]}) [r] 2.0) (nth (host (S1-pow 2.0)) i) [1 2 4 8])))
  (testing "parameters by name, and :return"
    (is (close? (host ((mg/deterministic choose) 1.3 0.4))
                (host (:probs (:params ((mg/dist choose) {:beta 1.3 :q 0.4}))))))
    (check-dist ((mg/dist two {:return 0}) 0.9) (first (host (two 0.9))) [0 1 2])))

(deftest a-value-outside-the-domain-is-an-error
  (is (re-find #"not in its domain: 1" (str (memo-message #(dc/log-prob ((mg/dist pickx) 0.7) 1)))))
  (is (re-find #"not in its domain" (str (memo-message #(dc/log-prob ((mg/dist choose) 1.0 0.5) 3)))))
  (is (close? (host (dc/log-prob ((mg/dist joint) 0.5) (array/array [2 30])))
              (Math/log (get-in (host (joint 0.5)) [2 2]))))
  (testing "probability 0 scores -Inf"
    (is (= Double/NEGATIVE_INFINITY (double (host (dc/log-prob ((mg/dist choose) 1.0 0.0) 2)))))))

;; ---------------------------------------------------------------------------
;; Gradients
;; ---------------------------------------------------------------------------

(deftest gradients-reach-the-parameters
  (let [c (mg/dist choose)
        lp-beta (fn [o] (fn [b] (dc/log-prob (c b 0.4) o)))
        lp-q (fn [o] (fn [q] (dc/log-prob (c 1.3 q) o)))]
    (doseq [o [0 1 2]]
      (is (fd-close? ((array/grad (lp-beta o)) 1.3) (lp-beta o) 1.3) (str "beta, o = " o))
      (is (fd-close? ((array/grad (lp-q o)) 0.4) (lp-q o) 0.4) (str "q, o = " o))))
  (let [l1 (mg/dist L1 {:given '[u]})
        lp (fn [a] (dc/log-prob (l1 [1] a) 9))]
    (is (fd-close? ((array/grad lp) 1.5) lp 1.5) "alpha, through two levels of RSA"))
  (testing "through gen.clj's choice-gradients"
    (let [fit (gen [] (let [b (trace :beta (d/uniform 0.5 3.0))]
                        (trace :a ((mg/dist choose) b 0.4))))
          tr (:trace (gfi/generate fit [] (cm/choicemap :beta 1.3 :a 1) (rng/key 1)))
          g (cm/get-choice (:gradients (grad/choice-gradients tr (sel/select :beta))) :beta)]
      (is (fd-close? g (fn [b] (dc/log-prob ((mg/dist choose) b 0.4) 1)) 1.3)))))

(deftest a-zero-cell-has-no-gradient
  ;; d/categorical logs every cell, then picks: a zero cell anywhere made the
  ;; gradient NaN (GenGDT's stand-in). The drawn cell is picked, then logged.
  (testing "the analytic gradient beside a zero cell: d/dbeta log 1/(1 + e^beta)"
    (let [g (host ((array/grad (fn [b] (dc/log-prob ((mg/dist flee) b 0.0) 0))) 1.0))]
      (is (close? (- (/ (Math/exp 1.0) (+ 1.0 (Math/exp 1.0)))) g))))
  (testing "GenGDT's mind at q = 0: PASS is certain, its gradient 0"
    (is (close? 0.0 (host ((array/grad (fn [b] (dc/log-prob ((mg/dist choose) b 0.0) 0))) 1.0))))))

(deftest a-nan-gradient-is-an-error
  (testing "the RSA speaker exp(alpha log L0): right values, NaN gradient"
    (is (Double/isNaN (double (host ((array/grad (fn [a] (array/log (array/take (array/reshape (S1 a) [-1]) 0)))) 1.5)))))
    (is (close? (host (dc/log-prob ((mg/dist S1 {:given '[r]}) [5] 1.5) 1))
                (host (dc/log-prob ((mg/dist S1-pow {:given '[r]}) [5] 1.5) 1))))
    (is (= "NaN gradient with respect to parameter alpha of memo S1"
           (memo-message #((array/grad (fn [a] (dc/log-prob ((mg/dist S1 {:given '[r]}) [5] a) 1))) 1.5)))))
  (testing "the guard passes a finite gradient unchanged"
    (is (close? (host ((array/grad (fn [a] (array/log (array/take (array/reshape (S1-pow a) [-1]) 0)))) 1.5))
                (host ((array/grad (fn [a] (dc/log-prob ((mg/dist S1-pow {:given '[r]}) [5] a) 1))) 1.5)))))
  (testing "through deterministic, and by name"
    (is (= "NaN gradient with respect to parameter alpha of memo S1"
           (memo-message #((array/grad (fn [a] (array/sum ((mg/deterministic S1) {:alpha a})))) 1.5))))))

;; ---------------------------------------------------------------------------
;; The table's checks, and construction
;; ---------------------------------------------------------------------------

(deftest the-table-is-checked
  (is (= "The table of memo impossible does not sum to 1: it sums to 0.0"
         (memo-message #((mg/dist impossible)))))
  (is (re-find #"^The table of memo two does not sum to 1: it sums to (2\.0|1\.99)"
               (str (memo-message #((mg/dist two {:return 1}) 0.9)))))
  (is (= "The table of memo nan-table has NaN" (memo-message #((mg/dist nan-table)))))
  (is (= "The table of memo negative has a negative cell: it is not a distribution"
         (memo-message #((mg/dist negative)))))
  (testing "a given axis names its row"
    ;; L0 is 1/2 at r = 9's two utterances: 0.5 ** 2000 underflows
    (is (= "The table of memo S1-pow does not sum to 1 for r = 9: it sums to 0.0"
           (memo-message #((mg/dist S1-pow {:given '[r]}) [9] 2000.0))))))

(deftest construction-and-calls-are-checked
  (is (re-find #"^dist takes a memo" (str (memo-message #(mg/dist inc)))))
  (is (= "Memo no-axes has no axes: it is not a distribution" (memo-message #(mg/dist no-axes))))
  (is (= "dist: z is not an axis of memo choose" (memo-message #(mg/dist choose {:given '[z]}))))
  (is (= "dist: every axis of memo pickx is given, so none is sampled"
         (memo-message #(mg/dist pickx {:given '[x]}))))
  (is (= "Memo two has 2 returns: name the distribution's with :return" (memo-message #(mg/dist two))))
  (is (= "dist: :return 2 is not a return of memo two, which has 2" (memo-message #(mg/dist two {:return 2}))))
  (is (= "The distribution of memo L1 given [u] takes their values first, as a vector"
         (memo-message #((mg/dist L1 {:given '[u]}) 2 1.5))))
  (is (= "The value 3 of given axis u of memo L1 is not in its domain"
         (memo-message #((mg/dist L1 {:given '[u]}) [3] 1.5))))
  (is (= "deterministic: :return 5 is not a return of memo two, which has 2"
         (memo-message #(mg/deterministic two {:return 5})))))

(deftest deterministic-is-the-table
  (is (close? (host (choose 1.3 0.4)) (host ((mg/deterministic choose) 1.3 0.4))))
  (is (close? (second (host (two 0.9))) (host ((mg/deterministic two {:return 1}) 0.9))))
  (let [m (gen [b] (splice :mind (mg/deterministic choose) b 0.4))
        tr (gfi/simulate m [1.3] (rng/key 1))]
    (is (close? (host (choose 1.3 0.4)) (host (:retval tr))))
    (is (close? 0.0 (host (:score tr))))))

;; ---------------------------------------------------------------------------
;; In gen.clj models: batched runs and the GFI laws
;; ---------------------------------------------------------------------------

;; Vectorized functions of a parameter. weight-if's if takes its first branch
;; for every array; on host numbers, with beta > 0, its second.
(defn ^:memo/vectorized weight [beta u] (array/exp (array/* beta u)))
(defn ^:memo/vectorized weight-if [beta u]
  (if (array/< beta 0.0) (array/* beta u) (array/exp (array/* beta u))))

(defmemo weighted [[a Opt]] [beta]
  (agent :chooses [o Opt] :wpp (weight beta (utility o)))
  (return (Pr (= agent.o a))))

(defmemo weighted-if [[a Opt]] [beta]
  (agent :chooses [o Opt] :wpp (weight-if beta (utility o)))
  (return (Pr (= agent.o a))))

(def choice (mg/dist choose))
(def weighted-choice (mg/dist weighted))
(def pick (mg/dist pickx))
(def listener (mg/dist L1 {:given '[u]}))

(def mind-model
  "A scientist's model of a person: beta unknown, two choices by minds."
  (gen [q]
    (let [beta (trace :beta (d/uniform 0.5 2.0))
          a (trace :a (choice beta q))
          x (trace :x (pick beta))]
      [a x])))

(def weighted-model
  "beta unknown, a choice by a mind computing with a vectorized function of
  it."
  (gen []
    (let [beta (trace :beta (d/uniform 0.5 2.0))]
      (trace :a (weighted-choice beta)))))

(def rsa-model
  "An utterance, then the listener's referent given it."
  (gen [alpha]
    (let [u (trace :u (d/categorical [0.25 0.25 0.25 0.25]))]
      (trace :r (listener [(nth [1 2 4 8] (long (array/item u)))] alpha)))))

(deftest batched-runs-stay-batched
  (let [ps (gb/simulate mind-model [0.4] 200 (rng/key 1))]
    (is (gb/batched? ps))
    (is (nil? (gb/fallback-reason ps))))
  (testing "generate with every choice given: each particle's weight is the scalar one"
    (let [cs (cm/choicemap :beta 1.3 :a 1 :x 20)
          {:keys [particles weights]} (gb/generate mind-model [0.4] cs 50 (rng/key 2))
          w (host (:weight (gfi/generate mind-model [0.4] cs (rng/key 3))))]
      (is (gb/batched? particles))
      (is (every? #(close? w %) (host weights)))))
  (testing "a vectorized function of a batched parameter, spot-checked per particle (memoclj-m2jw)"
    (let [ps (gb/simulate weighted-model [] 100 (rng/key 4))]
      (is (gb/batched? ps))
      (is (nil? (gb/fallback-reason ps))))))

(deftest vectorized-functions-under-the-batch-tracer
  (let [betas [0.6 1.0 1.7]
        under (fn [m] (batch/batched [3] (m (batch/wrap (array/array betas) [3]))))]
    (testing "each particle's table is its own call's"
      (is (close? (mapv #(host (weighted %)) betas) (host (batch/unwrap (under weighted) [3])))))
    ;; the spot-check runs on eager backends: on MLX it would force the lazy
    ;; arrays (DESIGN.md §5)
    (when (array/concrete? (array/array 0.0))
      (testing "a vectorized function taking one branch for a whole array is still found"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"gives other values on whole arrays"
                              (under weighted-if)))))))

(laws/deflaws memo-dist
  [[:mind-model mind-model [0.4] [[0.7] [0.0]]]
   [:rsa-model rsa-model [1.5] [[2.5]]]]
  25)
