(ns memo.bounded-test
  "M3: bounded minds (DESIGN.md §7, memo.bounded). Their laws (§9): the
  exact-over-samples lowering against brute-force enumeration of the raw
  samples, written here on the host; the exact core as the limit when the
  budget grows; the Monte Carlo lowering's average against the
  exact-over-samples table (unbiased in exp space); gradients against finite
  differences; the budget's checks and the size limit as loud errors; gen.clj's
  laws for encapsulated randomness over memo.gen/estimated."
  (:require [clojure.java.basis :as basis]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [memo.bounded :as b]
            [memo.core :refer [defmemo]]
            [memo.errors :as err]
            [memo.gen :as mg]
            [memo.test-support :as ts :refer [host]]))

;; gen.laws from the gen.clj in use's test/ (as memo.gen-test loads it)
(let [root (get-in (basis/current-basis) [:libs 'gen.clj/gen.clj :deps/root])]
  (.addURL ^clojure.lang.DynamicClassLoader (clojure.lang.RT/baseLoader)
           (.toURL (.toURI (io/file root "test"))))
  (require '[gen.laws :as laws]))

(alter-var-root #'err/*warnings?* (constantly false))

;; ---------------------------------------------------------------------------
;; Brute force: every sequence of raw samples, on the host
;; ---------------------------------------------------------------------------

(defn- seqs
  "Every sequence of k values from vals."
  [k vals]
  (if (zero? k) [[]] (for [v vals, t (seqs (dec k) vals)] (cons v t))))

(defn- prob [dist xs] (reduce * 1.0 (map dist xs)))

(defn- mean [xs] (/ (reduce + 0.0 xs) (count xs)))

(defn- softmax [logits]
  (let [mx (apply max logits)
        e (map #(Math/exp (- % mx)) logits)
        z (reduce + e)]
    (mapv #(/ % z) e)))

(defn- argmax-uniform
  "The argmax rule as memo's: uniform over the values equal to the maximum."
  [vs]
  (let [mx (apply max vs)
        n (count (filter #(== mx %) vs))]
    (mapv #(if (== mx %) (/ 1.0 n) 0.0) vs)))

(defn- average
  "Σ over the sequences of samples of P(sequence) · (choice sequence), a
  vector of choice probabilities."
  [weighted]
  (reduce (fn [acc [p v]] (mapv + acc (map #(* p %) v))) (repeat (count (second (first weighted))) 0.0) weighted))

(defn- close-all? [expected actual & [tol]]
  (let [tol (or tol 1e-9)]
    (and (= (count (flatten expected)) (count (flatten actual)))
         (every? true? (map (fn [e a] (<= (Math/abs (- (double e) (double a))) (+ tol (* tol (Math/abs (double e))))))
                            (flatten expected) (flatten actual))))))

(defn- budget [spec] {:memo/budget {'alice spec}})

(defn- number [x] (double (if (number? x) x (array/item x))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def Y [0 1 2])
(def A [0 1])
(def S [0 1 2 3])
(def Three [0 1 2])

;; alice's utility of option a when bob's choice is y
(defn utility [a y] (if (= a 0) y (- 2 y)))
(def prior-y {0 (/ 1.0 6) 1 (/ 2.0 6) 2 (/ 3.0 6)})

;; Options judged on what bob did: the belief does not depend on the option,
;; so the options share alice's samples (Vul et al.'s one and done).
(defmemo pick [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

(defn- pick-brute [k beta per-option?]
  (average (if per-option?
             (for [y0 (seqs k Y), y1 (seqs k Y)]
               [(* (prob prior-y y0) (prob prior-y y1))
                (softmax [(* beta (mean (map #(utility 0 %) y0))) (* beta (mean (map #(utility 1 %) y1)))])])
             (for [ys (seqs k Y)]
               [(prob prior-y ys) (softmax (for [a A] (* beta (mean (map #(utility a %) ys)))))]))))

;; Options judged on their outcomes: what alice imagines depends on the option,
;; so each option has samples of its own. A goal through EU.
(defn outcome [o s2] (if (= o 0) ([0.5 0.5 0.0 0.0] s2) ([0.2 0.0 0.0 0.8] s2)))

(defmemo plan [[a A]] [beta]
  (alice :wants g s2)
  (alice :chooses [o A] :wpp (exp (* beta (EU g))))
  (alice :given [s2 S] :wpp (outcome o s2))
  (return (Pr (= alice.o a))))

(defn- plan-brute [k beta]
  (let [p0 {0 0.5 1 0.5 2 0.0 3 0.0} p1 {0 0.2 1 0.0 2 0.0 3 0.8}]
    (average (for [s0 (seqs k S), s1 (seqs k S)]
               [(* (prob p0 s0) (prob p1 s1)) (softmax [(* beta (mean s0)) (* beta (mean s1))])]))))

;; The same, by argmax (:to-maximize), whose ties a few samples make common
(defmemo plan-max [[a A]] []
  (alice :wants g s2)
  (alice :chooses [o A] :to-maximize (EU g))
  (alice :given [s2 S] :wpp (outcome o s2))
  (return (Pr (= alice.o a))))

(defn- plan-max-brute [k]
  (let [p0 {0 0.5 1 0.5 2 0.0 3 0.0} p1 {0 0.2 1 0.0 2 0.0 3 0.8}]
    (average (for [s0 (seqs k S), s1 (seqs k S)]
               [(* (prob p0 s0) (prob p1 s1)) (argmax-uniform [(mean s0) (mean s1)])]))))

;; Nested queries: what alice's future self expects of the world depends on
;; its outcome s2, which alice's EU averages over, so the future self's
;; samples are drawn afresh for each value of s2 (shared across the options,
;; since its belief about the world does not depend on them)
(defn outcome2 [o s2] (if (= o 0) ([0.7 0.3] s2) ([0.2 0.8] s2)))
(defn world-given [s2 w] (if (= s2 0) ([0.9 0.1] w) ([0.3 0.7] w)))
(defn value2 [w] ([0.0 1.0] w))

(defmemo plan2 [[a A]] [beta]
  (alice :wants g (value2 world.w))
  (alice :chooses [o A] :wpp (exp (* beta (EU g))))
  (alice :given [s2 A] :wpp (outcome2 o s2))
  (alice :thinks (world :knows s2) (world :chooses [w A] :wpp (world-given s2 w)))
  (return (Pr (= alice.o a))))

(defn- plan2-brute [k beta]
  (let [ps2 (fn [o] {0 (outcome2 o 0) 1 (outcome2 o 1)})
        pw (fn [s2] {0 (world-given s2 0) 1 (world-given s2 1)})]
    (average (for [w0 (seqs k A), w1 (seqs k A), s0 (seqs k A), s1 (seqs k A)]
               (let [inner {0 (mean (map value2 w0)) 1 (mean (map value2 w1))}]
                 [(* (prob (pw 0) w0) (prob (pw 1) w1) (prob (ps2 0) s0) (prob (ps2 1) s1))
                  (softmax [(* beta (mean (map inner s0))) (* beta (mean (map inner s1)))])])))))

;; a choice over two dims: each of its four cells has samples of its own
(defn outcome-two [o p s2] (get-in [[[0.5 0.5 0.0 0.0] [0.0 0.0 1.0 0.0]] [[0.3 0.0 0.0 0.7] [0.1 0.4 0.4 0.1]]] [o p s2]))

(defmemo plan-two [[a A] [c A]] [beta]
  (alice :wants g s2)
  (alice :chooses [o A] [p A] :wpp (exp (* beta (EU g))))
  (alice :given [s2 S] :wpp (outcome-two o p s2))
  (return (Pr (= alice.o a) (= alice.p c))))

(defn- plan-two-brute [k beta]
  (let [cells [[0 0] [0 1] [1 0] [1 1]]
        dist (fn [[o p]] (into {} (for [s S] [s (outcome-two o p s)])))
        flat (average (for [s00 (seqs k S), s01 (seqs k S), s10 (seqs k S), s11 (seqs k S)]
                        (let [ss [s00 s01 s10 s11]]
                          [(reduce * 1.0 (map #(prob (dist %1) %2) cells ss))
                           (softmax (map #(* beta (mean %)) ss))])))]
    [[(flat 0) (flat 1)] [(flat 2) (flat 3)]]))

;; options considered over a choice's joint cells
(defn weight-two [o p] (get-in [[1.0 2.0] [3.0 4.0]] [o p]))

(defmemo two-options [[a A] [c A]] []
  (alice :chooses [o A] [p A] :wpp (weight-two o p))
  (return (Pr (= alice.o a) (= alice.p c))))

(defn- two-options-brute [k]
  (let [cells [[0 0] [0 1] [1 0] [1 1]]
        subsets (filter #(= k (count %)) (map (fn [m] (filterv #(bit-test m %) (range 4))) (range 16)))
        flat (average (for [s subsets]
                        [(/ 1.0 (count subsets))
                         (let [z (reduce + (map #(apply weight-two (cells %)) s))]
                           (mapv (fn [i] (if (some #{i} s) (/ (apply weight-two (cells i)) z) 0.0)) (range 4)))]))]
    [[(flat 0) (flat 1)] [(flat 2) (flat 3)]]))

;; alice observes that bob's y is not 0, then chooses
(defmemo seen [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :observes-that (> bob.y 0))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

(defn- seen-brute [k beta algorithm]
  (if (= algorithm :rejection)
    ;; from the prior, kept when y > 0; with none kept, no mass: estimates 0
    (average (for [ys (seqs k Y)]
               (let [kept (filter pos? ys)]
                 [(prob prior-y ys)
                  (softmax (for [a A] (if (seq kept) (* beta (mean (map #(utility a %) kept))) 0.0)))])))
    ;; from the posterior
    (let [post {1 0.4 2 0.6}]
      (average (for [ys (seqs k [1 2])]
                 [(prob post ys) (softmax (for [a A] (* beta (mean (map #(utility a %) ys)))))])))))

;; soft evidence: rejection keeps a sample with the probability of what alice
;; observed
(defn soft-weight [y] ([0.2 0.5 0.9] y))

(defmemo soft-seen [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :observes-event :wpp (soft-weight bob.y))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

(defn- soft-seen-brute [k beta algorithm]
  (if (= algorithm :rejection)
    (average (for [ys (seqs k Y), keep (seqs k [true false])]
               (let [kept (keep-indexed #(when (nth keep %1) %2) ys)]
                 [(* (prob prior-y ys) (reduce * 1.0 (map #(if %2 (soft-weight %1) (- 1 (soft-weight %1))) ys keep)))
                  (softmax (for [a A] (if (seq kept) (* beta (mean (map #(utility a %) kept))) 0.0)))])))
    (let [z (reduce + (map #(* (prior-y %) (soft-weight %)) Y))
          post (into {} (for [y Y] [y (/ (* (prior-y y) (soft-weight y)) z)]))]
      (average (for [ys (seqs k Y)]
                 [(prob post ys) (softmax (for [a A] (* beta (mean (map #(utility a %) ys)))))])))))

;; alice observes bob's x (a choice): rejection keeps the samples whose x is
;; the observed one
(def X [0 1 2])
(defn x-given-t [t x] (get-in [[0.6 0.3 0.1] [0.1 0.3 0.6]] [t x]))
(defn reward [a t] (if (= a t) 1.0 0.0))

(defmemo watched [[a A] [xo X]] [beta]
  (alice :thinks (bob :chooses [t A] :wpp (+ 1 t)) (bob :chooses [x X] :wpp (x-given-t t x)))
  (alice :observes bob.x :is xo)
  (alice :chooses [a A] :wpp (exp (* beta (E (reward a bob.t)))))
  (return (Pr (= alice.a a))))

(defn- watched-brute [k beta algorithm]
  (let [pt {0 (/ 1.0 3) 1 (/ 2.0 3)}
        joint (into {} (for [t A, x X] [[t x] (* (pt t) (x-given-t t x))]))]
    (vec (for [xo X]
           (if (= algorithm :rejection)
             (average (for [s (seqs k (keys joint))]
                        (let [kept (map first (filter #(= xo (second %)) s))]
                          [(prob joint s) (softmax (for [a A] (if (seq kept) (* beta (mean (map #(reward a %) kept))) 0.0)))])))
             (let [z (reduce + (map #(joint [% xo]) A))
                   post (into {} (for [t A] [t (/ (joint [t xo]) z)]))]
               (average (for [ts (seqs k A)]
                          [(prob post ts) (softmax (for [a A] (* beta (mean (map #(reward a %) ts)))))]))))))))

;; options considered
(defn weight3 [o] ([1.0 2.0 4.0] o))

(defmemo three [[a Three]] []
  (alice :chooses [o Three] :wpp (weight3 o))
  (return (Pr (= alice.o a))))

(defmemo three-max [[a Three]] []
  (alice :chooses [o Three] :to-maximize (weight3 o))
  (return (Pr (= alice.o a))))

(defmemo three-given [[a Three]] []
  (alice :given [o Three] :wpp (weight3 o))
  (return (Pr (= alice.o a))))

(defn- three-brute [k rule]
  (let [subsets (filter #(= k (count %)) (map (fn [m] (filterv #(bit-test m %) [0 1 2])) (range 8)))]
    (average (for [s subsets]
               [(/ 1.0 (count subsets))
                (let [chosen (rule (mapv weight3 s))]
                  (mapv (fn [o] (if-let [i (first (keep-indexed #(when (= o %2) %1) s))] (chosen i) 0.0)) [0 1 2]))]))))

(defn- by-weight [ws] (let [z (reduce + ws)] (mapv #(/ % z) ws)))

;; options considered and samples together
(defn utility3 [o y] (get-in [[0 1 2] [2 1 0] [1 1 1]] [o y]))

(defmemo three-sampled [[a Three]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [o Three] :wpp (exp (* beta (E (utility3 o bob.y)))))
  (return (Pr (= alice.o a))))

(defn- three-sampled-brute [k beta]
  (average (for [s [[0 1] [0 2] [1 2]], ys (seqs k Y)]
             [(/ (prob prior-y ys) 3.0)
              (let [p (softmax (for [o s] (* beta (mean (map #(utility3 o %) ys)))))]
                (mapv (fn [o] (if-let [i (first (keep-indexed #(when (= o %2) %1) s))] (p i) 0.0)) [0 1 2]))])))

;; a bounded agent as another agent models it
(defmemo nested [[a A]] [beta]
  (bob :thinks
    (alice :thinks (carol :chooses [y Y] :wpp (+ 1 y)))
    (alice :chooses [a A] :wpp (exp (* beta (E (utility a carol.y))))))
  (bob :chooses [b A] :wpp (Pr (= alice.a b)))
  (return (Pr (= bob.b a))))

;; an observer inferring a bounded agent's type from its choice: the samples
;; are marginalized before the observer's posterior is normalized
(defmemo infer [[t A] [a A]] [beta]
  (sci :knows t)
  (sci :thinks
    (alice :chooses [ty A] :uniformly)
    (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
    (alice :chooses [a A] :wpp (exp (* (+ beta (* 2 ty)) (E (utility a bob.y))))))
  (sci :observes alice.a :is a)
  (return (as sci (Pr (= alice.ty t)))))

;; queries other than E, each as a plug-in estimate
(defmemo var-pick [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [a A] :wpp (exp (* beta a (Var bob.y))))
  (return (Pr (= alice.a a))))

(defmemo entropy-pick [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [a A] :wpp (exp (* beta a (H bob.y))))
  (return (Pr (= alice.a a))))

(defmemo posterior-pick [[a A]] [beta]
  (alice :thinks (bob :chooses [z A] :wpp (+ 1 (* 2 z))))
  (alice :chooses [a A] :wpp (exp (* beta (Pr (= a bob.z)))))
  (return (Pr (= alice.a a))))

(defmemo kl-pick [[a A]] [beta]
  (alice :thinks (bob :chooses [x A] :wpp (+ 1 x)) (bob :chooses [w A] :wpp (- 3 w)))
  (alice :chooses [a A] :wpp (exp (* beta a (KL bob.x bob.w))))
  (return (Pr (= alice.a a))))

(defn- plug-in-variance [ys] (let [m (mean ys)] (mean (map #(* (- % m) (- % m)) ys))))
(defn- plug-in-entropy [ys]
  (let [n (double (count ys))] (- (reduce + (map (fn [[_ c]] (* (/ c n) (Math/log (/ c n)))) (frequencies ys))))))
(defn- nan-to-num [x] (cond (Double/isNaN x) 0.0 (Double/isInfinite x) (if (pos? x) Double/MAX_VALUE (- Double/MAX_VALUE)) :else x))
(defn- plug-in-kl [xs ws]
  (let [n (double (count xs))
        f (fn [vs v] (/ (count (filter #(= v %) vs)) n))]
    (reduce + (for [v A] (* (f xs v) (nan-to-num (- (Math/log (f xs v)) (Math/log (f ws v)))))))))

;; a memo calling a memo whose agent is bounded: the budget is passed on
(defmemo calls-pick [[a A]] [beta]
  (return (pick [a] beta)))

(defmemo ^:cache cached-pick [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

;; ---------------------------------------------------------------------------
;; Exact over samples against brute force
;; ---------------------------------------------------------------------------

(deftest exact-over-samples-is-brute-force
  (testing "options sharing samples (the belief does not depend on the option)"
    (doseq [k [1 2 3]]
      (is (close-all? (pick-brute k 1.3 false) (host (pick 1.3 (budget {:samples k})))) (str "k " k))))
  (testing "every option its own samples, with :per-option"
    (doseq [k [1 2]]
      (is (close-all? (pick-brute k 1.3 true) (host (pick 1.3 (budget {:samples k :per-option true})))) (str "k " k))))
  (testing "options with samples of their own (the outcomes depend on the option), through EU"
    (doseq [k [1 2 3]]
      (is (close-all? (plan-brute k 0.7) (host (plan 0.7 (budget {:samples k})))) (str "k " k))))
  (testing "a query inside a query: the inner samples afresh for each value the outer averages over"
    (doseq [k [1 2]]
      (is (close-all? (plan2-brute k 0.9) (host (plan2 0.9 (budget {:samples k})))) (str "k " k))))
  (testing "a choice over two dims: each cell its own samples"
    (doseq [k [1 2]]
      (is (close-all? (plan-two-brute k 0.8) (host (plan-two 0.8 (budget {:samples k})))) (str "k " k))))
  (testing "argmax, with ties"
    (doseq [k [1 2 3]]
      (is (close-all? (plan-max-brute k) (host (plan-max (budget {:samples k})))) (str "k " k))))
  (testing "an observation: one and done samples the posterior, rejection the prior"
    (doseq [k [1 2 3], algorithm [:one-and-done :rejection]]
      (is (close-all? (seen-brute k 1.1 algorithm) (host (seen 1.1 (budget {:samples k :algorithm algorithm}))))
          (str "k " k " " algorithm))))
  (testing "soft evidence"
    (doseq [k [1 2 3], algorithm [:one-and-done :rejection]]
      (is (close-all? (soft-seen-brute k 1.1 algorithm) (host (soft-seen 1.1 (budget {:samples k :algorithm algorithm}))))
          (str "k " k " " algorithm))))
  (testing "an observed choice (the beliefs move to the observed value's dim)"
    (doseq [k [1 2], algorithm [:one-and-done :rejection]]
      ;; the table is over [a xo]; brute force gives, per xo, the vector over a
      (is (close-all? (apply mapv vector (watched-brute k 2.0 algorithm))
                      (host (watched 2.0 (budget {:samples k :algorithm algorithm}))))
          (str "k " k " " algorithm))))
  (testing "options considered"
    (doseq [k [1 2]]
      (is (close-all? (three-brute k by-weight) (host (three (budget {:options k}))))
          (str "wpp, k " k))
      (is (close-all? (three-brute k argmax-uniform) (host (three-max (budget {:options k}))))
          (str "argmax, k " k)))
    (testing "over a choice's joint cells"
      (doseq [k [1 2 3]]
        (is (close-all? (two-options-brute k) (host (two-options (budget {:options k})))) (str "k " k))))
    (testing "with samples"
      (doseq [k [1 2]]
        (is (close-all? (three-sampled-brute k 1.2) (host (three-sampled 1.2 (budget {:samples k :options 2}))))
            (str "k " k))))
    (testing "every option considered is the exact choice"
      (is (= (host (three)) (host (three (budget {:options 3}))))))
    (testing "a :given is not a deliberate choice: all its options count"
      (is (= (host (three-given)) (host (three-given (budget {:options 1})))))))
  (testing "as another agent models it"
    (is (close-all? (pick-brute 2 1.0 false) (host (nested 1.0 (budget {:samples 2}))))))
  (testing "an observer's inference marginalizes the samples first"
    (let [pa (fn [beta a] ((pick-brute 1 beta false) a))]
      (is (close-all? (vec (for [t A] (vec (for [a A] (/ (pa (+ 1.0 (* 2 t)) a) (+ (pa 1.0 a) (pa 3.0 a)))))))
                      (host (infer 1.0 (budget {:samples 1})))))))
  (testing "Var, H, the posterior shortcut and KL as plug-in estimates"
    (doseq [k [1 2 3]]
      (is (close-all? (average (for [ys (seqs k Y)] [(prob prior-y ys) (softmax [0.0 (plug-in-variance ys)])]))
                      (host (var-pick 1.0 (budget {:samples k})))) (str "Var, k " k))
      (is (close-all? (average (for [ys (seqs k Y)] [(prob prior-y ys) (softmax [0.0 (plug-in-entropy ys)])]))
                      (host (entropy-pick 1.0 (budget {:samples k})))) (str "H, k " k))
      (is (close-all? (average (for [zs (seqs k A)]
                                 (let [f (fn [v] (/ (count (filter #(= v %) zs)) (double k)))]
                                   [(prob {0 0.25 1 0.75} zs) (softmax [(* 2.0 (f 0)) (* 2.0 (f 1))])])))
                      (host (posterior-pick 2.0 (budget {:samples k})))) (str "posterior, k " k)))
    (doseq [k [1 2]]
      (let [joint (into {} (for [x A, w A] [[x w] (* ({0 (/ 1.0 3) 1 (/ 2.0 3)} x) ({0 0.6 1 0.4} w))]))]
        (is (close-all? (average (for [s (seqs k (keys joint))]
                                   [(prob joint s) (softmax [0.0 (* 0.001 (plug-in-kl (map first s) (map second s)))])]))
                        (host (kl-pick 0.001 (budget {:samples k}))))
            (str "KL, k " k))))))

;; ---------------------------------------------------------------------------
;; The exact limit
;; ---------------------------------------------------------------------------

(defn- max-error [x y] (apply max (map #(Math/abs (- (double %1) (double %2))) (flatten x) (flatten y))))

(deftest the-exact-core-is-the-limit
  (doseq [[label f ks spec] [["pick" #(pick 1.3 %) [1 4 16 64] {}]
                             ["plan, per option" #(plan 0.7 %) [1 2 4 8 16] {}]
                             ["seen, rejection" #(seen 1.1 %) [1 4 16 64] {:algorithm :rejection}]]]
    (testing label
      (let [exact (host (f {}))
            errors (mapv #(max-error exact (host (f (budget (assoc spec :samples %))))) ks)]
        (is (apply > errors) (str "errors " errors))
        (is (< (peek errors) (/ (first errors) 8)) (str "errors " errors))))))

;; ---------------------------------------------------------------------------
;; The budget, the limit, the report
;; ---------------------------------------------------------------------------

(defn- message [thunk]
  (try (thunk) nil (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(deftest a-budget-is-checked
  (doseq [[budget-map msg] [[{'carol {:samples 2}} "The budget names carol, who is not an agent of memo pick"]
                            [{'observer {:samples 2}} "The observer cannot be given a budget"]
                            [{'self {:samples 2}} "self cannot be given a budget"]
                            [{'alice {:sample 2}} "The budget of alice has an unknown key :sample"]
                            [{'alice {}} "The budget of alice bounds nothing"]
                            [{'alice {:samples 0}} "The budget of alice has :samples 0, not a positive integer"]
                            [{'alice {:samples 2 :algorithm :lw}} "The budget of alice has :algorithm :lw, not one of :one-and-done, :rejection"]
                            [{'alice {:options 2 :algorithm :rejection}} "The budget of alice has an :algorithm but no :samples"]
                            [{'alice {:samples 2 :draws 3}} "The budget of alice has :draws without :monte-carlo true"]
                            [{'alice {:samples 2 :per-option 1}} "The budget of alice has :per-option 1, not true or false"]
                            [{'alice {:samples 2 :limit -1}} "The :limit of alice's budget is -1, not a positive whole number"]
                            [{'alice 8} "The budget of alice is 8, not a map"]
                            [{"alice" {:samples 2}} "A budget's agent is \"alice\", not an agent's name"]
                            [[:alice] "The budget is [:alice], not a map"]]]
    (is (= msg (message #(pick 1.0 {:memo/budget budget-map}))) (pr-str budget-map)))
  (testing "a keyword names an agent too"
    (is (= (host (pick 1.0 (budget {:samples 2}))) (host (pick 1.0 {:memo/budget {:alice {:samples 2}}})))))
  (testing "the call's options"
    (is (= "Unknown call option :memo/budgets" (message #(pick 1.0 {:memo/budgets {}}))))
    (is (= ":memo/key is 0, not a key" (message #(pick 1.0 {:memo/key 0}))))
    (is (= "A Monte Carlo budget needs a key: call with :memo/key"
           (message #(pick 1.0 (budget {:samples 2 :monte-carlo true}))))))
  (testing "an empty budget is the exact memo"
    (is (= (host (pick 1.0)) (host (pick 1.0 {:memo/budget {}}))))))

(defmemo snap [[a A]] []
  (alice :snapshots-self-as past)
  (alice :chooses [a A] :uniformly)
  (return (Pr (= alice.a a))))

(deftest a-budget-names-agents-of-the-memo-or-its-callees
  (is (= "The budget names past, alice's snapshot" (message #(snap {:memo/budget {'past {:samples 1}}}))))
  (testing "an agent of a memo called"
    (is (close-all? (host (pick 1.0 (budget {:samples 2}))) (host (calls-pick 1.0 (budget {:samples 2})))))))

(defmemo told [] []
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (return (as alice (E bob.y))))

(defmemo soft [] []
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :observes-event :wpp (E bob.y))
  (return 0))

(defmemo other [[b A]] []
  (carol :thinks (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y))))
  (carol :chooses [b A] :wpp (exp (* b (as alice (E bob.y)))))
  (return (Pr (= carol.b b))))

(deftest a-bounded-estimate-drives-its-agents-choice
  (doseq [[label thunk] [["returned" #(told (budget {:samples 2}))]
                         ["weighting an observation" #(soft (budget {:samples 2}))]
                         ["read by another agent's choice" #(other (budget {:samples 2}))]]]
    (testing label
      (is (= "alice's bounded estimate is used outside alice's own choice" (message thunk)))))
  (testing "exact, the same models run"
    (is (some? (told)))
    (is (some? (soft)))
    (is (some? (other)))))

(deftest the-size-limit
  (testing "above the limit, a loud error naming the size"
    (is (= "alice's E from 40 samples enumerates 152300281 states, above the limit of 1000000"
           (message #(plan 1.0 (budget {:samples 40}))))))
  (testing "the agent's :limit, and the call's :memo/limit"
    (is (= "alice's E from 5 samples enumerates 3136 states, above the limit of 3000"
           (message #(plan 1.0 (budget {:samples 5 :limit 3000})))))
    (is (= "alice's E from 5 samples enumerates 3136 states, above the limit of 10"
           (message #(plan 1.0 (assoc (budget {:samples 5}) :memo/limit 10)))))
    (is (some? (plan 1.0 (assoc (budget {:samples 5}) :memo/limit 1e4))))
    (is (some? (plan 1.0 (budget {:samples 5 :limit 3136})))))
  (testing "options considered"
    (is (= "alice considering 1 of 3 options enumerates 3 states, above the limit of 2"
           (message #(three (budget {:options 1 :limit 2}))))))
  (testing "the report"
    (let [no-lines (fn [r] (mapv #(-> % (dissoc :line) (update :sources (partial mapv (fn [s] (dissoc s :line))))) r))]
      (is (= [{:agent 'alice :states 400 :limit 1000000 :monte-carlo false
               :sources [{:kind :samples :states 400 :what "E" :k 3 :categories 4 :sets 2}]}]
             (no-lines (b/report plan {'alice {:samples 3}}))))
      (is (= [{:agent 'alice :states 3 :limit 1000000 :monte-carlo false
               :sources [{:kind :options :states 3 :k 2 :options 3}]}]
             (no-lines (b/report three {'alice {:options 2}}))))
      (is (every? integer? (map :line (b/report plan {'alice {:samples 3}})))))))

(deftest a-budget-reaches-memos-called-and-cached-ones
  (is (close-all? (host (pick 1.0 (budget {:samples 2}))) (host (calls-pick 1.0 (budget {:samples 2})))))
  (testing "a ^:cache memo caches by budget"
    (let [exact (host (cached-pick 1.0))]
      (is (close-all? (pick-brute 1 1.0 false) (host (cached-pick 1.0 (budget {:samples 1})))))
      (is (= exact (host (cached-pick 1.0)))))))

;; ---------------------------------------------------------------------------
;; Gradients
;; ---------------------------------------------------------------------------

(defn ^:memo/vectorized weight-y [y q] (array/where (array/== y 0) q 1.0))

;; at q = 0, bob's y = 0 has probability 0, and its derivative in q is not 0
(defmemo graded [[a A]] [q beta]
  (alice :thinks (bob :chooses [y Y] :wpp (weight-y y q)))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

(deftest gradients-agree-with-finite-differences
  (doseq [k [1 3], q0 [0.0 0.7]]
    (let [opts (budget {:samples k})
          f (fn [q beta] (number (array/take (graded q beta opts) 0 0)))
          dq (number ((array/grad (fn [q] (array/take (graded q 1.0 opts) 0 0))) (array/array q0)))
          db (number ((array/grad (fn [beta] (array/take (graded q0 beta opts) 0 0))) (array/array 1.0)))
          h 1e-6]
      (testing (str "k " k " q " q0)
        (is (close-all? [(/ (- (f (+ q0 h) 1.0) (f q0 1.0)) h)] [dq] 1e-5) "in q, the belief sampled")
        (is (close-all? [(/ (- (f q0 (+ 1.0 h)) (f q0 (- 1.0 h))) (* 2 h))] [db] 1e-5) "in beta")))))

;; ---------------------------------------------------------------------------
;; Monte Carlo
;; ---------------------------------------------------------------------------

(defn- mc-mean
  "The mean and standard error of f's first cell over n keys."
  [f n]
  (let [xs (mapv #(first (host (f (rng/key %)))) (range n))
        m (mean xs)
        sd (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n)))]
    [m (/ sd (Math/sqrt n))]))

(deftest monte-carlo-averages-to-the-exact-over-samples-table
  (doseq [[label f exact] [["shared samples" #(pick 1.3 (assoc (budget {:samples 2 :monte-carlo true :draws 3}) :memo/key %))
                            (pick 1.3 (budget {:samples 2}))]
                           ["per option, through EU" #(plan 0.7 (assoc (budget {:samples 3 :monte-carlo true :draws 2}) :memo/key %))
                            (plan 0.7 (budget {:samples 3}))]
                           ["rejection" #(seen 1.1 (assoc (budget {:samples 2 :algorithm :rejection :monte-carlo true}) :memo/key %))
                            (seen 1.1 (budget {:samples 2 :algorithm :rejection}))]
                           ["options considered" #(three (assoc (budget {:options 2 :monte-carlo true :draws 2}) :memo/key %))
                            (three (budget {:options 2}))]]]
    (testing label
      (let [[m se] (mc-mean f 1500)
            e (first (host exact))]
        (is (< (Math/abs (- m e)) (* 5 se)) (str m " ± " se " against " e))
        (is (pos? se) "the draws differ")))))

(deftest monte-carlo-keys
  (let [opts (budget {:samples 2 :monte-carlo true :draws 2})]
    (testing "a key gives its draws"
      (is (= (host (pick 1.3 (assoc opts :memo/key (rng/key 4)))) (host (pick 1.3 (assoc opts :memo/key (rng/key 4)))))))
    (testing "another key others (the draws are never folded into constants)"
      (is (not= (host (pick 1.3 (assoc opts :memo/key (rng/key 4)))) (host (pick 1.3 (assoc opts :memo/key (rng/key 5)))))))
    (testing "the belief sampled cannot be differentiated"
      (is (= "Monte Carlo samples cannot be differentiated through the agent's belief"
             (message #((array/grad (fn [q] (array/take (graded q 1.0 (assoc opts :memo/key (rng/key 0))) 0 0)))
                        (array/array 0.5))))))))

;; ---------------------------------------------------------------------------
;; The bridge: dist, estimated
;; ---------------------------------------------------------------------------

(deftest the-bridge
  (testing "dist takes an exact-over-samples budget"
    (is (close-all? [(Math/log (first (pick-brute 2 1.0 false)))]
                    [(number (dc/log-prob ((mg/dist pick) 1.0 (budget {:samples 2})) 0))])))
  (testing "but not a Monte Carlo one"
    (is (= "The budget of this call to memo pick declares Monte Carlo, so its table is an estimate"
           (message #((mg/dist pick) 1.0 (budget {:samples 2 :monte-carlo true}))))))
  (let [est (mg/estimated pick)
        args [1.3 (budget {:samples 2 :monte-carlo true :draws 2})]
        exact (host (pick 1.3 (budget {:samples 2})))]
    (testing "estimated: E[ξ] is the exact-over-samples cell"
      (let [n 1500
            xs (mapv #(Math/exp (number (:weight (gfi/assess est args (cm/choicemap :value 1) (rng/key %))))) (range n))
            m (mean xs)
            se (/ (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))) (Math/sqrt n))]
        (is (< (Math/abs (- m (second exact))) (* 5 se)) (str m " ± " se " against " (second exact)))))
    (testing "the joint sampler: E[[value = v] / ξ] = 1 (thesis Eq. 4.4)"
      (let [n 3000
            trs (mapv #(gfi/simulate est args (rng/key (+ 50000 %))) (range n))]
        (doseq [v [0.0 1.0]]
          (let [xs (mapv #(if (== v (number (cm/get-choice (:choices %) :value))) (Math/exp (- (number (:score %)))) 0.0) trs)
                m (mean xs)
                se (/ (Math/sqrt (/ (reduce + (map (fn [x] (* (- x m) (- x m))) xs)) (dec n))) (Math/sqrt n))]
            (is (< (Math/abs (- m 1.0)) (* 5 se)) (str "value " v ": " m " ± " se))))))
    (testing "simulate draws values in the domain, weighed by their estimate"
      (let [tr (gfi/simulate est args (rng/key 1))]
        (is (contains? #{0.0 1.0} (number (cm/get-choice (:choices tr) :value))))
        (is (neg? (number (:score tr))))))
    (testing "the per-participant variance of the summed log-estimate"
      (let [r (mg/log-estimate-variance est (repeat 20 [args 0]) {:draws 50 :key (rng/key 9)})]
        (is (= 50 (:draws r)))
        (is (pos? (:sd r)))
        (is (< (:sd (mg/log-estimate-variance est (repeat 20 [[1.3 (budget {:samples 2 :monte-carlo true :draws 16})] 0])
                                              {:draws 50 :key (rng/key 9)}))
               (:sd r))
            "more draws per trial, less variance")))
    (testing "without a Monte Carlo budget, the density is exact"
      (is (close-all? [(Math/log (first exact))]
                      [(number (:weight (gfi/assess est [1.3 (budget {:samples 2})] (cm/choicemap :value 0) (rng/key 0))))])))))

(def estimated-pick (mg/estimated pick))

;; one map, so that an update sees the argument unchanged (gen.clj compares
;; a splice's arguments by identity) and keeps ω
(def mc-budget (budget {:samples 2 :monte-carlo true :draws 2}))

(def pm-model
  "beta ~ U(0.5, 3), and alice's choice by the estimated density."
  (gen [] (let [beta (trace :beta (d/uniform 0.5 3.0))]
            (splice :choice estimated-pick beta mc-budget)
            beta)))

;; an estimator has no proposal density (gen.clj's own estimated-laws)
(def estimated-laws (dissoc laws/encapsulated :score-is-assessed-on-the-same-key))

(laws/deflaws bounded-estimated
  [["estimated pick" estimated-pick [1.3 mc-budget] []]
   ["pm-model" pm-model [] []]]
  40 estimated-laws)
