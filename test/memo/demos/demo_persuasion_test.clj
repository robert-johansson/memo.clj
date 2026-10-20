(ns memo.demos.demo-persuasion-test
  "memo's demo/demo-persuasion.ipynb: Bayesian persuasion (Kamenica &
  Gentzkow): a prosecutor designs a test, over a product domain of two
  accuracies (101 x 101 members), for a judge who reasons about it."
  (:require [clojure.test :refer [deftest is]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defdomain defenum fields]]
            [memo.test-support :as ts]))

(defenum State Innocent Guilty)
(defenum Result Exonerating Incriminating)
(defenum Verdict Acquit Convict)

(def Probability (mapv #(+ 0.5 (* 0.5 (/ % 100.0))) (range 101)))   ; np.linspace(0.5, 1, 101)
(defdomain Test :pi_i (count Probability) :pi_g (count Probability))

(defn run-test [r s t]
  (let [[i g] (fields Test t)
        p-i (nth Probability i)
        p-g (nth Probability g)]
    (nth (nth [[p-i (- 1 p-i)] [(- 1 p-g) p-g]] (long s)) (long r))))

(defmemo prosecutor [[t Test]] [level]
  ;; the prosecutor maximizes the probability that the judge convicts
  (pros :chooses [t Test] :wpp (exp (* 20 (judge [t] level))))
  (return (Pr (= pros.t t))))

(defmemo judge [[t Test]] [level]
  (judge :thinks
    (defendant :chooses [s State] :wpp (if (= s 0) 0.7 0.3))
    (prosecutor :chooses [t Test] :wpp (if (> level 0) (prosecutor [t] (- level 1)) 1))
    (defendant :knows prosecutor.t)
    (defendant :chooses [r Result] :wpp (run-test r s prosecutor.t)))
  (defendant :chooses [s State] :wpp (if (= s 0) 0.7 0.3))
  (defendant :knows t)
  (defendant :chooses [r Result] :wpp (run-test r s t))
  (judge :observes prosecutor.t :is t)
  (judge :observes defendant.r :is defendant.r)
  (judge :chooses [v Verdict] :to-maximize (Pr (= v defendant.s)))
  (return (Pr (= judge.v 1))))

(defn exact-tie?
  "Whether the judge's verdict ties, in exact arithmetic, for test t: some
  result r leaves the judge's posterior on guilt at exactly 1/2. The judge
  observes the test, so the prosecutor's prior over tests drops out, at
  every level. Computed with rationals: Probability is 1/2 + i/200."
  [t]
  (let [[i g] (fields Test t)
        p-i (+ 1/2 (/ i 200))
        p-g (+ 1/2 (/ g 200))]
    (or (= (* 7/10 p-i) (* 3/10 (- 1 p-g)))          ; r = Exonerating
        (= (* 7/10 (- 1 p-i)) (* 3/10 p-g)))))      ; r = Incriminating

(deftest demo-persuasion
  ;; The judge's :to-maximize breaks exact ties uniformly. Whether a tie is
  ;; exact depends on rounding: float32 (Python memo) keeps some that
  ;; float64 breaks, and the other way round (DESIGN.md §2.4). Those cells
  ;; may differ; every other cell must match. The prosecutor's table sums
  ;; over the judge's, so it differs by a little everywhere: it is not
  ;; compared here, but its argmax is.
  (ts/check-demo 'memo.demos.demo-persuasion-test "demo-persuasion"
                 {:ties {:judge (fn [_ t] (exact-tie? t))
                         :prosecutor (constantly true)}})
  (let [fx (ts/fixture "demos/demo-persuasion.edn")
        {[{:keys [args value]}] :calls} (first (filter #(= :prosecutor (:name %)) (:models fx)))
        ours (vec (flatten (ts/host (apply prosecutor args))))
        argmax (fn [v] (apply max-key v (range (count v))))]
    (is (= (argmax value) (argmax ours)) "the prosecutor's best test")
    (is (< (apply max (map #(Math/abs (- %1 %2)) value ours)) 0.005)
        "the prosecutor's table, but for the judge's ties")))
