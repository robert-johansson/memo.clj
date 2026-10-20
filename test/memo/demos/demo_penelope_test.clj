(ns memo.demos.demo-penelope-test
  "memo's demo/demo-penelope.ipynb: Penelope's day and night selves plan
  around each other, by mutual recursion over days, with cached models."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum DayAction SH GO)
(defenum NightAction Retire Party)
(defenum NightType Responsible Fun)

(def P (mapv #(/ % 100.0) (range 101)))   ; np.linspace(0, 1, 101)
(def g 1.0) (def b 1.0)
(def e 1.0) (def l 2.0)

(defn ber "jax.scipy.stats.bernoulli.pmf" [k p] (if (== k 1) p (- 1 p)))
(defn one-night-payoff [night-action] (nth [e l] (long night-action)))
(defn one-day-payoff [day-action night-action]
  (nth (nth [[0.0 0.0] [g (- b)]] (long day-action)) (long night-action)))

(defmemo ^:cache day-model [[p P] [da DayAction]] [(n 0)]
  (day :knows p da)
  (day :thinks
    (night :knows p)
    (night :given [t NightType] :wpp (ber t p))
    (night :chooses [na NightAction]
           :wpp (if (= t #{NightType.Responsible})
                  (= na #{NightAction.Retire})
                  (ber na (night-model 0 [p] n)))))
  (day :snapshots-self-as next_day)
  (return (as day (+ (E (one-day-payoff da night.na))
                     (if (<= n 0)
                       0
                       (imagine
                        (next_day :observes night.na :is night.na)
                        (next_day :chooses [p_ P] :to-minimize (abs (- p_ (Pr (= night.t #{NightType.Fun})))))
                        (next_day :chooses [da_ DayAction] :to-maximize (day-model [p_ da_] (- n 1)))
                        (E (as next_day (day-model [p_ da_] (- n 1))))))))))

(defmemo ^:cache night-model [[p P]] [(n 0)]
  (night :knows p)
  (night :wants payoff (+ (one-night-payoff na)
                          (if (<= n 0)
                            0
                            (E (if (= day.da_ #{DayAction.GO}) (night-model 1 [p] (- n 1)) 0)))))
  (night :chooses [sigma P] :to-maximize (EU payoff))
  (night :chooses [na NightAction] :wpp (ber na sigma))
  (night :thinks
    (day :knows p sigma)
    (day :thinks
      (night :knows p sigma)
      (night :given [t NightType] :wpp (ber t p))
      (night :chooses [na NightAction]
             :wpp (if (= t #{NightType.Responsible}) (= na #{NightAction.Retire}) (ber na sigma))))
    (day :observes night.na :is na)
    (day :chooses [p_ P] :to-minimize (abs (- p_ (Pr (= night.t #{NightType.Fun})))))
    (day :chooses [da_ DayAction] :to-maximize (if (<= n 0) 1 (day-model [p_ da_] (- n 1)))))
  (return (E night.sigma))
  (return (E (as night (EU payoff)))))

(deftest demo-penelope
  (ts/check-demo 'memo.demos.demo-penelope-test "demo-penelope"))
