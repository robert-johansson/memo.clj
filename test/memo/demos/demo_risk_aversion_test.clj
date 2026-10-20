(ns memo.demos.demo-risk-aversion-test
  "memo's demo/demo-risk-aversion.ipynb: minimizing value at risk. The demo
  defines model and utility twice; the second ones are model-2 (as the
  fixture names it) and utility-shock here."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Action Safe Bold)

(def Outcome (mapv #(+ -10.0 (* 20.0 (/ % 100.0))) (range 101)))   ; np.linspace(-10, 10, 101)

(defn normpdf [x] (/ (Math/exp (* -0.5 x x)) (Math/sqrt (* 2 Math/PI))))

(defn utility [a o] (+ (nth [5 9] (long a)) (* (nth [3 10] (long a)) o)))

(defn utility-shock [a o shock] (+ (nth [5 9] (long a)) (* (nth [3 10] (long a)) o) (+ -5 (* 10 shock))))

(defmemo model [[a Action]] []
  (terry :chooses [a Action] :to-minimize (imagine
                                           (world :chooses [o Outcome] :wpp (normpdf o))
                                           (- (** (Var (utility a world.o)) 0.5) (E (utility a world.o)))))
  (return (Pr (= terry.a a))))

(defmemo model-2 [[a Action]] []
  (terry :chooses [a Action] :to-minimize (imagine
                                           (world :chooses [o Outcome] :wpp (normpdf o))
                                           (world :chooses [s Outcome] :wpp (normpdf s))
                                           (- (** (Var (utility-shock a world.o world.s)) 0.5)
                                              (E (utility-shock a world.o world.s)))))
  (return (Pr (= terry.a a))))

(deftest demo-risk-aversion
  (ts/check-demo 'memo.demos.demo-risk-aversion-test "demo-risk-aversion"))
