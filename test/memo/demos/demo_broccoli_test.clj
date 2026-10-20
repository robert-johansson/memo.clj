(ns memo.demos.demo-broccoli-test
  "memo's demo/demo-broccoli.ipynb: a child who cannot represent false
  beliefs reads an adult's expression."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Treat BROCCOLI GOLDFISH)
(defenum EmotionExpression SMILE SCOWL)

(defmemo main [[broccoli_ee EmotionExpression] [offered_treat Treat]] []
  (child :has-theory :child)
  (child :thinks
    (adult :given [preferred_treat Treat] :uniformly)
    (adult :chooses [broccoli_ee EmotionExpression] :such-that
           (if (= preferred_treat #{Treat.BROCCOLI})
             (= broccoli_ee #{EmotionExpression.SMILE})
             (= broccoli_ee #{EmotionExpression.SCOWL}))))
  (child :observes adult.broccoli_ee :is broccoli_ee)
  (child :chooses [offered_treat Treat] :to-maximize (Pr (= offered_treat adult.preferred_treat)))
  (return (Pr (= child.offered_treat offered_treat))))

(deftest demo-broccoli
  (ts/check-demo 'memo.demos.demo-broccoli-test "demo-broccoli"))
