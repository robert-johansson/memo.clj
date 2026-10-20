(ns memo.demos.demo-polarization-test
  "memo's demo/demo-polarization.ipynb: two economists read one study and
  move apart."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Opinion_of_Bill BAD_POLICY GOOD_POLICY)                         ; is the bill good?
(defenum Bill_Outcome NO_SPENDING SPENDING_INCREASE)                     ; actual effect of bill
(defenum Study_Result NO_SPENDING SPENDING_INCREASE)                     ; study conclusion
(defenum Optimal_Economic_Policy FISCALLY_CONSERVATIVE FISCALLY_LIBERAL) ; economic philosophy
(defenum Economist ALICE BOB)

(defmemo prior [[e Economist]] []
  (economist :knows e)
  (economist :thinks
    (world :knows e)
    (world :chooses [v1 Optimal_Economic_Policy]
           :wpp (if (= e #{Economist.ALICE})
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_LIBERAL}) 0.9 0.1)
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_LIBERAL}) 0.1 0.9)))
    (world :chooses [v2 Bill_Outcome] :wpp 1)
    (world :chooses [d Study_Result] :wpp (if (= d v2) 0.9 0.1))
    (world :chooses [h Opinion_of_Bill]
           :wpp (if (= v2 #{Bill_Outcome.NO_SPENDING})
                  0.5
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_CONSERVATIVE})
                    (if (= h #{Opinion_of_Bill.GOOD_POLICY}) 0.1 0.9)
                    (if (= h #{Opinion_of_Bill.GOOD_POLICY}) 0.9 0.1)))))
  (return (as economist (Pr (= world.h 1)))))

(defmemo posterior [[e Economist] [d Study_Result]] []
  (economist :knows e)
  (economist :thinks
    (world :knows e)
    (world :chooses [v1 Optimal_Economic_Policy]
           :wpp (if (= e #{Economist.ALICE})
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_LIBERAL}) 0.9 0.1)
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_LIBERAL}) 0.1 0.9)))
    (world :chooses [v2 Bill_Outcome] :wpp 1)
    (world :chooses [d Study_Result] :wpp (if (= d v2) 0.9 0.1))
    (world :chooses [h Opinion_of_Bill]
           :wpp (if (= v2 #{Bill_Outcome.NO_SPENDING})
                  0.5
                  (if (= v1 #{Optimal_Economic_Policy.FISCALLY_CONSERVATIVE})
                    (if (= h #{Opinion_of_Bill.GOOD_POLICY}) 0.1 0.9)
                    (if (= h #{Opinion_of_Bill.GOOD_POLICY}) 0.9 0.1)))))
  (economist :observes world.d :is d)
  (return (as economist (Pr (= world.h 1)))))

(deftest demo-polarization
  (ts/check-demo 'memo.demos.demo-polarization-test "demo-polarization"))
