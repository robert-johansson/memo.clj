(ns memo.demos.demo-vizzini-test
  "memo's demo/demo-vizzini.py: the battle of wits, by level of reasoning."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Cups Near_Vizzini Near_Westley)

(defmemo vizzini-pick [[cup Cups]] [level]
  (vizzini :wants survive (not= my_cup westley.poison))
  (vizzini :thinks
    (westley :wants kill (= vizzini.my_cup poison))
    (westley :chooses [poison Cups] :to-maximize (EU kill))
    (westley :thinks
      (vizzini :chooses [my_cup Cups] :wpp (if (> level 0)
                                             (vizzini-pick [my_cup] (- level 1))
                                             (= my_cup #{Cups.Near_Vizzini})))))
  (vizzini :chooses [my_cup Cups] :to-maximize (EU survive))
  (return (Pr (= vizzini.my_cup cup))))

(deftest demo-vizzini
  (ts/check-demo 'memo.demos.demo-vizzini-test "demo-vizzini"))
