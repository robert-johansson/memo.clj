(ns memo.demos.demo-sally-anne-test
  "memo's demo/demo-sally-anne.ipynb: the false-belief task. The demo's host
  function do is do-action here (do is a Clojure special form)."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Loc BOX BASKET)
(defenum Action ACT_STAY ACT_MOVE)
(defenum Obs [OBS_NONE -1] [OBS_STAY 0] [OBS_MOVE 1])

(defn do-action [l a] (nth (nth [[0 1] [1 0]] (long a)) (long l)))

(defmemo model [[marble_pos_t0 Loc] [obs Obs] [where_look Loc]] []
  (child :knows marble_pos_t0 obs where_look)
  (child :thinks
    (sally :knows marble_pos_t0)
    (sally :thinks
      (anne :knows marble_pos_t0)
      (anne :chooses [a Action] :wpp (if (= a #{Action.ACT_MOVE}) 0.01 0.99))
      (anne :chooses [marble_pos_t1 Loc] :wpp (= (do-action marble_pos_t0 a) marble_pos_t1))
      (anne :chooses [o Obs] :wpp (if (or (= o #{Obs.OBS_NONE}) (= o a)) 1 0)))
    (sally :observes anne.o :is obs)
    (sally :chooses [where_look Loc] :wpp (Pr (= anne.marble_pos_t1 where_look))))
  (return (as child (Pr (= sally.where_look where_look)))))

(deftest demo-sally-anne
  (ts/check-demo 'memo.demos.demo-sally-anne-test "demo-sally-anne"))
