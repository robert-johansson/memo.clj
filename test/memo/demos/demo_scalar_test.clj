(ns memo.demos.demo-scalar-test
  "memo's demo/demo-scalar.py: scalar implicature (some, not all) over
  10,001 counts."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(def NN 10000)
(def N (vec (range (inc NN))))
(defenum U NONE SOME ALL)

(defn meaning [n u] (nth [(== n 0) (> n 0) (== n NN)] (long u)))

(defmemo scalar [[n N] [u U]] []
  (listener :thinks
    (speaker :chooses [n N] :wpp 1)
    (speaker :chooses [u U] :wpp (imagine
                                  (listener :knows u)
                                  (listener :chooses [n N] :wpp (meaning n u))
                                  (Pr (= listener.n n)))))
  (listener :observes speaker.u :is u)
  (listener :chooses [n N] :wpp (E (= speaker.n n)))
  (return (Pr (= listener.n n))))

(deftest demo-scalar
  (ts/check-demo 'memo.demos.demo-scalar-test "demo-scalar"))
