(ns memo.demos.demo-pc-test
  "memo's demo/demo-pc.ipynb: planning as inference, by imagine and by a
  product domain (Python memo's domain(...), memo.clj's defdomain). The
  demo's first cell is WebPPL, run by bash."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defdomain defenum values]]
            [memo.test-support :as ts]))

(defenum Restaurant Indoor Outdoor)
(defenum Weather Sunny Stormy)

(defn utility [r s]
  (+ (if (== r (Restaurant :Outdoor)) 11 10)
     (if (and (== s (Weather :Stormy)) (== r (Restaurant :Outdoor))) -100 0)))

(defmemo model [[r Restaurant]] []
  (alice :chooses [r Restaurant] :wpp (imagine
                                       (world :chooses [s Weather] :wpp 0.5)
                                       (exp (E (utility r world.s)))))
  (return (Pr (= alice.r r))))

(defdomain RS :r (count (values Restaurant)) :s (count (values Weather)))
(defn get-r [rs] (RS :r rs))
(defn get-s [rs] (RS :s rs))

(defmemo model- [[r Restaurant]] []
  (alice :chooses [rs RS] :wpp (exp (utility (get-r rs) (get-s rs))))
  (return (Pr (= (get-r alice.rs) r))))

(deftest demo-pc
  (ts/check-demo 'memo.demos.demo-pc-test "demo-pc"))
