(ns memo.demos.demo-cheryl-test
  "memo's demo/demo-cheryl.ipynb: Cheryl's birthday, as five rounds of
  announcements about who knows what."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Month February March April)
(def Day (vec (range 1 32)))
(defenum U DUNNO KNOWN)

(defn possible "31 days hath..." [m d] (<= d (nth [29 31 30] (long m))))

(defmemo a-u1 [[m Month] [u U]] []
  (a :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d)))
  (a :observes c.m :is m)
  (return (= u (as a (= (Var c.d) 0)))))           ; Alice's variance over Cheryl's d

(defmemo a-u2 [[m Month] [u U]] []
  (a :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d))
    (b :thinks
      (c :chooses [m Month] :wpp 1)
      (c :chooses [d Day] :wpp (possible m d)))
    (b :observes c.d :is c.d))
  (a :observes c.m :is m)
  (return (= u (as a (> (Pr (as b (= (Var c.m) 0))) 0)))))

(defmemo b-u3 [[d Day] [u U]] []
  (b :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d)))
  (b :observes c.d :is d)
  (return (= u (as b (= (Var c.m) 0)))))

(defmemo b-u4 [[d Day] [u1 U] [u2 U] [u U]] []
  (b :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d))
    (a :thinks
      (c :chooses [m Month] :wpp 1)
      (c :chooses [d Day] :wpp (possible m d)))
    (a :observes c.m :is c.m)
    (a :chooses [u1 U] :wpp (a-u1 [c.m u1]))
    (a :chooses [u2 U] :wpp (a-u2 [c.m u2])))
  (b :observes c.d :is d)
  (b :observes a.u1 :is u1)
  (b :observes a.u2 :is u2)
  (return (= u (as b (= (Var c.m) 0)))))

(defmemo a-u5 [[m Month] [u1 U] [u2 U] [u3 U] [u4 U] [u U]] []
  (a :knows u1 u2)
  (a :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d))
    (b :thinks
      (c :chooses [m Month] :wpp 1)
      (c :chooses [d Day] :wpp (possible m d)))
    (b :knows u1 u2)
    (b :observes c.d :is c.d)
    (b :chooses [u3 U] :wpp (b-u3 [c.d u3]))
    (b :chooses [u4 U] :wpp (b-u4 [c.d u1 u2 u4])))
  (a :observes c.m :is m)
  (a :observes b.u3 :is u3)
  (a :observes b.u4 :is u4)
  (return (= u (as a (= (Var c.d) 0)))))

(defmemo puzzle [[m Month] [d Day] [u1 U] [u2 U] [u3 U] [u4 U] [u5 U]] []
  (z :thinks
    (c :chooses [m Month] :wpp 1)
    (c :chooses [d Day] :wpp (possible m d))
    (c :chooses [u1 U] :wpp (a-u1 [m u1]))
    (c :chooses [u2 U] :wpp (a-u2 [m u2]))
    (c :chooses [u3 U] :wpp (b-u3 [d u3]))
    (c :chooses [u4 U] :wpp (b-u4 [d u1 u2 u4]))
    (c :chooses [u5 U] :wpp (a-u5 [m u1 u2 u3 u4 u5])))
  (z :observes c.u1 :is u1)
  (z :observes c.u2 :is u2)
  (z :observes c.u3 :is u3)
  (z :observes c.u4 :is u4)
  (z :observes c.u5 :is u5)
  (z :knows m d)
  (return (as z (E (and (= c.m m) (= c.d d))))))

(deftest demo-cheryl
  (ts/check-demo 'memo.demos.demo-cheryl-test "demo-cheryl"))
