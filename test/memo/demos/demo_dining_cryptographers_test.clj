(ns memo.demos.demo-dining-cryptographers-test
  "memo's demo/demo-dining-cryptographers.ipynb: Chaum's dining
  cryptographers. The demo xors bits held as the integers 0 and 1; memo.clj
  computes xor on booleans only (DESIGN.md §5), so the twin compares each
  bit with 1 first."
  (:require [clojure.test :refer [deftest is]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Bit NOT_PAID PAID)
(defenum Who A_PAID B_PAID C_PAID NSA_PAID)

(defmemo model [[a_ Bit] [b_ Bit] [bx Bit] [cx Bit] [w Who]] []
  (a :knows a_ w)
  (a :thinks
    (world :chooses [w Who] :wpp (not= w 0))
    (b :chooses [b_ Bit] :wpp 1)
    (c :chooses [c_ Bit] :wpp 1)
    (b :knows world.w c.c_)
    (c :knows world.w a_)
    (b :chooses [bx Bit] :wpp (= (xor (xor (= b_ 1) (= c.c_ 1)) (= world.w 1)) (= bx 1)))
    (c :chooses [cx Bit] :wpp (= (xor (xor (= c_ 1) (= a_ 1)) (= world.w 2)) (= cx 1))))
  (a :observes b.b_ :is b_)
  (a :observes b.bx :is bx)
  (a :observes c.cx :is cx)
  (return (as a (Pr (= world.w w)))))

(defmemo model-as-written [[a_ Bit] [b_ Bit] [bx Bit] [cx Bit] [w Who]] []
  (a :knows a_ w)
  (a :thinks
    (world :chooses [w Who] :wpp (not= w 0))
    (b :chooses [b_ Bit] :wpp 1)
    (c :chooses [c_ Bit] :wpp 1)
    (b :knows world.w c.c_)
    (c :knows world.w a_)
    (b :chooses [bx Bit] :wpp (= (xor (xor b_ c.c_) (= world.w 1)) bx))
    (c :chooses [cx Bit] :wpp (= (xor (xor c_ a_) (= world.w 2)) cx)))
  (a :observes b.b_ :is b_)
  (a :observes b.bx :is bx)
  (a :observes c.cx :is cx)
  (return (as a (Pr (= world.w w)))))

(deftest demo-dining-cryptographers
  (ts/check-demo 'memo.demos.demo-dining-cryptographers-test "demo-dining-cryptographers"))

(deftest xor-on-integers-is-a-loud-error
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"memo.clj computes xor on booleans only"
                        (model-as-written))))
