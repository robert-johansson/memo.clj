(ns memo.demos.demo-newcomb-test
  "memo's demo/demo-newcomb.ipynb: Newcomb's problem, and two models the
  demo shows must not compile."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Pick AB B)

(defn payout [g y] (nth (nth [[1e3 0.0] [(+ 1e3 1e6) 1e6]] (long g)) (long y)))

(defmemo fearful [[p Pick]] []
  (alice :chooses [y Pick] :to-maximize (imagine
                                         (god :knows y)
                                         (god :chooses [g Pick] :to-maximize (= g y))
                                         (E (payout god.g y))))
  (return (Pr (= alice.y p))))

(defmemo realist [[p Pick]] []
  (alice :thinks (god :chooses [g Pick] :wpp 1))
  (alice :chooses [y Pick] :to-maximize (E (payout god.g y)))
  (return (Pr (= alice.y p))))

(def compile-forms
  {:should-fail-to-compile
   '(defmemo should-fail-to-compile [[p Pick]] []
      (alice :thinks (god :chooses [g Pick] :wpp (imagine
                                                  (alice :chooses [y Pick] :to-maximize (= g y))
                                                  (E (= g alice.y)))))
      (alice :chooses [y Pick] :to-maximize (= g y))
      (return (Pr (= alice.y p))))
   :should-also-fail-to-compile
   '(defmemo should-also-fail-to-compile [[p Pick]] []
      (alice :thinks (god :chooses [g Pick] :wpp 1))
      (alice :chooses [y Pick] :to-maximize (= y god.g))
      (return (Pr (= alice.y p))))})

(deftest demo-newcomb
  (ts/check-demo 'memo.demos.demo-newcomb-test "demo-newcomb" {:compile-forms compile-forms}))
