(ns memo.demos.demo-politeness-test
  "memo's demo/demo-politeness.ipynb: polite speech, trading informativeness
  against the listener's feelings (Yoon, Tessler, Goodman & Frank)."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(def S [0 1 2 3])   ; true world states (number of hearts)
(defenum W terrible not_amazing not_terrible amazing)

(defn V "Value to the listener: linear in s." [s] s)
(defn state-prior "An uninformative state prior." [s] 1)

(defn L
  "Literal likelihood L(w | s), soft semantics."
  [w s]
  (nth (nth [[1 0.5 0 0]      ; terrible
             [1 1 1 0]        ; not amazing
             [0 1 1 1]        ; not terrible
             [0 0 0.5 1]]     ; amazing
            (long w))
       (long s)))

(defmemo S1 [[s S] [w W]] [alpha phi]
  (speaker :knows s)
  (speaker :thinks
    (listener :thinks
      (speaker :given [s S] :wpp (state-prior s))
      (speaker :chooses [w W] :wpp (L w s))))
  (speaker :chooses [w W] :wpp (exp (* alpha (imagine
                                              (listener :observes speaker.w :is w)
                                              (listener :knows s)
                                              (+ (* phi (as listener (log (Pr (= speaker.s s)))))
                                                 (* (- 1 phi) (as listener (E (V speaker.s)))))))))
  (return (Pr (= speaker.w w))))

(deftest demo-politeness
  (ts/check-demo 'memo.demos.demo-politeness-test "demo-politeness"))
