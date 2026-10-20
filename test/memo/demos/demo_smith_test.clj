(ns memo.demos.demo-smith-test
  "memo's demo/demo-smith.ipynb: Mr. Smith's two children, read two ways."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Gender BOY GIRL)
(defenum Child OLDER YOUNGER)

(defmemo interpretation-1 [] []
  "Interpret the sentence as: either c1 is a boy, or c2 is a boy, or both."
  (reader :thinks (smith :given [c1 Gender] [c2 Gender] :uniformly))
  (reader :observes-that (or (= smith.c1 #{Gender.BOY}) (= smith.c2 #{Gender.BOY})))
  (return (as reader (Pr (= smith.c1 #{Gender.BOY}) (= smith.c2 #{Gender.BOY})))))

(defmemo interpretation-2 [] []
  "Interpret the sentence as: one of the children was observed to be a boy."
  (reader :thinks
    (smith :given [c1 Gender] [c2 Gender] :uniformly)
    (smith :chooses [which_child Child] :uniformly)
    (smith :chooses [that_child_gender Gender] :to-be (if (= which_child 0) c1 c2)))
  (reader :observes-that (= smith.that_child_gender #{Gender.BOY}))
  (return (as reader (Pr (= smith.c1 #{Gender.BOY}) (= smith.c2 #{Gender.BOY})))))

(deftest demo-smith
  (ts/check-demo 'memo.demos.demo-smith-test "demo-smith"))
