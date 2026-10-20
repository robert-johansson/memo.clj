(ns memo.demos.demo-chimp-belief-revision-test
  "memo's demo/demo-chimp-belief-revision.ipynb: a chimp chooses a box from
  evidence of food, then revises with a second piece of evidence."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Box FST_BOX SND_BOX)
(defenum FoodStatus EMPTY_BOX FOODY_BOX)
(defenum EvidenceType CRUMBS NOISES VISUAL)

(defn ber
  "jax.scipy.stats.bernoulli.pmf: p for 1, 1 - p for 0."
  [k p]
  (if (== k 1) p (- 1 p)))

(defn p-evidence
  "Probability of foody evidence given evidence type and food status."
  [et fs]
  (nth (nth [[0.5 0.6]      ; crumbs: very weak evidence
             [0.3 0.6]      ; noises: moderately weak evidence
             [0.01 0.9]]    ; visual: strong evidence
            (long et))
       (long fs)))

(defmemo first-choice [[et1 EvidenceType] [b Box]] []
  (chimp :knows et1)
  (chimp :wants get_food (= (if (= b #{Box.FST_BOX}) world.box1_fs world.box2_fs) #{FoodStatus.FOODY_BOX}))
  (chimp :thinks
    (world :chooses [box1_fs FoodStatus] :wpp 1)
    (world :chooses [box2_fs FoodStatus] :wpp 1)
    (world :knows et1)
    (world :chooses [e1 FoodStatus] :wpp (ber e1 (p-evidence et1 box1_fs))))
  (chimp :observes-that (= world.e1 #{FoodStatus.FOODY_BOX}))
  (chimp :chooses [b Box] :wpp (exp (* 10.0 (EU get_food))))
  (return (Pr (= chimp.b b))))

(defmemo second-choice [[et1 EvidenceType] [et2 EvidenceType] [b Box]] []
  (chimp :knows et1 et2)
  (chimp :wants get_food (= (if (= b #{Box.FST_BOX}) world.box1_fs world.box2_fs) #{FoodStatus.FOODY_BOX}))
  (chimp :thinks
    (world :chooses [box1_fs FoodStatus] :wpp 1)
    (world :chooses [box2_fs FoodStatus] :wpp 1)
    (world :knows et1 et2)
    (world :chooses [e1 FoodStatus] :wpp (ber e1 (p-evidence et1 box1_fs)))
    (world :chooses [e2 FoodStatus] :wpp (ber e2 (p-evidence et2 box2_fs))))
  (chimp :observes-that (= world.e1 #{FoodStatus.FOODY_BOX}))
  (chimp :observes-that (= world.e2 #{FoodStatus.FOODY_BOX}))
  (chimp :chooses [b Box] :wpp (exp (* 10.0 (EU get_food))))
  (return (Pr (= chimp.b b))))

(deftest demo-chimp-belief-revision
  (ts/check-demo 'memo.demos.demo-chimp-belief-revision-test "demo-chimp-belief-revision"))
