(ns memo.demos.demo-mouths-of-babes-test
  "memo's demo/demo-mouths-of-babes.ipynb: a child names a novel event to
  inform, or to connect. The demo draws through ipywidgets' interact; the
  recorder calls it once, with the widgets' initial values."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Vocab Ball Bear Duck Milk Shoe)

(defmemo novel-event-inform [[fixed_event Vocab] [novel_event Vocab] [utterance Vocab]] [β]
  (baby :knows fixed_event novel_event)
  (baby :wants inform (Pr (= parent.referent_guess novel_event)))
  (baby :thinks
    (parent :thinks
      (scene :chooses [fixed_event Vocab] :wpp 1)
      (scene :chooses [novel_event Vocab] :wpp 1))
    (parent :observes scene.fixed_event :is fixed_event)
    (parent :thinks
      (baby :knows scene.fixed_event scene.novel_event)
      (baby :chooses [referent Vocab] :wpp (or (= referent scene.fixed_event) (= referent scene.novel_event)))
      (baby :chooses [utterance Vocab] :wpp (+ (= utterance referent) 0.25))))
  (baby :chooses [utterance Vocab] :wpp (exp (* β (EU inform))))
  (baby :thinks
    (parent :observes baby.utterance :is utterance)
    (parent :guesses [referent_guess Vocab] :wpp (Pr (= referent_guess baby.referent))))
  (return (Pr (= baby.utterance utterance))))

(defmemo novel-event-connect [[fixed_event Vocab] [novel_event Vocab] [utterance Vocab]] [β]
  (baby :knows fixed_event novel_event)
  (baby :wants connect (Pr (= parent.referent_guess referent)))
  (baby :thinks
    (parent :thinks
      (scene :chooses [fixed_event Vocab] :wpp 1)
      (scene :chooses [novel_event Vocab] :wpp 1))
    (parent :observes scene.fixed_event :is fixed_event)
    (parent :thinks
      (baby :knows scene.fixed_event scene.novel_event)
      (baby :chooses [referent Vocab] :wpp (or (= referent scene.fixed_event) (= referent scene.novel_event)))
      (baby :chooses [utterance Vocab] :wpp (+ (= utterance referent) 0.25))))
  (baby :chooses [referent Vocab] :wpp (exp (* β (EU connect))))
  (baby :chooses [utterance Vocab] :wpp (exp (* β (EU connect))))
  (baby :thinks
    (parent :observes baby.utterance :is utterance)
    (parent :guesses [referent_guess Vocab] :wpp (Pr (= referent_guess baby.referent))))
  (return (Pr (= baby.utterance utterance))))

(deftest demo-mouths-of-babes
  (ts/check-demo 'memo.demos.demo-mouths-of-babes-test "demo-mouths-of-babes"))
