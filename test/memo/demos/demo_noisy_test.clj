(ns memo.demos.demo-noisy-test
  "memo's demo/demo-noisy.ipynb: prosodic stress in a noisy channel, by
  levels of recursive reasoning between speaker S and listener L."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum Meaning Passed Aced)
(defenum Observation Only_Passed At_Least_Passed)
(defenum Utterance I_Passed I_Aced)
(defenum Prosody Unstressed Stressed)

(defmemo O [[m Meaning] [o Observation]] []
  " Generate an observation for the speaker given a true meaning "
  (world :knows m)
  (world :chooses [o Observation] :wpp (>= o m))
  (return (Pr (= world.o o))))

(defn denotes
  "\"I Passed\" is consistent with passing and acing; \"I Aced\" only with acing."
  [u m]
  (nth (nth [[1 1] [0 1]] (long u)) (long m)))

(defn corrupt
  "The probability that the listener hears u_ for utterance u with prosody p:
  u itself with probability 0.70 unstressed, 0.99 stressed."
  [u p u_]
  (nth [(if (== u u_) 0.70 0.30) (if (== u u_) 0.99 0.01)] (long p)))

(defn prior "Prior over meanings." [m] 1)

(def beta 3.0)   ; the speaker's softmax
(def cost 0.1)   ; the cost of stressed prosody

(defmemo S [[o Observation] [u Utterance] [p Prosody]] [level]
  (speaker :thinks
    (world :chooses [m Meaning] :wpp (prior m))
    (world :chooses [o Observation] :wpp (O [m o])))
  (speaker :observes world.o :is o)
  (speaker :chooses [u Utterance] [p Prosody]
           :wpp (exp (* #{beta} (- (imagine
                                    (world :knows u p)
                                    (world :chooses [u_ Utterance] :wpp (corrupt u p u_))
                                    (listener :knows world.u_ p)
                                    (listener :chooses [m Meaning] :wpp (L [world.u_ p m] level))
                                    (- (KL world.m listener.m)))
                                   (* p #{cost})))))
  (return (Pr (= speaker.u u) (= speaker.p p))))

(defmemo L [[u Utterance] [p Prosody] [m Meaning]] [level]
  (listener :thinks
    (speaker :given [m Meaning] :wpp (prior m))
    (speaker :given [o Observation] :wpp (O [m o]))
    (speaker :chooses [u Utterance] [p Prosody] :wpp (if (> level 0) (S [o u p] (- level 1)) (denotes u m)))
    (speaker :given [u_ Utterance] :wpp (corrupt u p u_)))
  (listener :observes speaker.u_ :is u)
  (listener :observes speaker.p :is p)
  (listener :knows m)
  (return (as listener (Pr (= speaker.m m)))))

(deftest demo-noisy
  (ts/check-demo 'memo.demos.demo-noisy-test "demo-noisy"))
