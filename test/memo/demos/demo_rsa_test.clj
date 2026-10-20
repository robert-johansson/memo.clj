(ns memo.demos.demo-rsa-test
  "memo's demo/demo-rsa.py: recursive RSA (the Handbook's reference game).
  The demo then fits beta to data by grid search and by gradient descent
  through the model; the calls it makes outside jax.grad are recorded."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum U [GREEN 2r0001] [PINK 2r0010] [SQUARE 2r0100] [ROUND 2r1000])   ; utterances
(defenum R [GREEN_SQUARE 2r0101] [GREEN_CIRCLE 2r1001] [PINK_CIRCLE 2r1010]) ; referents

(defn denotes [u r] (not= 0 (bit-and (long u) (long r))))

(defmemo L [[u U] [r R]] [beta t]
  (listener :thinks
    (speaker :given [r R] :wpp 1)
    (speaker :chooses [u U] :wpp (* (denotes u r) (if (= t 0) 1 (exp (* beta (L [u r] beta (- t 1))))))))
  (listener :observes speaker.u :is u)
  (listener :chooses [r R] :wpp (Pr (= speaker.r r)))
  (return (Pr (= listener.r r))))

(deftest demo-rsa
  (ts/check-demo 'memo.demos.demo-rsa-test "demo-rsa"))
