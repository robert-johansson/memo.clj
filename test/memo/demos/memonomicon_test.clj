(ns memo.demos.memonomicon-test
  "memo's demo/Memonomicon.ipynb, the tutorial. It defines
  alice_chooses_card three times; the later ones are alice-chooses-card-2
  and -3 here, as the fixture names them, and bob calls the third, as the
  notebook's does. The notebook then fits beta by gradient descent; its
  calls outside jax.grad are recorded."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def cards [1 2 3])

(defmemo alice-chooses-card [] []
  (cast [alice])
  (alice :chooses [c cards] :wpp 1)
  (return (E alice.c)))

(defmemo alice-chooses-card-2 [[c cards]] []        ; notice the new axis c
  (cast [alice])
  (alice :chooses [c cards] :wpp 1)
  (return (E (= alice.c c))))                         ; the probability that alice chooses c

(defmemo alice-chooses-card-3 [[c cards]] [beta]    ; notice the new parameter beta
  (cast [alice])
  (alice :chooses [c cards] :wpp (exp (* beta c)))  ; notice the new wpp
  (return (E (= alice.c c))))

(defmemo bob-chooses-card [[c cards]] []
  (cast [alice bob])
  (bob :thinks                                      ; Bob's mental model of Alice
    (alice :chooses [c cards] :wpp (alice-chooses-card-3 [c] 3.0)))
  (bob :chooses [c cards] :wpp (E (not (= c alice.c))))   ; Bob avoids Alice's card
  (return (E (= bob.c c))))

(defmemo all-outcomes [[a cards] [b cards]] []
  (cast [alice bob])
  (alice :chooses [c cards] :wpp (alice-chooses-card-3 [c] 3.0))
  (bob :chooses [c cards] :wpp (bob-chooses-card [c]))
  (return (E (and (= alice.c a) (= bob.c b)))))

(def N [0 1 2 3])   ; the number of nice people
(def U [0 1 2])     ; the utterance: none, some or all of the people are nice

(defn meaning [n u] (nth [(== n 0) (> n 0) (== n 3)] (long u)))

(defmemo scalar [[n N] [u U]] [beta]
  (cast [speaker listener])
  (listener :thinks
    (speaker :given [n N] :wpp 1)
    (speaker :chooses [u U] :wpp (imagine
                                  (listener :knows u)
                                  (listener :chooses [n N] :wpp (meaning n u))
                                  (exp (* beta (E (= listener.n n)))))))
  (listener :observes speaker.u :is u)
  (listener :chooses [n N] :wpp (E (= speaker.n n)))
  (return (E (= listener.n n))))

(deftest memonomicon
  (ts/check-demo 'memo.demos.memonomicon-test "Memonomicon"))
