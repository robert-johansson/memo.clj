(ns memo.demos.demo-mdp-test
  "memo's demo/demo-mdp.ipynb: planning in a maze (Q-values by value
  iteration, 1000 steps), and inverse planning of a goal from an action.
  The transition function runs on whole arrays (^:memo/vectorized): on
  host numbers it would be called 778,000 times at each of the 1000 steps."
  (:refer-clojure :exclude [+ * mod ==])
  (:require [clojure.test :refer [deftest]]
            [gen.array :as array :refer [+ * mod ==]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def H 21)
(def W 21)
(def S (vec (range (clojure.core/* H W))))   ; the state space
(def G [0 (dec (clojure.core/* H W))])       ; the goals: the NW and SE corners
(def A [0 1 2 3])                            ; left, right, up, down

(def maze
  "memo's assets/logo-maze.png, as 1 - the image (1 is a wall)."
  (->> ["000000000000000000000" "000000000000000000000" "001100011000111111100"
        "001110111000100000100" "001011101000100000000" "001001001000111100000"
        "001000001000100000000" "001000011000100000011" "001000011000111111111"
        "001100000000000000000" "001100000000000000000" "000000011000000000000"
        "001110011000110001100" "001000001000111011100" "001000001000101110100"
        "001001001000100100100" "001000001000100000100" "001000001000110000100"
        "001111111000110000100" "000000000000000001100" "000000000000000001100"]
       (mapcat (fn [row] (map #(Character/digit ^char % 10) row)))
       vec))

(defn ^:memo/vectorized Tr
  "P(s_ | s, a): move unless the next cell is a wall, then stay."
  [s a s_]
  (let [x (mod s W)
        y (array/floor-div s W)
        a (array/astype a :int32)
        nx (array/clip (+ x (array/take (array/array [-1 1 0 0]) a)) 0 (dec W))
        ny (array/clip (+ y (array/take (array/array [0 0 -1 1]) a)) 0 (dec H))
        next-state (+ nx (* W ny))
        free (== (array/take (array/array maze) (array/astype next-state :int32)) 0)]
    (+ (* 1.0 (array/logical-and free (== next-state s_)))                      ; the next cell is free
       (* 1.0 (array/logical-and (array/logical-not free) (== s s_))))))       ; a wall: stay

(defn ^:memo/vectorized R "The reward." [s a g] (array/- (* 1.0 (== s g)) 0.1))
(defn ^:memo/vectorized is-terminating [s g] (== s g))
(defn gamma "The discount factor." [] 1.0)

(defmemo ^:cache Q [[s S] [a A] [g G]] [t]
  (alice :knows s a g)
  (alice :given [s_ S] :wpp (Tr s a s_))
  (alice :chooses [a_ A] :to-maximize (if (< t 1) 0.0 (Q [s_ a_ g] (- t 1))))
  (return (E (+ (R s a g) (if (< t 1)
                            0.0
                            (if (is-terminating s g)
                              0.0
                              (* (gamma) (Q [alice.s_ alice.a_ g] (- t 1)))))))))

(defmemo invplan [[s S] [a A] [g G]] [t]
  (observer :knows a s g)
  (observer :thinks
    (alice :chooses [g G] :wpp 1)
    (alice :knows s)
    (alice :chooses [a A] :wpp (exp (* 2 (Q [s a g] t)))))
  (observer :observes alice.a :is a)
  (return (as observer (E (= alice.g g)))))

(deftest ^:slow demo-mdp
  ;; Q(1000): about 6 s on MLX (Metal), 5 minutes on the JVM backend.
  (ts/check-demo 'memo.demos.demo-mdp-test "demo-mdp"))
