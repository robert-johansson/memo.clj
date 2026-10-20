(ns memo.demos.demo-empowerment-test
  "memo's demo/demo-empowerment.py: 5-step empowerment in a gridworld (after
  Klyubin, Polani & Nehaniv 2005), by the Blahut-Arimoto algorithm written
  as memo models. The demo builds its models in a module (make_module) over
  X, Y, Z = Ax, S, S; they are plain defmemos here. Its tables reach
  3125 x 169 x 169 cells, so the transition runs on whole arrays."
  (:refer-clojure :exclude [+ * ==])
  (:require [clojure.test :refer [deftest]]
            [gen.array :as array :refer [+ * ==]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defdomain defenum]]
            [memo.test-support :as ts]))

(def N 13)
(def world "A wall at the center." (assoc-in (vec (repeat N (vec (repeat N 0)))) [(quot N 2) (quot N 2)] 1))

(defdomain S :x N :y N)
(defenum A N S W E O)
(defdomain Ax :a1 5 :a2 5 :a3 5 :a4 5 :a5 5)

(defn tr1
  "One move: N, S, W, E or stay; into the wall or off the grid, stay."
  [s a]
  (let [a (array/astype a :int32)
        x_ (array/clip (+ (S :x s) (array/take (array/array [0 0 -1 1 0]) a)) 0 (dec N))
        y_ (array/clip (+ (S :y s) (array/take (array/array [-1 1 0 0 0]) a)) 0 (dec N))
        s_ (+ (* x_ N) y_)]
    (array/where (== (array/take (array/array (vec (flatten world))) (array/astype s_ :int32)) 1) s s_)))

(defn ^:memo/vectorized Tr
  "Whether the five moves of ax take s to s_."
  [s_ ax s]
  (== s_ (reduce (fn [s k] (tr1 s (Ax k ax))) s [:a1 :a2 :a3 :a4 :a5])))

(defmemo q [[x Ax] [z S]] [t]
  (alice :knows z)
  (alice :chooses [x Ax] :wpp (imagine
                               (bob :knows x z)
                               (bob :chooses [y S] :wpp (Tr y x z))
                               (bob :thinks
                                 (charlie :knows y z)
                                 (charlie :chooses [x Ax] :wpp (if (> t 0) (Q [x y z] (- t 1)) 1)))
                               (exp (E (as bob (H charlie.x))))))
  (return (Pr (= alice.x x))))

(defmemo Q [[x Ax] [y S] [z S]] [t]
  (alice :knows x y z)
  (alice :thinks
    (bob :knows z)
    (bob :chooses [x Ax] :wpp (q [x z] t))
    (bob :chooses [y S] :wpp (Tr y x z)))
  (alice :observes bob.y :is y)
  (return (as alice (Pr (= bob.x x)))))

(defmemo C [[z S]] [t]
  (alice :knows z)
  (alice :chooses [x Ax] :wpp (q [x z] t))
  (alice :chooses [y S] :wpp (Tr y x z))
  (return (/ (- (+ (H alice.x) (H alice.y)) (H alice.x alice.y)) (log 2))))   ; in bits

(defmemo empowerment [[s S]] [t]
  (return (C [s] t)))

(deftest ^:slow demo-empowerment
  ;; Its tables have 3125 x 169 x 169 cells: 714 MB each in float64. On the
  ;; JVM backend, which holds every value it computes, it fits in memory
  ;; because memo.lower keeps no value past its last use and spot-checks Tr
  ;; at sampled elements only (memo.lower-test); without either, it ran out
  ;; of a 16 GB heap. With them it needs the 8 GB heap the :test alias sets
  ;; (it fails at 6 GB).
  (ts/check-demo 'memo.demos.demo-empowerment-test "demo-empowerment"))
