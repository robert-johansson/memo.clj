(ns memo.demos.demo-trucks-test
  "memo's demo/demo-trucks.ipynb: the food-truck problem (Baker, Jara-Ettinger,
  Saxe & Tenenbaum): planning as an MDP and as a POMDP over a parking lot,
  then inferring beliefs and desires from a path. The demo's interact and
  tqdm drive its figures; the recorder stands in for them."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :as domains :refer [defdomain defenum]]
            [memo.test-support :as ts]))

(def W 10)
(def H 5)
(defdomain S :x W :y H)
(def maze "Walls at y = 2 for x >= 1, indexed as S packs (x, y)."
  (vec (for [x (range W) y (range H)] (if (and (>= x 1) (= y 2)) 1 0))))

(def SPOT_X_LOC (S {:x (dec W) :y 0}))
(def SPOT_Y_LOC (S {:x 0 :y (dec H)}))

(defn is-terminating [s] (or (== s SPOT_X_LOC) (== s SPOT_Y_LOC)))

(defenum A LEFT RIGHT UP DOWN NOP)
(def coord-actions [[-1 0] [1 0] [0 -1] [0 1] [0 0]])

(defn Tr [s a s_]
  (let [[x y] (domains/fields S s)
        [dx dy] (nth coord-actions (long a))
        next-state (S {:x (min (max (+ x dx) 0) (dec W)) :y (min (max (+ y dy) 0) (dec H))})]
    (+ (if (and (zero? (nth maze next-state)) (== next-state s_)) 1.0 0.0)    ; free: move there
       (if (and (= 1 (nth maze next-state)) (== s s_)) 1.0 0.0))))           ; a wall: stay

(def Utility [-20 0 50 100])
(defdomain G :uK (count Utility) :uL (count Utility) :uM (count Utility))
(defenum D L_PARKED M_PARKED)

(defn R [d s g]
  (let [[uK uL uM] (domains/fields G g)]
    (cond
      (== s SPOT_Y_LOC) (nth Utility uK)
      (and (== s SPOT_X_LOC) (== d (D :L_PARKED))) (nth Utility uL)
      (and (== s SPOT_X_LOC) (== d (D :M_PARKED))) (nth Utility uM)
      :else 0)))

(defn cost [a] (if (== a (A :NOP)) 0 1))

(def gamma 0.99)
(def beta 2.5)

(defmemo ^:cache Q-mdp [[d D] [s S] [a A] [g G]] [t]
  (alice :knows d s a g)
  (alice :given [s_ S] :wpp (Tr s a s_))
  (alice :chooses [a_ A] :to-maximize (if (> t 0) (Q-mdp [d s_ a_ g] (- t 1)) 1))
  (return (E (as alice (+ (- (R d s_ g) (cost a))
                          (if (<= t 0)
                            0
                            (if (is-terminating s_) 0 (* #{gamma} (Q-mdp [d s_ a_ g] (- t 1))))))))))

(def B (mapv #(/ % 6.0) (range 7)))   ; np.linspace(0, 1, 7)
(defn get-belief [b d] (nth [b (- 1 b)] (long d)))

(defenum O L_PARKED M_PARKED NOTHING)

(defn can-see-X [s] (< (S :y s) 2))

(defn obs [d s o]
  (cond
    (and (can-see-X s) (== d (D :L_PARKED))) (nth [0.99 0.0 0.01] (long o))
    (and (can-see-X s) (== d (D :M_PARKED))) (nth [0.0 0.99 0.01] (long o))
    (not (can-see-X s)) (nth [0.0 0.0 1.0] (long o))
    :else 0.0))

(defmemo ^:cache BeliefUpdate [[b B] [s S] [o O] [b_ B]] []
  (alice :knows b s)
  (alice :thinks
    (env :knows b s)
    (env :chooses [d D] :wpp (get-belief b d))
    (env :chooses [o O] :wpp (obs d s o)))
  (alice :observes env.o :is o)
  (alice :knows b_)
  (return (exp (* -10 (as alice (abs (- (Pr (= env.d 0)) (get-belief b_ 0))))))))

(defmemo ^:cache Q [[b B] [s S] [a A] [g G]] [t]
  (alice :knows b s a g)
  (alice :thinks
    (env :knows b s a)
    (env :chooses [d D] :wpp (get-belief b d)))
  (alice :given [s_ S] :wpp (Tr s a s_))
  (alice :thinks
    (env :knows s_)
    (env :chooses [o O] :wpp (obs d s_ o)))
  (alice :snapshots-self-as future_alice)
  (return (E (as alice
                 (+ (E (- (R env.d s_ g) (cost a)))
                    (if (<= t 0)
                      0.0
                      (if (is-terminating s_)
                        0.0
                        (imagine
                         (future_alice :observes env.o :is env.o)
                         (future_alice :chooses [b_ B] :wpp (BeliefUpdate [b s_ env.o b_]))
                         (future_alice :chooses [a_ A] :to-maximize (Q [b s_ a_ g] (- t 1)))
                         (* #{gamma} (E (as future_alice (Q [b s_ a_ g] (- t 1)))))))))))))

(def d0 (D :L_PARKED))
(def s0 (S {:x 3 :y 3}))
(def actions (mapv #(A %) [:LEFT :LEFT :LEFT :UP :UP :RIGHT :RIGHT :RIGHT
                           :LEFT :LEFT :LEFT :DOWN :DOWN :DOWN]))

(defmemo ^:cache invplan [[b B] [s S] [g G]] [t]
  (viewer :thinks
    (alice :given [b B] [g G] [s S] :wpp (if (<= t 0) (= s #{s0}) (invplan [b s g] (- t 1))))
    (alice :chooses [a A] :wpp (exp (* #{beta} (Q [b s a g] 100)))))
  (viewer :observes-that (= alice.a #{(nth actions t)}))
  (viewer :thinks
    (alice :given [s_ S] :wpp (Tr s a s_))
    (alice :given [o O] :wpp (obs #{d0} s_ o))
    (alice :given [b_ B] :wpp (BeliefUpdate [b s_ o b_])))
  (viewer :knows b s g)
  (return (as viewer (Pr (= b alice.b_) (= s alice.s_) (= g alice.g)))))

(deftest ^:slow demo-trucks
  ;; Q's values reach about 100, and some cells are small differences of
  ;; such values over 100 iterations: Python memo's float32 rounding there is
  ;; about 1e-5 (one cell, 0.0416800 in Python, is 0.0416826).
  (ts/check-demo 'memo.demos.demo-trucks-test "demo-trucks" {:atol 1e-4}))
