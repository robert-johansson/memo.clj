(ns memo.demos.demo-pomdp-test
  "memo's demo/demo-pomdp.ipynb: a crying baby as a POMDP, solved over a
  discretized belief space by value iteration with snapshots."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.test-support :as ts]))

(defenum S Hungry Sated)
(defenum A Feed Sing Ignore)
(defenum O Crying Quiet)

(def B (mapv #(/ % 49.0) (range 50)))   ; the belief space, P(hungry): np.linspace(0, 1, 50)

(defn get-belief [b s] (nth [b (- 1 b)] (long s)))

(defn Tr [s a s_]
  (let [z (nth (nth [[0.0 1.0 1.0]      ; P(hungry | s, a) if hungry
                     [0.0 0.1 0.1]]     ; if sated
                    (long s))
               (long a))]
    (nth [z (- 1 z)] (long s_))))

(defn Obs [o s a]
  (let [z (nth (nth [[0.8 0.9 0.8]      ; P(cry | s, a) if hungry
                     [0.1 0.0 0.1]]     ; if sated
                    (long s))
               (long a))]
    (nth [z (- 1 z)] (long o))))

(defn R [s a] (+ (nth [-10 0] (long s)) (nth [-5 -0.5 0] (long a))))

(defmemo ^:cache Q [[b B] [a A]] [t]
  (alice :knows b a)
  (alice :thinks
    (env :knows b a)
    (env :chooses [s S] :wpp (get-belief b s))
    (env :chooses [s_ S] :wpp (Tr s a s_))
    (env :chooses [o O] :wpp (Obs o s_ a)))
  (alice :snapshots-self-as future_alice)
  (return (as alice (+ (E (R env.s a))
                       (if (<= t 0)
                         0.0
                         (* 0.9 (imagine
                                 (future_alice :observes env.o :is env.o)
                                 (future_alice :chooses [b_ B] :wpp (exp (* -100.0 (abs (- (E (= env.s_ 0)) b_)))))
                                 (env :forgets-about s s_ o)
                                 (future_alice :chooses [a_ A] :to-maximize (Q [b_ a_] (- t 1)))
                                 (E (as future_alice (Q [b_ a_] (- t 1)))))))))))

(deftest demo-pomdp
  (ts/check-demo 'memo.demos.demo-pomdp-test "demo-pomdp"))
