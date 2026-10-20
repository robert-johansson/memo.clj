(ns memo.demos.demo-eig-test
  "memo's demo/demo-eig.ipynb: the expected information gain of questions
  about two dice."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.test-support :as ts]))

(def is-prime  [0 0 1 1 0 1 0 1 0 0 0 1 0])
(def is-square [1 1 0 0 1 0 0 0 0 1 0 0 0])
(def is-pow-2  [0 1 1 0 1 0 0 0 1 0 0 0 0])

(def Qs
  [#(== % 7) #(== % 12) #(> % 10) #(> % 8) #(> % 6) #(> % 5)
   #(== (mod % 2) 0) #(== (mod % 2) 1) #(== (mod % 3) 0) #(== (mod % 4) 0) #(== (mod % 5) 0)
   #(== 1 (nth is-prime %)) #(== 1 (nth is-square %)) #(== 1 (nth is-pow-2 %))])

(def N (vec (range 1 7)))
(def Q (vec (range (count Qs))))
(def A [0 1])

(defn respond [q a n] (== (if ((nth Qs (long q)) (long n)) 1 0) a))

(defmemo eig [[q Q]] []
  (alice :knows q)
  (alice :thinks
    (bob :chooses [n_red N] :wpp 1)
    (bob :chooses [n_blu N] :wpp 1)
    (bob :knows q)
    (bob :chooses [a A] :wpp (respond q a (+ n_red n_blu))))
  (alice :snapshots-self-as future_self)
  (return (as alice (imagine
                     (future_self :observes bob.a :is bob.a)
                     (- (H bob.n_red bob.n_blu) (E (as future_self (H bob.n_red bob.n_blu))))))))

(deftest demo-eig
  ;; EIG is a difference of entropies near 3.6, so Python memo's float32
  ;; rounding of them shows: for n == 12 the EIG is the binary entropy of
  ;; 1/36, 0.1269305; memo.clj on the JVM gives 0.12693055, Python 0.12693143.
  (ts/check-demo 'memo.demos.demo-eig-test "demo-eig" {:atol 2e-6}))
