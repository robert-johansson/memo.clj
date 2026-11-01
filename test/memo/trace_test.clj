(ns memo.trace-test
  "memo.core/*trace* (M4, DESIGN.md §14.3): Python memo's debug_trace."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
            [memo.test-support :as ts]))

(def X [0 1])

(defmemo rec [[x X]] [a (b 0)]
  (return (if (> a 0) (rec [x] :a (- a 1) :b (+ b 1)) b)))

(defmemo ^:cache cached-flip [[x X]] [p]
  (alice :chooses [x X] :wpp (if (= x 1) p (- 1 p)))
  (return (Pr (= alice.x x))))

(defn- traced-lines [thunk]
  (str/split-lines (with-out-str (binding [memo/*trace* true] (thunk)))))

(deftest off-by-default
  (is (= "" (with-out-str (rec 2)))))

(deftest a-call-and-its-calls
  (let [lines (traced-lines #(rec 2))]
    (is (= [" --> rec(2)"
            "   --> rec(a=1, b=1)"
            "     --> rec(a=0, b=2)"
            "    <--  rec(a=0, b=2) has shape [2]"]
           (take 4 lines)))
    (is (re-matches #"\s+time = \d+\.\d{6} sec" (nth lines 4)))
    (is (= "<--  rec(2) has shape [2]" (nth lines 7)) "the outer call ends last, unindented")
    (is (= 9 (count lines)) "three calls, three lines each")))

(deftest a-cached-answer-prints-nothing
  (cached-flip 0.25)
  (is (= [""] (traced-lines #(cached-flip 0.25))) "answered from the cache")
  (is (= " --> cached-flip(0.75)" (first (traced-lines #(cached-flip 0.75)))) "a new argument runs"))

(deftest cost-mode-prints-the-cost
  (let [lines (traced-lines #(memo/cost rec 0 {:b 9}))]
    (is (some #(re-matches #"\s+cost = \{:time \d+, :space \d+, :work \d+, :size \d+\}" %) lines))))

(deftest a-differentiated-argument-is-not-read
  (let [lines (atom nil)
        g (array/grad (fn [p] (reset! lines (traced-lines #(cached-flip p)))
                        (array/sum (cached-flip p))))]
    (is (ts/close? 0.0 (ts/host (g (array/array 0.3)))))
    (is (= " --> cached-flip(#array[])" (first @lines)) "its shape, not its value")))

(defmemo ^{:doc "an array parameter"} lookup [[x X]] [^:array t] (return #{(nth t 0)}))

(deftest a-vector-argument-prints-as-its-shape
  (is (= " --> lookup(#array[50])" (first (traced-lines #(lookup (vec (range 50.0))))))))

(defmemo picks [[x X]] []
  (alice :thinks (bob :chooses [y X] :wpp (+ 1 y)))
  (alice :chooses [x X] :wpp (exp (E (= x bob.y))))
  (return (Pr (= alice.x x))))

(deftest a-budget-shows
  (is (= " --> picks() with budget {alice {:samples 2, :algorithm :one-and-done, :draws 1}}"
         (first (traced-lines #(picks {:memo/budget {'alice {:samples 2}}}))))))
