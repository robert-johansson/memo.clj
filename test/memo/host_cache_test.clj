(ns memo.host-cache-test
  "Host functions that opt in to a cache of their tables
  (^:memo/cache-table, M4, DESIGN.md §14.4): an expensive pure function (a
  simulator, a language model's score) tabulated once per arguments, not at
  every call."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.errors :as err]
            [memo.lower :as lower]
            [memo.test-support :as ts]))

(def X [0 1 2])
(def calls (atom 0))

(defn ^:memo/cache-table expensive [x] (swap! calls inc) (* 10.0 x))
(defn plain [x] (swap! calls inc) (* 10.0 x))

(defmemo scored [[x X]] [k]
  (alice :chooses [y X] :wpp (exp (* 0.1 (expensive (* k y)))))
  (return (Pr (= alice.y x))))

(defmemo scored-plain [[x X]] [k]
  (alice :chooses [y X] :wpp (exp (* 0.1 (plain (* k y)))))
  (return (Pr (= alice.y x))))

(defn- counted [thunk] (reset! calls 0) (thunk) @calls)

(defn- message [thunk]
  (try (thunk) nil (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(deftest a-marked-function-is-tabulated-once-per-arguments
  (scored 1)
  (is (= 0 (counted #(scored 1))) "the same arguments: the table kept")
  (is (= 3 (counted #(scored 2))) "another value of its dynamic argument: a new table, 3 cells")
  (is (= 0 (counted #(scored 2))))
  (is (= 3 (counted #(scored-plain 1))) "unmarked: tabulated at every call")
  (is (= 3 (counted #(scored-plain 1))))
  (is (ts/close? (ts/host (scored-plain 2)) (ts/host (scored 2))) "the same table either way"))

(deftest redefining-the-function-drops-its-tables
  (scored 3)
  (is (= 0 (counted #(scored 3))))
  (let [old expensive]
    (try
      (alter-var-root #'expensive (constantly (with-meta (fn [x] (swap! calls inc) (* 20.0 x)) (meta old))))
      (is (= 3 (counted #(scored 3))) "a new value in the var: tabulated again")
      (finally (alter-var-root #'expensive (constantly old))))))

(deftest the-cache-is-bounded
  (binding [lower/*host-cache-cells* 4]
    (scored 5)
    (scored 6)                        ; 3 + 3 cells > 4: the cache starts over with this table
    (is (= 0 (counted #(scored 6))))
    (is (= 3 (counted #(scored 5))) "evicted")))

(deftest a-differentiated-value-still-refused
  (is (= "Cannot differentiate through the host function expensive"
         (message #((array/grad (fn [k] (array/sum (scored k)))) (array/array 1.0))))))
