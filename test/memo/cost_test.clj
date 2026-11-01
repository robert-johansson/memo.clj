(ns memo.cost-test
  "Cost reflection (DESIGN.md §14.1).
  - size @: every model of the corpus whose Python twin parses has the size
    Python memo 1.3.3's codegen.weigh gives it (each fixture file's :sizes,
    from test/fixtures/generate.py), compared on the Clojure twins in the
    namespace that fixture belongs to.
  - time @ and space @ count memo.clj's own run: their laws here (a constant
    is free, a recursion is linear in its depth, a memo call counts its
    callee per call site, a host function its :memo/cost, constants count
    whatever memo.lower caches, a budget's k raises the count) and their
    errors. test.py's cost cases are in memo.testpy-test."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(defn- fixture-namespace
  "The test namespace holding a fixture file's Clojure twins:
  slice1.edn -> memo.slice1-test, demos/demo-rsa.edn -> memo.demos.demo-rsa-test."
  [rel]
  (let [[dir file] (if (str/includes? rel "/") (str/split rel #"/") [nil rel])
        base (str/lower-case (str/replace file #"\.edn$" ""))]
    (symbol (str "memo." (when dir (str dir ".")) base "-test"))))

(defn- fixtures-with-sizes []
  (let [root (io/file "test/fixtures")]
    (for [f (file-seq root)
          :when (str/ends-with? (.getName f) ".edn")
          :let [rel (str/replace (str (.relativize (.toPath root) (.toPath f))) "\\" "/")
                {:keys [sizes]} (edn/read-string (slurp f))]
          :when (seq sizes)]
      [rel sizes])))

(def ^:private not-twins
  "Twins written apart from their Python model on purpose, so their sizes
  differ by design."
  {["testpy.edn" :exotic-param-4]
   "Python's model returns x; the twin returns (f x), which reaches the same runtime error in memo.clj"})

(defn- comparisons
  "{:compared [{:where :python :memo-clj}] :unmatched [where]}: each recorded
  size against the twin of the same name, when that twin is a memo (a twin of
  a model that does not compile defines none)."
  []
  (reduce (fn [acc [rel sizes]]
            (let [ns-sym (fixture-namespace rel)]
              (require ns-sym)
              (reduce (fn [acc [model size]]
                        (let [m (some-> (ns-resolve ns-sym (symbol (name model))) deref)]
                          (if (and (memo/memo? m) (not (not-twins [rel model])))
                            (update acc :compared conj {:where [rel model] :python size
                                                        :memo-clj (:memo.impl/size (meta m))})
                            (update acc :unmatched conj [rel model]))))
                      acc sizes)))
          {:compared [] :unmatched []}
          (fixtures-with-sizes)))

(deftest size-matches-python-memo
  (let [{:keys [compared unmatched]} (comparisons)
        wrong (remove #(= (:python %) (:memo-clj %)) compared)]
    (println (format "size @: %d models compared, %d agree; %d recorded sizes have no memo twin"
                     (count compared) (- (count compared) (count wrong)) (count unmatched)))
    (is (< 250 (count compared)) "the corpus is compared, not a handful")
    (is (empty? wrong) (str (count wrong) " differ, e.g. " (pr-str (take 12 wrong))))))

;; -----------------------------------------------------------------------------
;; time @ and space @

(def X [0 1])

(defmemo const-memo [] [] (return 3))

(defmemo rec [[x X]] [a (b 0)]
  (return (if (> a 0) (rec [x] :a (- a 1) :b (+ b 1)) b)))

(defmemo twice [[x X]] [a] (return (+ (rec [x] :a a) (rec [x] :a a))))
(defmemo once-and-x [[x X]] [a] (return (+ (rec [x] :a a) x)))

(defn plain [x] (* 2 x))
(defn ^{:memo/cost 50} pricey [x] (* 2 x))
(defn ^{:memo/cost -1} negative-cost [x] x)
(defmemo calls-plain [[x X]] [] (return (plain x)))
(defmemo calls-pricey [[x X]] [] (return (pricey x)))
(defmemo calls-negative-cost [[x X]] [] (return (negative-cost x)))

;; no parameter: memo.lower computes all of it once, as constants
(defmemo all-constant [] [] (alice :chooses [y X] :wpp 1) (return (E alice.y)))

(def A [0 1])
(def Y [0 1 2])
(defn utility [a y] (if (= a y) 1.0 0.0))
(defmemo pick [[a A]] [beta]
  (alice :thinks (bob :chooses [y Y] :wpp (+ 1 y)))
  (alice :chooses [a A] :wpp (exp (* beta (E (utility a bob.y)))))
  (return (Pr (= alice.a a))))

(defmemo deep [] [a] (return (if (> a 0) (cost @(deep :a (- a 1))) 7)))
(defmemo times-plain [] [] (return (time @(plain 1))))
(defmemo measures [] [a] (return (+ (* 0 (time @(rec :a a))) (size @(rec :a a)))))

(defn- t [m & args] (:time (apply memo/cost m args)))

(defn- message
  "The message of the memo error thunk throws, or nil."
  [thunk]
  (try (thunk) nil (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(deftest a-constant-is-free
  (is (= 0 (t const-memo)))
  (is (pos? (:space (memo/cost const-memo))) "its one value still takes memory"))

(deftest a-recursion-is-linear-in-its-depth
  (let [base (t rec 0)]
    (is (pos? base))
    (doseq [k [1 2 3 5]]
      (is (= (* (inc k) base) (t rec k)) (str "depth " k)))))

(deftest a-memo-call-counts-its-callee-per-call-site
  ;; the same program but for the second operand: a second call of rec, or x
  (doseq [a [0 2]]
    (is (= (t rec a) (- (t twice a) (t once-and-x a))) (str "a = " a))))

(deftest a-host-function-counts-its-memo-cost-per-cell
  (is (= (* 2 49) (- (t calls-pricey) (t calls-plain))) "two cells, 50 each against 1")
  (is (= "The host function negative-cost's :memo/cost is -1, not a non-negative integer"
         (message #(t calls-negative-cost)))))

(deftest constants-count-whatever-the-cache-holds
  (let [before (memo/cost all-constant)]
    (all-constant)                                       ; computes and caches its constants
    (is (= before (memo/cost all-constant)))
    (is (pos? (:time before)))
    (is (pos? (:space before)))))

(deftest a-budget-raises-the-count-with-k
  (let [k #(t pick 1.0 {:memo/budget {'alice {:samples %}}})]
    (is (< (t pick 1.0) (k 1)) "the bounded program does more than the exact one")
    (is (< (k 1) (k 2) (k 4)))))

(deftest inside-a-memo
  (is (= 7.0 (double (ts/host (deep 0)))) "the guard false: the else branch")
  (is (number? (ts/host (deep 3))) "a recursion through cost @ ends at its guard")
  (is (= (double (:size (memo/cost rec 2))) (double (ts/host (measures 2)))))
  (is (= (memo/cost rec 2) (memo/cost rec {:a 2})) "by position or by name"))

(def ^:private tabulated (atom 0))
(defn counting-weight [_] (swap! tabulated inc) 1.0)
(defmemo tempered [[c X]] [beta depth]
  (alice :chooses [c X] :wpp (exp (* beta (counting-weight c))))
  (return (+ (Pr (= alice.c c)) (rec [c] :a depth))))

(deftest a-fit-does-not-rerun-for-values-that-cannot-change-the-count
  (memo/cost tempered 0.1 2)
  (let [n @tabulated]
    (is (= (memo/cost tempered 0.1 2) (memo/cost tempered 0.7 2) (memo/cost tempered 3.0 {:depth 2})))
    (is (= n @tabulated) "beta reaches no guard and no memo call: no run, the cached count")
    (is (not= (memo/cost tempered 0.1 2) (memo/cost tempered 0.1 4)) "depth reaches a memo call")
    (is (< n @tabulated) "so a new depth runs")))

(deftest its-errors-are-loud
  (is (= "parameter not statically known"
         (ex-message (ts/compile-error 'memo.cost-test
                                       '(defmemo bad-arg [] []
                                          (alice :chooses [y X] :wpp 1)
                                          (return (E (cost @(rec alice.y)))))))))
  (is (= "(cost @(f ...)) measures the whole memo, without axes"
         (ex-message (ts/compile-error 'memo.cost-test '(defmemo bad-axes [] [] (return (cost @(rec [0] 1))))))))
  (is (= "Unknown memo plain" (message #(times-plain))) "time @ of a host function")
  (is (= "1 is not a memo" (message #(memo/cost 1)))))
