(ns memo.cost-test
  "Cost reflection (DESIGN.md §14.1). size @: every model of the corpus whose
  Python twin parses has the size Python memo 1.3.3's codegen.weigh gives it
  (each fixture file's :sizes, from test/fixtures/generate.py), compared on
  the Clojure twins in the namespace that fixture belongs to."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.core :as memo]))

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
