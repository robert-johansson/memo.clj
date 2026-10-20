(ns memo.fuzz-test
  "The differential fuzzer's findings, replayed without Python
  (test/fixtures/fuzz.edn, Python memo's outcomes), and a live campaign
  that needs Python memo (skipped, loudly, without it)."
  (:require [clojure.test :refer [deftest is testing]]
            [memo.fuzz :as fuzz]
            [memo.python :as py]
            [memo.test-support :as ts]))

(deftest regressions
  (let [{:keys [regressions]} (ts/fixture "fuzz.edn")]
    (is (= (set (map :id fuzz/regressions)) (set (map :id regressions))) "the fixture is current")
    (doseq [r regressions]
      (testing (str (name (:id r)) ": " (:why r))
        (let [[ok ours] (fuzz/replay r)]
          (is ok (str "memo.clj " (pr-str ours) "\npython " (pr-str (:python r)))))))))

(deftest ^:slow campaign
  (if-not (py/available?)
    (println "\nSKIPPED memo.fuzz-test/campaign: Python memo is not available (memo-mcp/venv)\n")
    (let [{:keys [counts disagreements]} (fuzz/campaign {:n 300 :seed 20261020})]
      (println "\nfuzz campaign, 300 programs:" (pr-str counts) "\n")
      (is (empty? disagreements) (pr-str (map #(select-keys % [:form :how]) disagreements))))))
