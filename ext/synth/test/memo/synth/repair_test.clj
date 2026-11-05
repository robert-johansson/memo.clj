(ns memo.synth.repair-test
  "Unbalanced code at the REPL (memoclj-vako): where the delimiters break,
  and the three repair policies (indent, indent-show, refuse)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.synth.tools :as tools]))

(deftest the-break-is-located
  (is (nil? (tools/delimiter-problem "(defn f [x] (+ x 1))\n(f 2)")) "balanced")
  (testing "an opener never closed: the innermost, and how many"
    (let [p (tools/delimiter-problem "(defmemo m [] []\n  (alice :thinks\n    (bob :chooses [x X] :uniformly))\n  (return 1)")]
      (is (= :unclosed (:kind p)))
      (is (= [1 1 1] [(:line p) (:column p) (:open p)]))))
  (testing "a closer that closes nothing"
    (let [p (tools/delimiter-problem "(f 1))\n(g 2)")]
      (is (= [:unmatched 1 6] [(:kind p) (:line p) (:column p)]))))
  (testing "a closer of the wrong kind"
    (let [p (tools/delimiter-problem "(let [a 1)\n  a)")]
      (is (= [:mismatch 1 10] [(:kind p) (:line p) (:column p)]))
      (is (str/includes? (:text p) "closes the [ opened at line 1, column 6"))))
  (testing "strings, comments and characters are not delimiters"
    (is (nil? (tools/delimiter-problem "(str \")(\" \\( ) ; a ) in a comment\n(f)")))
    (is (= 3 (:line (tools/delimiter-problem "(str \"a\nb\")\n(g (h)"))) "a string spanning lines keeps the count")))

(def unbalanced
  "Indented as if bob's statement were alice's, one closer short in the middle."
  "(def X [0 1 2])\n(defmemo m [] []\n  (alice :thinks\n    (bob :chooses [x X] :uniformly)\n  (alice :chooses [y X] :wpp 1)\n  (return 1))")

(deftest the-policies
  (doseq [[policy expect] [[:indent :repaired] [:indent-show :shown] [:refuse :refused]]]
    (testing (name policy)
      (let [{id :session} (tools/open! :coin)
            r (binding [tools/*repair-policy* policy] (tools/eval! id unbalanced))]
        (is (= policy (:repair-policy r)))
        (case expect
          :repaired (do (is (:repaired? r)) (is (str/includes? (:text r) "rebuilt from the indentation"))
                        (is (not (str/includes? (:text r) "The code as it ran"))))
          :shown (do (is (:repaired? r)) (is (str/includes? (:text r) "The code as it ran, rebuilt:")))
          :refused (do (is (not (:repaired? r))) (is (= :read (:kind r)))
                       (is (str/includes? (:text r) "the delimiters do not balance: the ( at line"))))
        (tools/close! id)))))
