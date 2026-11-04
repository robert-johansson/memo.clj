(ns memo.synth.report-test
  "The run report: counts per task, the reasoning among the correct
  episodes, medians; entries without :reasoning regraded from their
  programs (repl1's)."
  (:require [clojure.test :refer [deftest is testing]]
            [memo.synth.check-test :as fixtures]
            [memo.synth.report :as report]))

(deftest summarize-counts-and-medians
  (let [rows (report/summarize
              [{:task :sally :outcome :correct :reasoning :as-asked :turns 10 :tokens 100 :wall-seconds 5.0}
               {:task :sally :outcome :correct :reasoning :shortcut :turns 30 :tokens 300 :wall-seconds 15.0}
               {:task :sally :outcome :wrong :reasoning :shortcut :turns 20 :tokens 200 :wall-seconds 10.0}
               {:task :coin :outcome :missing :turns 30 :tokens 50 :wall-seconds 2.0}])
        by (into {} (map (juxt :task identity)) rows)]
    (testing "the suite's order, then the total"
      (is (= [:coin :sally :total] (map :task rows))))
    (testing "reasoning counted among the correct only"
      (is (= {:n 3 :correct 2 :as-asked 1 :shortcut 1 :unchecked 0}
             (select-keys (by :sally) [:n :correct :as-asked :shortcut :unchecked]))))
    (testing "medians over all episodes"
      (is (= [20 200 10.0] ((juxt :turns :tokens :seconds) (by :sally))))
      (is (= 4 (:n (by :total))))
      (is (= 25.0 (:turns (by :total)))))))

(deftest entries-without-reasoning-are-regraded
  (is (= :no-model (report/reasoning-of {:task :sally :program fixtures/sally-shortcut}))
      "sally's task asks for marble too")
  (is (= :shortcut (report/reasoning-of
                    {:task :sally
                     :program (str fixtures/sally-shortcut "
(defmemo marble [[l Loc]] [p]
  (anne :chooses [moved Bool] :wpp (if moved p (- 1 p)))
  (return (Pr (= (if anne.moved Loc.BOX Loc.BASKET) l))))")})))
  (is (= :as-asked (report/reasoning-of {:task :coin :program fixtures/coin-genuine})))
  (is (= :as-asked (report/reasoning-of {:task :coin :reasoning :as-asked :program fixtures/monty-shortcut}))
      "a recorded :reasoning is kept")
  (is (nil? (report/reasoning-of {:task :coin})) "no program, nothing to regrade"))
