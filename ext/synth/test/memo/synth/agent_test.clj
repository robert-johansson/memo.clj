(ns memo.synth.agent-test
  "The driver's notes (no LLM): the countdown, and the progress note that
  names the task's memos not defined yet (repl1's alarm tested host
  functions for 30 turns without a defmemo)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.synth.agent :as agent]
            [memo.synth.tools :as tools]))

(deftest notes
  (let [asked (atom 0)
        missing (fn [ms] #(do (swap! asked inc) ms))]
    (testing "at half the turns and at ten left, the memos not defined yet"
      (is (= "15 turns left, and alarm is not defined yet. Write the defmemo now and look at it with table; test host functions through it."
             (agent/notes 30 15 (missing ['alarm]))))
      (is (str/starts-with? (agent/notes 30 10 (missing ['sally 'marble])) "10 turns left, and sally and marble are not"))
      (is (nil? (agent/notes 30 15 (missing [])))) "nothing when every memo is defined")
    (testing "the host is asked only at those turns"
      (reset! asked 0)
      (is (every? nil? (for [left (range 29 5 -1) :when (not (#{15 10} left))]
                         (agent/notes 30 left (missing ['alarm])))))
      (is (zero? @asked)))
    (testing "the countdown in the last five turns"
      (is (= "5 turns left: submit your model soon." (agent/notes 30 5 (missing ['alarm]))))
      (is (= "1 turn left: submit your model now." (agent/notes 30 1 (missing [])))))
    (testing "a capped episode (3 turns): the budget submits, so no call to submit"
      (is (= "2 turns left; when they run out, the model you have defined is submitted."
             (agent/notes 3 2 (missing ['alarm]))))
      (is (= "1 turn left; when they run out, the model you have defined is submitted."
             (agent/notes 3 1 (missing [])))))))

(deftest status-names-the-missing-memos
  (let [{id :session} (tools/open! :sally)]
    (is (= #{'sally 'marble} (set (:missing (tools/status! id)))))
    (tools/eval! id "(defenum Loc BASKET BOX)
(defmemo marble [[l Loc]] [p]
  (anne :chooses [moved Bool] :wpp (if moved p (- 1 p)))
  (return (Pr (= (if anne.moved Loc.BOX Loc.BASKET) l))))")
    (is (= ['sally] (:missing (tools/status! id))))
    (tools/eval! id "(def sally 3)")
    (is (= ['sally] (:missing (tools/status! id))) "a var that is not a memo is missing")
    (tools/close! id)))

(deftest resume-skips-what-is-done
  (let [f (java.io.File/createTempFile "episodes" ".edn")]
    (try
      (testing "no file: nothing done"
        (.delete f)
        (is (= #{} (agent/done-episodes f 2))))
      (testing "this seed's entries, and entries without a seed, count; another seed's do not"
        (spit f (str (pr-str {:task :coin :episode 0}) "\n"
                     (pr-str {:task :coin :episode 1 :seed 2}) "\n\n"
                     (pr-str {:task :act :episode 0 :seed 3}) "\n"))
        (is (= #{[:coin 0] [:coin 1]} (agent/done-episodes f 2)))
        (is (= #{[:coin 0] [:act 0]} (agent/done-episodes f 3))))
      (testing "a line cut short is a loud error naming it"
        (spit f (str (pr-str {:task :coin :episode 0 :seed 2}) "\n{:task :act :epis") :append true)
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"line 6 does not read" (agent/done-episodes f 2))))
      (finally (.delete f)))))

(deftest a-failed-episode-is-graded-and-the-run-goes-on
  (let [calls (atom []) entries (atom [])
        call (fn [req] (swap! calls conj (:op req))
               (case (:op req)
                 :open {:session "s"}
                 :submit {:text "Submitted: missing" :grading {:task :coin :outcome :missing :reasoning :no-model}
                          :program "(def x 1)"}
                 {:text "ok"}))
        ;; agent/run as it fails on a turn cut inside its think block: open, throw, close in finally
        fake-run (fn [_ spec _]
                   (let [st ((:open spec))]
                     (try (throw (ex-info "cut inside its think block"
                                          {:reason :cut-in-thinking :log [{:turn 1 :reply "" :calls []}]}))
                          (finally ((:close spec) st)))))]
    (with-redefs [gen.llm.agent/run fake-run]
      (agent/run-task nil (first (filter #(= :coin (:id %)) memo.synth.tasks/tasks))
                      {:episodes 2 :turns 30 :turn-tokens 8192 :thinking? true :seed 2 :n-ctx 131072}
                      call #(swap! entries conj %) #{}))
    (is (= 2 (count @entries)) "the second episode ran after the first failed")
    (is (every? #(= :cut-in-thinking (:stop %)) @entries))
    (is (every? #(= :missing (:outcome %)) @entries) "graded on what was defined")
    (is (= "(def x 1)" (:program (first @entries))))
    (is (= [:open :submit :open :submit] @calls) "each failed episode submitted once, in close")))
