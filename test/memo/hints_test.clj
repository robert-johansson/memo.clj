(ns memo.hints-test
  "memo.clj's own hints beside Python memo's messages (DESIGN principle 6),
  from the LLM REPL runs (memoclj-uhkq): a memo with an agent named
  observer, memo's name for the top level too, gets a note on each of its
  errors. The message stays Python memo's, and so does its hint."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(defenum Taste SWEET SOUR)
(defenum Fruit APPLE BANANA CHERRY)
(defn util [t f] (nth (nth [[1.0 2.0 0.0] [2.0 0.0 1.0]] t) f))

;; Python memo accepts an agent named observer
(defmemo taste-obs [[f Fruit] [t Taste]] [beta]
  (observer :thinks (child :chooses [tt Taste] :uniformly)
                    (child :chooses [fruit Fruit] :wpp (exp (* beta (util tt fruit)))))
  (observer :observes child.fruit :is f)
  (observer :knows t)
  (return (as observer (Pr (= child.tt t)))))

(defn- hint
  "[message hint] of the memo error form raises."
  [form]
  (let [e (ts/compile-error 'memo.hints-test form)]
    (is e "the form is a compile error")
    [(ex-message e) (str (:memo/hint (ex-data e)))]))

(def python-unknown-choice
  "Python memo's hint for a choice who does not have, which stays."
  "Did you perhaps misspell")

(deftest an-agent-named-observer
  (testing "accepted, as Python memo accepts it, with Python memo's numbers"
    ;; Python memo 1.3.3 (memo-mcp, 2026-11-04), taste_obs(1.0)
    (is (ts/close? [[0.26894146 0.73105854] [0.8807971 0.11920292] [0.26894143 0.73105854]]
                   (ts/host (taste-obs 1.0)))))
  (testing "its errors get the note, Python memo's message and hint kept"
    (let [[m h] (hint '(defmemo taste-axis [[f Fruit] [t Taste]] [beta]
                         (observer :thinks (child :chooses [tt Taste] :uniformly)
                                           (child :chooses [fruit Fruit] :wpp (exp (* beta (util tt fruit)))))
                         (observer :observes child.fruit :is f)
                         (return (as observer (Pr (= child.tt t))))))]
      (is (= "Unknown choice observer.t" m) "Python memo's message")
      (is (str/starts-with? h python-unknown-choice))
      (is (str/includes? h "an agent named observer, which is also memo's name for the top level"))
      (is (str/includes? h "(observer :knows x)"))))
  (testing "a memo without one gets no note"
    (let [[_ h] (hint '(defmemo no-observer [[t Taste]] []
                         (obs :thinks (child :chooses [tt Taste] :uniformly))
                         (return (as obs (Pr (= child.tt t))))))]
      (is (not (str/includes? h "agent named observer"))))))
