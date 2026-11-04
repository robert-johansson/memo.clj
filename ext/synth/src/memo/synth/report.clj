(ns memo.synth.report
  "A run's episodes as a table (memoclj-pcw0): per task, how many episodes,
  how many are correct by their tables, and of those how many reason as the
  task asks (the grader's :reasoning: as-asked, shortcut, unchecked); the
  medians of turns, tokens and seconds. Entries written before the grader
  checked the reasoning (repl1) are regraded from their programs.

    clojure -M:host -m memo.synth.report NAME=episodes.edn ...

  prints one table per run, then the as-asked programs of each run, for
  reading by hand (as-asked is an upper bound: any agent reading beliefs
  counts, and the answer need not come from it)."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [memo.core :as memo]
            [memo.synth.check :as check]
            [memo.synth.tasks :as tasks]
            [memo.synth.tools :as tools]))

(defn read-episodes
  "The entries of an episodes.edn."
  [file]
  (into [] (comp (remove str/blank?) (map edn/read-string)) (str/split-lines (slurp file))))

(def ^:private by-id (delay (into {} (map (juxt :id identity)) (tasks/with-expected tasks/all-tasks))))

(defn regrade
  "The :reasoning of an entry's program, by today's shape checks: the
  program evaluated in a fresh namespace (its definitions only), each of
  the task's memos checked. nil without a program."
  [{:keys [task program]}]
  (when program
    (let [t (@by-id task)
          n (check/fresh-ns)
          [kept] (check/vet (check/read-forms (str program) n))
          _ (binding [*out* (java.io.StringWriter.) *err* (java.io.StringWriter.)] (check/evaluate n kept))]
      (tools/reasoning
       (vec (for [[sym sig] (:signatures t)
                  :let [v (ns-resolve n sym)]
                  p (if (and v (memo/memo? @v))
                      (check/shape sym @v (select-keys sig [:beliefs]))
                      [{:kind :missing}])]
              (assoc p :memo sym)))))))

(defn reasoning-of
  "An entry's :reasoning, regraded when it has none (and a program)."
  [entry]
  (or (:reasoning entry) (regrade entry)))

(defn structure-of
  "An entry's structure grading (:as-intended, :not-built), computed from
  its program when it has none; nil for a task that requires none."
  [{:keys [task program grading]}]
  (or (:structure grading)
      (let [sigs (:signatures (@by-id task))
            st (into {} (for [[sym sig] sigs :when (:structure sig)]
                          [sym (check/structure (str program) sym (:structure sig))]))]
        (when (seq st) (if (every? #{:as-intended} (vals st)) :as-intended :not-built)))))

(defn- median [xs]
  (when (seq xs)
    (let [v (vec (sort xs)), n (count v)]
      (if (odd? n) (v (quot n 2)) (/ (+ (v (dec (quot n 2))) (v (quot n 2))) 2.0)))))

(defn summarize
  "Per task (in the suite's order, then a total): {:task :n :correct
  :as-asked :shortcut :unchecked :turns :tokens :seconds}, the reasoning
  counted among the correct episodes, the medians over all of them."
  [entries]
  (let [entries (map #(assoc % :reasoning (reasoning-of %) :structure (structure-of %)) entries)
        row (fn [task es]
              (let [correct (filter #(= :correct (:outcome %)) es)
                    r (frequencies (map :reasoning correct))]
                {:task task :n (count es) :correct (count correct)
                 :as-asked (r :as-asked 0) :shortcut (r :shortcut 0) :unchecked (r :unchecked 0)
                 :structure (when (some :structure es) (count (filter #(= :as-intended (:structure %)) correct)))
                 :turns (median (map :turns es)) :tokens (median (map :tokens es))
                 :seconds (median (keep #(some-> (:wall-seconds %) double) es))}))
        groups (group-by :task entries)]
    (conj (vec (for [t (map :id tasks/all-tasks) :when (groups t)] (row t (groups t))))
          (row :total entries))))

(defn table
  "summarize's rows as a markdown table."
  [rows]
  (let [fmt (fn [x] (cond (nil? x) "-" (float? x) (format "%.0f" x) :else (str x)))]
    (str/join "\n"
              (concat ["| task | n | correct | as-asked | shortcut | unchecked | structure built | turns | tokens | s |"
                       "|---|---|---|---|---|---|---|---|---|---|"]
                      (for [{:keys [task n correct as-asked shortcut unchecked structure turns tokens seconds]} rows]
                        (str "| " (name task) " | " n " | " correct " | " as-asked " | " shortcut " | " unchecked
                             " | " (fmt structure) " | " (fmt turns) " | " (fmt tokens) " | " (fmt seconds) " |"))))))

(defn -main [& specs]
  (doseq [spec specs]
    (let [[nm file] (if (str/includes? spec "=") (str/split spec #"=" 2) [spec spec])
          entries (read-episodes file)]
      (println (str "\n## " nm " (" (count entries) " episodes; reasoning among the correct; medians over all)\n"))
      (println (table (summarize entries)))))
  (doseq [spec specs]
    (let [[nm file] (if (str/includes? spec "=") (str/split spec #"=" 2) [spec spec])]
      (println (str "\n## " nm ": the as-asked programs, to read by hand\n"))
      (doseq [e (read-episodes file)
              :when (and (= :correct (:outcome e)) (= :as-asked (reasoning-of e)))]
        (println (str "### " (name (:task e)) " " (:episode e) "\n```clojure\n" (str/trim (str (:program e))) "\n```\n")))))
  (shutdown-agents))
