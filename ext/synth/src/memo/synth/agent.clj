(ns memo.synth.agent
  "The REPL agent for memo.clj models (memoclj-pcw0): Qwen, through
  gen.llm.agent, builds a task's model in small steps with memo's tools
  (memo.synth.tools), which run in a JVM of their own (gen.codegen.process).
  Each episode is one EDN line in the output file: its stop, turns, tokens,
  seconds, the grading of what it submitted (its tables' :outcome, its
  :reasoning: :as-asked or :shortcut), its program and its log.

  Properties (synth.*): model (a GGUF), out (a directory), tasks (ids,
  comma-separated; all by default), episodes (2), turns (30; 3 for the
  ablation's arm (b): the budget submits what is defined), turn-tokens
  (1536; 8192 with thinking), thinking (false), n-ctx (32768), seed (1),
  host (the tool host's command).

  A run resumes: started again with the same out and seed, it skips the
  episodes episodes.edn already holds (done-episodes)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.codegen.process :as process]
            [gen.llm.agent :as agent]
            [gen.llm.llama :as llama]
            [gen.llm.lm :as lm]
            [gen.llm.sample :as sample]
            [gen.random :as rng]
            [memo.synth.tasks :as tasks]))

(defn- prop [k default] (or (System/getProperty (str "synth." k)) default))
(defn- clip [s n] (let [s (str s)] (if (> (count s) n) (str (subs s 0 n) "…") s)))
(defn- seconds [t0] (/ (- (System/nanoTime) t0) 1e9))

(def tool-specs
  (array-map
   "eval" {:description "Evaluate Clojure forms in your namespace: def, defn, defenum, defmemo, or a call to see its value. Returns each form's value, what was printed (e.g. by (alice :inspect) in a defmemo), memo's warnings, or the error with memo's hint."
           :params [{:name "code" :description "One or more Clojure forms." :required? true}]}
   "table" {:description "Call a memo you defined and see its table, its axes labeled."
            :params [{:name "call" :description "One call of a memo, such as (forecast 0.8 1.0 0.5)." :required? true}]}
   "check" {:description "Check the task's memos: their names, axes and parameters, that their tables are finite and in [0, 1], and that their agents reason as the task asks (agents present, beliefs read where the task asks for an agent's belief). It does not tell you the answer."
            :params []}
   "submit" {:description "Submit the model you have defined. The episode ends."
             :params []}))

(defn- tools
  "gen.llm.agent's tools, each a request to the tool host (call); the
  episode's state an atom of {:session}. A submission ends the episode with
  its grading; a lost session (the host restarted) ends it with {:lost true}."
  [call]
  (let [request {"eval" (fn [p] {:op :eval :code (p "code")})
                 "table" (fn [p] {:op :table :call (p "call")})
                 "check" (fn [_] {:op :check})
                 "submit" (fn [_] {:op :submit})}]
    (into (array-map)
          (for [[n spec] tool-specs]
            [n (assoc spec :fn (fn [params state]
                                 (let [r (call (assoc ((request n) params) :session (:session @state)))]
                                   (cond
                                     (:restarted r) (do (swap! state assoc :ended true)
                                                        {:text (str (:text r)) :done {:lost true}})
                                     (:grading r) (do (swap! state assoc :ended true)
                                                      {:text (str (:text r)) :done r
                                                       :outcome (get-in r [:grading :outcome])})
                                     :else (cond-> {:text (str (:text r))}
                                             (:kind r) (assoc :kind (:kind r))
                                             (:repaired? r) (assoc :repaired? true)
                                             (contains? r :passed?) (assoc :passed? (:passed? r)))))))]))))

(defn notes
  "The note added to a turn's tool responses, turns-left after it, or nil:
  the countdown in the last five turns; at half the turns, and at ten left,
  the task's memos not defined yet, if any (missing, a thunk asking the
  tool host). repl1 ran out of turns twice, once testing host functions
  for 30 turns without a defmemo (alarm)."
  [turns turns-left missing]
  (let [progress (when (and (> turns-left 5) (#{(quot turns 2) 10} turns-left))
                   (when-let [ms (seq (missing))]
                     (str turns-left " turns left, and " (str/join " and " ms) " "
                          (if (next ms) "are" "is") " not defined yet. Write the defmemo now and look at it "
                          "with table; test host functions through it.")))
        left (str turns-left " turn" (when (not= 1 turns-left) "s") " left")
        countdown (when (<= turns-left 5)
                    (if (<= turns 5)
                      ;; a capped episode (the ablation's arm (b), memoclj-pcw0): every turn is in
                      ;; the countdown, and the budget submits; "submit now" would spend a turn
                      (str left "; when they run out, the model you have defined is submitted.")
                      (str left ": submit your model " (if (<= turns-left 2) "now." "soon."))))]
    (or progress countdown)))

(defn done-episodes
  "The [task episode] pairs results (an episodes.edn) already holds for
  seed, so that a run resumed in the same directory skips them; their keys
  depend on the seed, the task and the episode only, so the episodes left
  draw what they would have. An entry without :seed (written before
  entries recorded it) counts as seed's. A line that does not read (a run
  killed while writing it) is a loud error: remove it, and that episode
  runs again."
  [results seed]
  (if-not (.exists (io/file results))
    #{}
    (into #{}
          (keep-indexed
           (fn [i line]
             (when-not (str/blank? line)
               (let [e (try (edn/read-string line)
                            (catch Exception x
                              (throw (ex-info (str results " line " (inc i) " does not read (" (ex-message x)
                                                   "): remove it, and that episode runs again")
                                              {:line (inc i)}))))]
                 (when (= seed (get e :seed seed)) [(:task e) (:episode e)])))))
          (str/split-lines (slurp results)))))

(defn run-task
  "episodes of the REPL agent on task, but those in done ([task episode]
  pairs); record! takes each episode's entry."
  [m task {:keys [episodes turns turn-tokens thinking? sampling seed n-ctx]} call record! done]
  (doseq [ep (range episodes) :when (not (done [(:id task) ep]))]
    (let [t0 (System/nanoTime)
          episode (atom nil)
          submit (fn [state]
                   (when-not (:ended @state)
                     (let [r (call {:op :submit :session (:session @state)})]
                       (swap! state assoc :ended true)
                       r)))
          r (agent/run m {:system tasks/repl-system-prompt
                          :task (str (:prompt task) "\n\nBuild it at the REPL with the tools, and call check before you submit.")
                          :tools (tools call)
                          :open #(reset! episode (atom {:session (:session (call {:op :open :task (:id task)}))}))
                          :close #(when-not (:ended @%) (call {:op :close :session (:session @%)}))
                          :on-budget submit
                          :notes (fn [_ left]
                                   (notes turns left
                                          #(when-not (:ended @@episode)
                                             (:missing (call {:op :status :session (:session @@episode)})))))}
                       {:turns turns :turn-tokens turn-tokens :thinking? thinking? :sampling sampling
                        :context n-ctx
                        :key (rng/fold-in (rng/key seed) (+ (* 100 (mod (hash (:id task)) 100000)) ep))})
          result (:result r)
          stop (cond (:lost result) :session-lost (= :done (:stop r)) :submitted :else (:stop r))
          calls (mapcat :calls (:log r))
          entry {:task (:id task) :level (:level task) :episode ep :thinking? thinking? :stop stop
                 :turns (:turns r) :tokens (:tokens r) :seconds (:llm (:seconds r))
                 :tool-seconds (:tools (:seconds r)) :wall-seconds (seconds t0)
                 :outcome (get-in result [:grading :outcome] (if (:lost result) :lost :not-submitted))
                 :reasoning (get-in result [:grading :reasoning])
                 :grading (:grading result) :program (:program result)
                 :tools-used (frequencies (map :tool calls))
                 :eval-kinds (frequencies (keep :kind (filter #(= "eval" (:tool %)) calls)))
                 :log (mapv (fn [e] (-> e
                                        (update :reply clip 2000)
                                        (update :reasoning clip 4000)
                                        (update :calls (partial mapv (fn [c] (-> c (update :params update-vals #(clip % 1500))
                                                                                 (update :text clip 1500)))))))
                            (:log r))}]
      (record! entry)
      (println (format "%-12s episode %d: %s, %s%s after %d turns, %d tokens, %.0f s; evals %s"
                       (name (:id task)) ep (name stop) (name (:outcome entry))
                       (if-let [rs (:reasoning entry)] (str " (" (name rs) ")") "") (:turns r) (:tokens r)
                       (double (:wall-seconds entry)) (:eval-kinds entry)))
      (flush))))

(defn -main [& _]
  (let [model (prop "model" nil)
        out (io/file (prop "out" "out"))
        thinking? (= "true" (prop "thinking" "false"))
        n-ctx (parse-long (prop "n-ctx" "32768"))
        opts {:episodes (parse-long (prop "episodes" "2"))
              :turns (parse-long (prop "turns" "30"))
              :turn-tokens (parse-long (prop "turn-tokens" (if thinking? "8192" "1536")))
              :thinking? thinking?
              :sampling (if thinking? {:temp 1.0 :top-k 20 :top-p 0.95} sample/default-sampling)
              :seed (parse-long (prop "seed" "1"))
              :n-ctx n-ctx}
        ids (set (map keyword (some-> (prop "tasks" nil) (str/split #","))))
        ts (cond->> tasks/tasks (seq ids) (filter #(ids (:id %))))
        _ (.mkdirs out)
        results (io/file out "episodes.edn")
        host (process/start (str/split (prop "host" "clojure -M:host -m memo.synth.tools") #" ")
                            {:log (str (io/file out "host.log"))})
        done (done-episodes results (:seed opts))
        record! (fn [entry] (spit results (str (pr-str (assoc entry :model model :seed (:seed opts))) "\n") :append true))]
    (println "tool host ready; model" model "tasks" (mapv :id ts) "opts" (dissoc opts :sampling))
    (when (seq done)
      (println "resuming:" (count done) "episodes of this seed already in" (str results) "are skipped:" (sort done)))
    (flush)
    (try
      (with-open [e (llama/engine {:model model :n-ctx n-ctx :max-branches 4})
                  m (lm/lm e (llama/tokenizer e) {:max-branches 4})]
        (doseq [t ts] (run-task m t opts #(process/call host % 600000) record! done)))
      (finally (process/stop host)))
    (println "done:" (str results))
    (shutdown-agents)
    (System/exit 0)))
