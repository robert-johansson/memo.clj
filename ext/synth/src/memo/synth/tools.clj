(ns memo.synth.tools
  "memo.clj's REPL tools for an LLM (memoclj-pcw0), hosted in a JVM of their
  own (gen.codegen.process): a session per episode (gen.codegen.session)
  under memo.clj's prelude, evaluated with memo's hooks, and the tools the
  LLM calls on it. Requests, each {:op …}:

    :open {:task}               → {:text :session}
    :eval {:session :code}      → {:text :kind}, kind :ok, :memo, :clojure,
                                   :read, :refused or :timeout
    :table {:session :call}     → {:text}: a memo call's table, axes labeled
    :check {:session}           → {:text :passed?}: the task's memos'
                                   signatures and ranges, no answers
    :submit {:session}          → {:text :grading :program}; the session closed
    :close {:session}

  Run it as clojure -M:host -m memo.synth.tools (the driver starts it)."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.codegen.process :as process]
            [gen.codegen.session :as session]
            [memo.core :as memo]
            [memo.domains :as domains]
            [memo.errors :as err]
            [memo.synth.check :as check]
            [memo.synth.tasks :as tasks]))

(def ^:private by-id (delay (into {} (map (juxt :id identity)) (tasks/with-expected tasks/tasks))))
(defonce ^:private sessions (atom {}))

(def prelude
  '[(:require [gen.array :as array]
              [memo.core :as memo :refer [defmemo]]
              [memo.domains :as domains :refer [defenum defdomain]])])

(defn- clip [s n] (if (> (count s) n) (str (subs s 0 n) "…") s))

(defn- r4 [x]
  (cond (boolean? x) x
        (integer? x) x
        (number? x) (let [d (double x)] (if (Double/isFinite d) (Double/parseDouble (format "%.4g" d)) d))
        (sequential? x) (mapv r4 x)
        :else x))

(defn show
  "A value as the LLM sees it: numbers to 4 significant digits, arrays as
  their values (or their shape when large), a memo by its signature, a var
  (what def, defn, defenum and defmemo return) by its name and value."
  [v]
  (cond (var? v) (str "defined " (.sym ^clojure.lang.Var v) ": " (show @v))
        (domains/enum? v) (str "an enum: " (str/join ", " (map #(str %1 "=" %2) (domains/members v) (:values v))))
        (memo/memo? v) (let [m (meta v)]
                         (str "<memo " (:memo/name m) ", axes " (:memo/axes m) ", params " (:memo/params m) ">"))
        (array/backend-array? v) (let [x (r4 (array/->clj (array/stop-gradient v)))]
                                   (if (and (sequential? x) (> (count (flatten [x])) 60))
                                     (str "<array of shape " (vec (array/shape v)) ">")
                                     (pr-str x)))
        (number? v) (pr-str (r4 v))
        (fn? v) "<function>"
        :else (clip (pr-str v) 400)))

(defn- root-cause [^Throwable e] (loop [e e] (if-let [c (.getCause e)] (recur c) e)))

(defn explain
  "An evaluation error as the LLM reads it: memo's error as memo prints it
  (message, place, hint), or Clojure's, with a hint for what memo cannot
  see: an enum member outside a defmemo, which Clojure reads as a class."
  [e ns _forms]
  (if-let [me (err/root-memo-error e)]
    (err/format-error me)
    (let [c (root-cause e)
          msg (str (ex-message c))
          member (when (instance? ClassNotFoundException c)
                   (when-let [[_ d m] (re-matches #"([\w-]+)\.([\w-]+)" msg)]
                     (let [v (or (some-> (ns-resolve ns (symbol d)) deref)
                                 (some-> (domains/builtins (symbol d)) deref))]
                       (when (domains/enum? v) [d m]))))]
      (str (.getSimpleName (class c)) ": " msg
           (when-let [[d m] member]
             (str "\nhint: outside a defmemo, Clojure reads " d "." m " as a Java class name. In a defn or "
                  "def, write (" d " :" m ") for its value, or compare with the integer; inside a defmemo "
                  d "." m " works."))))))

(defn- repair [code]
  ((requiring-resolve 'parinferish.core/flatten) ((requiring-resolve 'parinferish.core/parse) code {:mode :indent})))

(def hooks
  {:repair repair
   :vet (fn [forms] (when-let [bad (check/unsafe forms)]
                      (str "refused: " bad " is not allowed here. "
                           (if (and (namespace bad) (re-matches #"[A-Z][\\w-]*" (namespace bad)))
                             (str "If " (namespace bad) " is an enum, write (" (namespace bad) " :" (name bad)
                                  ") for its member's value outside a defmemo, and " (namespace bad) "." (name bad)
                                  " inside one.")
                             "Use def, defn, defenum, defmemo and calls of pure functions; no IO, interop, eval or require."))))
   :explain explain
   :show show
   :each? true
   :timeout-ms 60000})

(defn- entry [id] (or (@sessions id) (throw (ex-info (str "no session " id) {}))))

(defn- kind [{:keys [ok? text timeout?]}]
  (cond ok? :ok
        timeout? :timeout
        (re-find #"(?m)^(error: )?refused" (str text)) :refused
        (re-find #"memo error" (str text)) :memo
        (re-find #"the code does not read|EOF while reading|Unmatched delimiter" (str text)) :read
        :else :clojure))

(defn open! [task-id]
  (let [task (or (@by-id task-id) (throw (ex-info (str "no task " task-id) {})))
        s (session/open {:prelude prelude :prefix "memo.synth.repl-"})
        id (str (:ns s))]
    (swap! sessions assoc id {:session s :task task})
    {:text "Session open: defmemo, defenum, defdomain, array, memo and domains are available." :session id}))

(defn eval!
  "code evaluated in the session; memo's warnings (printed to *err* when a
  defmemo compiles) come back with the result."
  [id code]
  (let [{s :session} (entry id)
        w (java.io.StringWriter.)
        r (binding [*err* w] (session/eval! s code hooks))
        warnings (str/trim (str w))]
    {:text (str (:text r) (when (seq warnings) (str "\nmemo's warnings:\n" (clip warnings 1200))))
     :kind (kind r) :repaired? (:repaired? r)}))

(defn table! [id call]
  (let [{s :session} (entry id)
        forms (session/read-forms call)
        head (when (and (= 1 (count forms)) (seq? (first forms))) (first (first forms)))
        m (when (symbol? head) (session/lookup s head))]
    (if-not (memo/memo? m)
      {:text (str "table takes one call of a memo you defined, e.g. (forecast 0.8 1.0 0.5); "
                  (if head (str head " is not a memo") (str "got " call)))}
      (if-let [bad (check/unsafe forms)]
        {:text (str "refused: " bad " is not allowed here.") :kind :refused}
        ;; evaluated, not recorded: a look at a table is no part of the program
        (let [n (the-ns (:ns s))
              v (try (session/within 60000 #(binding [*ns* n] (eval (first forms))))
                     (catch Throwable e e))]
          (if (instance? Throwable v)
            {:text (str "error: " (explain v n forms)) :kind (if (err/root-memo-error v) :memo :clojure)}
            (let [rows (memo/rows m v)
                  axes (:memo/axes (meta m))
                  k (keyword (str (:memo/name (meta m))))]
              {:text (str/join "\n" (for [row rows]
                                      (str (str/join ", " (map #(str % "=" (get row (keyword (str %)))) axes))
                                           (when (seq axes) " → ") (pr-str (r4 (get row k))))))
               :kind :ok})))))))

(defn check! [id]
  (let [{s :session task :task} (entry id)
        lines (for [[sym {:keys [params axes shape]}] (:signatures task)
                    :let [m (session/lookup s sym)]]
                (cond
                  (nil? m) (str sym ": not defined yet")
                  (not (memo/memo? m)) (str sym ": not a memo")
                  :else
                  (let [mm (meta m)
                        problems (cond-> []
                                   (not= (mapv str params) (mapv str (:memo/params mm)))
                                   (conj (str "its parameters are " (:memo/params mm) ", the task asks for " params))
                                   (not= (count axes) (count (:memo/axes mm)))
                                   (conj (str "it has " (count (:memo/axes mm)) " axes, the task asks for " (count axes)))
                                   (and (= (count axes) (count (:memo/axes mm))) (not= shape (:memo/shape mm)))
                                   (conj (str "its table's shape is " (:memo/shape mm) ", the task's domains give " shape)))
                        tests (for [c (:calls task) :when (= sym (:memo c))]
                                (let [r (check/call-memo (:ns s) sym (:args c) 60000)]
                                  (cond (:kind r) (str "calling (" sym " " (str/join " " (:args c)) ") fails: " (:message r))
                                        (not-every? #(Double/isFinite %) (flatten [(:value r)]))
                                        (str "(" sym " " (str/join " " (:args c)) ") has values that are not finite")
                                        (some #(or (< % -1e-9) (> % (+ 1 1e-9))) (flatten [(:value r)]))
                                        (str "(" sym " " (str/join " " (:args c)) ") has values outside [0, 1]: a probability?"))))
                        problems (into problems (remove nil? tests))]
                    (if (empty? problems)
                      (str sym ": signature as asked; runs at the task's test parameters; values finite and in [0, 1]")
                      (str sym ": " (str/join "; " problems))))))]
    {:text (str/join "\n" lines)
     :passed? (every? #(str/includes? % "signature as asked") lines)}))

(defn submit! [id]
  (let [{s :session task :task} (entry id)
        results (for [c (:calls task)]
                  (let [r (check/call-memo (:ns s) (:memo c) (:args c) 120000)]
                    (cond (:kind r) {:call [(:memo c) (:args c)] :outcome (:kind r) :message (:message r)}
                          (check/close? (:value r) (:expected c) 1e-3) {:call [(:memo c) (:args c)] :outcome :correct}
                          :else {:call [(:memo c) (:args c)] :outcome :wrong :value (r4 (:value r))})))
        outcome (or (some #(when (not= :correct (:outcome %)) (:outcome %)) results) :correct)
        program (session/program s)]
    (session/close! s)
    (swap! sessions dissoc id)
    {:text (str "Submitted: " (name outcome)) :grading {:task (:id task) :outcome outcome :results (vec results)}
     :program program}))

(defn close! [id]
  (when-let [{s :session} (@sessions id)] (session/close! s) (swap! sessions dissoc id))
  {:text "closed"})

(defn handle [{:keys [op task session code call]}]
  (session/within 300000
                  #(case op
                     :open (open! task)
                     :eval (eval! session code)
                     :table (table! session call)
                     :check (check! session)
                     :submit (submit! session)
                     :close (close! session))))

(defn -main [& _]
  (process/serve handle)
  (shutdown-agents)
  (System/exit 0))
