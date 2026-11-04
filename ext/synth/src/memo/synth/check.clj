(ns memo.synth.check
  "Checking a model an LLM wrote: extract its code, read it (no read-eval),
  vet it (definitions only; no IO, interop or eval), evaluate it in a fresh
  namespace, and grade its tables. Every failure is classified:

    :no-code      no code block
    :read         the reader failed (brackets, syntax)
    :unsafe       a form names something outside the allow-list
    :memo         a memo compile error (memo's message and hint)
    :clojure      another compile error (an unresolved symbol, arity, ...)
    :missing      the memo the task asks for was not defined
    :too-large    its arrays would exceed the cell limit
    :runtime      calling it threw (or timed out)
    :wrong        it ran, but a table differs from the reference"
  (:require [clojure.string :as str]
            [gen.array :as array]
            [memo.core :as memo]
            [memo.errors :as err]))

;; -----------------------------------------------------------------------------
;; Code

(defn extract-code
  "The completion's ```clojure (or bare ```) blocks, joined; an unclosed last
  block counts. nil when there is none."
  [text]
  (let [closed (map second (re-seq #"(?s)```(?:clojure|clj)?[ \t]*\r?\n(.*?)```" text))
        tail (when-let [[_ t] (re-find #"(?s)```(?:clojure|clj)?[ \t]*\r?\n((?:(?!```).)*)$" text)]
               (when (empty? closed) t))]
    (cond (seq closed) (str/join "\n" closed)
          tail tail)))

(defn read-forms
  "Every form of code, read in namespace n without read-eval."
  [code n]
  (binding [*read-eval* false, *ns* n]
    (let [rdr (clojure.lang.LineNumberingPushbackReader. (java.io.StringReader. code))]
      (loop [acc []]
        (let [f (read {:eof ::eof} rdr)]
          (if (= f ::eof) acc (recur (conj acc f))))))))

(def ^:private definitions '#{def defn defn- defenum defdomain defmemo declare})

(def ^:private denied
  "clojure.core functions that read, write, load or evaluate code, or reach
  the JVM. Qualified symbols outside allowed-ns, interop forms and java.*
  names are refused separately."
  '#{eval load load-file load-string load-reader read-string slurp spit require use import
     ns in-ns intern alter-var-root future-call pmap pcalls shutdown-agents set! new . ..
     monitor-enter monitor-exit with-redefs with-redefs-fn alter-meta! reset-meta! ns-unmap
     remove-ns create-ns resolve ns-resolve requiring-resolve var-set find-var the-ns all-ns
     ns-interns ns-publics locking add-watch add-tap tap> proxy reify deftype defrecord
     definterface defprotocol extend extend-type extend-protocol gen-class gen-interface})

(def ^:private allowed-ns
  #{"array" "gen.array" "Math" "memo" "memo.core" "domains" "memo.domains" "clojure.core"})

(defn- unsafe-symbol? [s]
  (let [nm (name s), nsp (namespace s)]
    (or (and (nil? nsp) (contains? denied s))
        (and nsp (not (contains? allowed-ns nsp)))
        (and nsp (contains? denied (symbol nm)))
        (and (> (count nm) 1) (or (str/starts-with? nm ".") (str/ends-with? nm ".")))
        (str/starts-with? nm "java.")
        (str/starts-with? nm "clojure.java"))))

(defn- statement-head?
  "Is x a memo statement, (agent :keyword ...)? Its head is an agent's name."
  [x]
  (and (seq? x) (symbol? (first x)) (keyword? (second x))))

(defn- symbols
  "Every symbol of a form, the heads of memo statements left out."
  [form]
  (cond (statement-head? form) (mapcat symbols (rest form))
        (symbol? form) [form]
        (coll? form) (mapcat symbols form)
        :else nil))

(defn unsafe
  "The first unsafe symbol of any of the forms, or nil (the REPL's vetting:
  calls are allowed, the denied names and interop are not)."
  [forms]
  (some #(when (unsafe-symbol? %) %) (mapcat symbols forms)))

(defn vet
  "[kept dropped unsafe]: the definition forms, the other top-level forms
  (ns, require, calls: dropped, not run), and the first unsafe symbol."
  [forms]
  (let [defn? #(and (seq? %) (contains? definitions (first %)))
        kept (filterv defn? forms)
        bad (some #(when (unsafe-symbol? %) %) (mapcat symbols kept))]
    [kept (remove defn? forms) bad]))

;; -----------------------------------------------------------------------------
;; Evaluation

(def ^:private counter (atom 0))

(defn fresh-ns
  "A namespace with clojure.core and memo.clj's usual aliases."
  []
  (let [n (create-ns (symbol (str "memo.synth.sample" (swap! counter inc))))]
    (binding [*ns* n]
      (refer-clojure)
      (require '[gen.array :as array]
               '[memo.core :as memo :refer [defmemo]]
               '[memo.domains :as domains :refer [defenum defdomain]]))
    n))

(defn- root-cause [^Throwable e]
  (loop [e e] (if-let [c (.getCause e)] (recur c) e)))

(defn- failure [kind ^Throwable e]
  (if-let [me (err/root-memo-error e)]
    {:kind :memo :message (ex-message me) :hint (:memo/hint (ex-data me))
     :formatted (try (err/format-error me) (catch Throwable _ nil))}
    (let [c (root-cause e)]
      {:kind kind :message (str (.getSimpleName (class c)) ": " (ex-message c))})))

(defn evaluate
  "Evaluates the definitions in n. {:ok true :warnings s} or a failure."
  [n forms]
  (let [w (java.io.StringWriter.)]
    (binding [*err* w]
      (loop [[f & more] forms]
        (if (nil? f)
          {:ok true :warnings (str w)}
          (let [r (try (binding [*ns* n] (eval f)) nil
                       (catch Throwable e (failure :clojure e)))]
            (if r (assoc r :warnings (str w) :form (pr-str (take 2 f))) (recur more))))))))

;; -----------------------------------------------------------------------------
;; Calls and grading

(def cell-limit 20000000)

(defn- cells
  "An upper bound on a memo's largest array: the product of its dims' sizes."
  [m]
  (let [instrs (:instrs (:memo.impl/compiled (meta m)))
        sizes (into {} (keep (fn [{:keys [i dim values size]}]
                               (case i :domain [dim (count values)] :ones [dim size] nil)))
                    instrs)]
    (reduce * 1 (vals sizes))))

(defn- host [x]
  (cond (boolean? x) (if x 1.0 0.0)
        (number? x) (double x)
        (sequential? x) (mapv host x)
        :else x))

(defn call-memo
  "Calls the memo var sym in n with args (timeout ms): {:value v} or a failure."
  [n sym args timeout]
  (let [v (ns-resolve n sym)]
    (cond
      (nil? v) {:kind :missing :message (str "no var " sym)}
      (not (memo/memo? @v)) {:kind :missing :message (str sym " is not a memo")}
      (> (cells @v) cell-limit) {:kind :too-large :message (str (cells @v) " cells")}
      :else
      (let [f (future (try {:value (host (array/->clj (array/stop-gradient (apply @v args))))}
                           (catch Throwable e (failure :runtime e))))
            r (deref f timeout ::timeout)]
        (if (= r ::timeout)
          (do (future-cancel f) {:kind :runtime :message (str "timed out after " timeout " ms")})
          r)))))

(defn close?
  [a b tol]
  (cond (and (number? a) (number? b)) (<= (Math/abs (- (double a) (double b))) tol)
        (and (sequential? a) (sequential? b)) (and (= (count a) (count b))
                                                   (every? true? (map #(close? %1 %2 tol) a b)))
        :else false))

(defn check
  "Checks a completion's text against a task's calls ([{:memo :args :expected}]).
  The result: {:outcome :correct} or {:outcome kind ...}, with :values and
  :dropped (top-level forms not run)."
  [text {:keys [calls tol] :or {tol 1e-3}}]
  (let [code (extract-code text)]
    (if-not code
      {:outcome :no-code}
      (let [n (fresh-ns)
            forms (try (read-forms code n) (catch Throwable e (assoc (failure :read e) ::read true)))]
        (if (map? forms)
          {:outcome :read :message (:message forms)}
          (let [[kept dropped bad] (vet forms)
                base {:dropped (mapv #(pr-str (take 2 %)) dropped) :n-forms (count kept)}]
            (if bad
              (merge base {:outcome :unsafe :message (str "unsafe symbol " bad)})
              (let [ev (evaluate n kept)]
                (if-not (:ok ev)
                  (merge base {:outcome (:kind ev)} (dissoc ev :kind))
                  (loop [[c & more] calls, values []]
                    (if (nil? c)
                      (merge base {:outcome :correct :values values :warnings (:warnings ev)})
                      (let [r (call-memo n (:memo c) (:args c) 120000)]
                        (cond
                          (:kind r) (merge base {:outcome (:kind r) :values values} (dissoc r :kind))
                          (close? (:value r) (host (:expected c)) tol) (recur more (conj values (:value r)))
                          :else (merge base {:outcome :wrong :values (conj values (:value r))
                                             :call (select-keys c [:memo :args])
                                             :expected (:expected c)}))))))))))))))
