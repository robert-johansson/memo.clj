(ns memo.handbook-doc-test
  "HANDBOOK.md is executable. Every ```clojure block in it runs, in order,
  in one namespace (the first block's ns form names it), with output and
  warnings discarded. Three markers make a block a check:

  - `;=> value` after a form: the form's result, read back as Clojure data
    (gen.array arrays by gen.array/->clj, booleans as 1.0 and 0.0), equals
    value, which is read, not evaluated. Numbers compare within float32
    tolerance (memo.test-support/close?), so the same text passes on every
    backend.
  - `;; error: message` as a block's first line: one of the block's forms
    throws, and the memo error inside (memo.errors/root-memo-error) has a
    message beginning with message. The forms after it do not run.
  - a line `prints` right after a block, then a ```text block: what the
    block printed, its lines with trailing spaces trimmed."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(def handbook (slurp (io/file "HANDBOOK.md")))

(defn blocks
  "The clojure blocks of a Markdown text: [line text printed], line being
  the line of the opening fence, and printed the text block that follows
  a line `prints`, or nil."
  [text]
  (let [m (re-matcher #"(?ms)^```clojure\n(.*?)^```\n(?:\nprints\n\n```text\n(.*?)^```\n)?" text)]
    (loop [out []]
      (if (.find m)
        (recur (conj out [(inc (count (re-seq #"\n" (subs text 0 (.start m))))) (.group m 1) (.group m 2)]))
        out))))

(defn- lines [s] (mapv str/trimr (str/split-lines (str/trim s))))

(defn- read-back [v]
  (cond (number? v) v
        (boolean? v) (if v 1.0 0.0)
        (array/array? v) (read-back (array/->clj v))
        (map? v) (into {} (map (fn [[k x]] [k (read-back x)])) v)
        (sequential? v) (mapv read-back v)
        :else v))

(defn- same? [expected actual]
  (cond
    (and (number? expected) (number? actual)) (ts/close? expected actual)
    (and (sequential? expected) (sequential? actual))
    (and (= (count expected) (count actual)) (every? true? (map same? expected actual)))
    (and (map? expected) (map? actual))
    (and (= (set (keys expected)) (set (keys actual)))
         (every? #(same? (get expected %) (get actual %)) (keys expected)))
    :else (= expected actual)))

(defn- forms
  "A block's forms, each with the values its ;=> markers expect."
  [text]
  (let [rdr (clojure.lang.LineNumberingPushbackReader.
             (java.io.StringReader. (str/replace text ";=>" " :handbook/=> ")))
        next-form #(read {:eof ::eof} rdr)]
    (loop [out []]
      (let [form (next-form)]
        (cond
          (= ::eof form) out
          (= :handbook/=> form) (recur (update-in out [(dec (count out)) :expect] (fnil conj []) (next-form)))
          :else (recur (conj out {:form form})))))))

(defn- error-marker [text]
  (second (re-find #"^;; error: (.*)" text)))

(defn- run-block
  "Evaluates a block's forms; the number of checks it made."
  [line text printed]
  (testing (str "HANDBOOK.md, the block at line " line)
    (let [fs (forms text)]
      (cond
        printed
        (let [out (with-out-str (doseq [{:keys [form]} fs] (eval form)))]
          (is (= (lines printed) (lines out)) (str "it printed\n" out))
          1)

        (error-marker text)
        (let [message (error-marker text)
              e (try (doseq [{:keys [form]} fs] (eval form)) nil (catch Throwable e e))
              m (some-> e err/root-memo-error)]
          (is (some? e) "no form threw")
          (is (some? m) (str "not a memo error: " (some-> e ex-message)))
          (is (some-> m ex-message (str/starts-with? message)) (str "the message is " (some-> m ex-message)))
          1)

        :else
        (reduce (fn [n {:keys [form expect]}]
                  (let [v (try (eval form) (catch Throwable e e))]
                    (if (instance? Throwable v)
                      (do (is false (str (pr-str form) " threw: " (ex-message v))) (reduced n))
                      (do (doseq [x expect]
                            (is (same? (read-back x) (read-back v))
                                (str (pr-str form) "\nexpected " (pr-str x) "\nactual   " (pr-str (read-back v)))))
                          (+ n (count expect))))))
                0 fs)))))

(deftest the-handbook-runs
  (let [bs (blocks handbook)
        first-form (:form (first (forms (second (first bs)))))
        ns-name (when (and (seq? first-form) (= 'ns (first first-form))) (second first-form))]
    (is (some? ns-name) "the first block is an ns form")
    (when ns-name (remove-ns ns-name))
    (let [checks (binding [*ns* (the-ns 'user)
                           *out* (java.io.StringWriter.)
                           *err* (java.io.StringWriter.)
                           err/*warnings?* false]
                   (reduce + (for [[line text printed] bs] (run-block line text printed))))]
      (is (> (count bs) 60) "HANDBOOK.md has its blocks")
      (is (> checks 70) "and its checks"))))
