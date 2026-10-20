(ns memo.pygen-test
  "The transpiler (memo.pygen) and Python memo from Clojure (memo.python).
  The round trip checks the transpiler against Python memo itself: the
  corpus's twins, transpiled and run in Python memo, reproduce the
  fixtures' values, which came from the Python originals. Needs Python
  memo (memo-mcp); without it, the round trip is skipped, loudly."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.pygen :as pygen]
            [memo.python :as py]
            [memo.test-support :as ts]))

(defn read-forms
  "The top-level forms of a source file."
  [path]
  (with-open [r (java.io.PushbackReader. (io/reader path))]
    (binding [*read-eval* false]
      (loop [acc []]
        (let [f (read {:eof ::eof} r)]
          (if (= ::eof f) acc (recur (conj acc f))))))))

(deftest emits-python-memo
  (is (= (str "@memo\n"
              "def flip[c: Coin](p):\n"
              "    alice: chooses(c in Coin, wpp=(p if c == 0 else 1 - p))\n"
              "    return Pr[alice.c == c]")
         (pygen/emit-defmemo '(defmemo flip [[c Coin]] [p]
                                (alice :chooses [c Coin] :wpp (if (= c Coin.HEADS) p (- 1 p)))
                                (return (Pr (= alice.c c))))
                             (fn [e m] (when (and (= e 'Coin) (= m 'HEADS)) 0)))))
  (is (= (str "@memo(cache=True)\n"
              "def watch(t):\n"
              "    cast: [alice, bob]\n"
              "    bob: thinks[\n"
              "        alice: chooses(x in X, wpp=1)\n"
              "    ]\n"
              "    bob: observes_that [alice.x > t]\n"
              "    return bob[E[alice.x]]")
         (pygen/emit-defmemo '(defmemo ^:cache watch [] [t]
                                (cast [alice bob])
                                (bob :thinks (alice :chooses [x X] :wpp 1))
                                (bob :observes-that (> alice.x t))
                                (return (as bob (E alice.x))))))))

;; the corpus

(defn- corpus-calls
  "A fixture's calls, as {:twin :args :kwargs :shape :value :digest}."
  [kind fx]
  (case kind
    :slice (for [{:keys [model args shape value]} (:cases fx)]
             {:twin model :args args :kwargs {} :shape shape :value value})
    :testpy (for [{:keys [name outcome args shape value digest]} (:cases fx) :when (= "pass" outcome)]
              {:twin name :args args :kwargs {} :shape shape :value value :digest digest})
    :handbook (for [t (:twins fx) :when (= "compiled" (:outcome t))
                    c (:calls t) :when (#{"call" "table"} (:kind c))]
                {:twin (:name t) :args (:args c) :kwargs (:kwargs c) :shape (:shape c) :value (:value c)})
    :demo (for [m (:models fx) :when (not= "ce" (:outcome m)) c (:calls m)]
            {:twin (get {:EU 'expected-utility} (:name m) (:name m))   ; demo-takeaway's renamed EU
             :args (:args c) :kwargs (:kwargs c) :shape (:shape c) :value (:value c) :digest (:digest c)})))

(def corpus
  (concat
   (for [i (range 1 7)]
     {:ns (symbol (str "memo.slice" i "-test")) :file (str "test/memo/slice" i "_test.clj")
      :fixture (str "slice" i ".edn") :kind :slice})
   [{:ns 'memo.testpy-test :file "test/memo/testpy_test.clj" :fixture "testpy.edn" :kind :testpy}
    {:ns 'memo.handbook-test :file "test/memo/handbook_test.clj" :fixture "handbook.edn" :kind :handbook}]
   (for [f (sort (.list (io/file "test/memo/demos"))) :when (not= f "corpus_test.clj")
         :let [stem (str/replace f #"_test\.clj$" "")]]
     {:ns (symbol (str "memo.demos." (str/replace stem "_" "-") "-test"))
      :file (str "test/memo/demos/" f)
      :fixture (str "demos/" (if (= stem "memonomicon") "Memonomicon" (str/replace stem "_" "-")) ".edn")
      :kind :demo})))

(def not-transpiled
  "Twins with no Python round trip, and why (besides those whose host
  functions the transpiler skips: Clojure that has no scalar Python form)."
  (let [cost "cost reflection (M4): Python memo's twin is there, memo.clj's is not"]
    {:cost-1 cost :cost-kwarg cost :cost-time cost :cost-time-matches-cost cost
     :cost-space cost :cost-size cost :cost-size-kwarg cost
     :quick "demo-takeaway's quick: cost reflection (M4)"
     :inspect-alice "defined by eval (to capture what :inspect prints)"
     :inline-agent "must not compile in memo.clj"}))

(defn- skipped-helper?
  "Whether a call failed in Python because it calls a host function the
  transpiler skipped."
  [skipped {:keys [outcome message]}]
  (boolean (and (= :re outcome) message
                (some #(str/includes? message (str "name '" (pygen/sanitize %) "' is not defined")) skipped))))

(defn- check-part
  "One part of the corpus round-tripped; the number of calls that ran."
  [{:keys [ns file fixture kind]}]
  (require ns)
  (let [{:keys [python skipped]} (pygen/transpile-forms (read-forms file) (the-ns ns) {:guarded? true})
        calls (vec (corpus-calls kind (ts/fixture fixture)))
        outcomes (py/run python (map (fn [c] {:model (pygen/sanitize (:twin c))
                                              :args (:args c) :kwargs (:kwargs c)})
                                     calls))]
    (count
     (for [[{:keys [twin args shape value digest]} o] (map vector calls outcomes)
           :when (testing (str ns " " (name twin) " " (pr-str args))
                   (cond
                     (not-transpiled twin) (do (is (string? (not-transpiled twin))) false)
                     (skipped-helper? skipped o) false
                     (not= :ok (:outcome o)) (do (is false (pr-str o)) false)
                     :else (do (is (= shape (:shape o)))
                               (if digest
                                 (is (ts/matches-digest? digest (:value o)))
                                 (is (ts/close? value (:value o))))
                               true)))]
       twin))))

(deftest ^:slow round-trip
  (if-not (py/available?)
    (println "\nSKIPPED memo.pygen-test/round-trip: Python memo is not available (memo-mcp/venv)\n")
    (let [ran (reduce + (map check-part corpus))]
      (println (str "\nround trip: " ran " calls of the corpus's twins ran in Python memo, as transpiled\n"))
      (is (< 300 ran) "most of the corpus transpiles"))))
