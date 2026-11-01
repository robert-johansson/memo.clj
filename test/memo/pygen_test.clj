(ns memo.pygen-test
  "The transpiler (memo.pygen) and Python memo from Clojure (memo.python).
  The round trip checks the transpiler against Python memo itself: the
  corpus's twins, transpiled and run in Python memo, reproduce the
  fixtures' values, which came from the Python originals. Needs Python
  memo (memo-mcp); without it, the round trip is skipped, loudly."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
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
  {:inspect-alice "defined by eval (to capture what :inspect prints)"
   :inline-agent "must not compile in memo.clj"})

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

;; -----------------------------------------------------------------------------
;; memo.pygen/source, for users (M4, DESIGN.md §14.5; experimental)

(defenum Side LEFT RIGHT)
(defn tilt [x] (* 0.25 x))
(defmemo lean [[s Side]] [p]
  (alice :chooses [s Side] :wpp (if (= s Side.LEFT) (+ p (tilt 1)) (- 1 p)))
  (return (Pr (= alice.s s))))
(defn ^:memo/vectorized tilt-v [x] x)

;; names Python cannot keep apart, or has its own
(defn big? [x] (> x 1))
(defn big-qm [x] x)
(def side_p 0.5)
(defmemo local-clash [] [side-p] (return (+ side-p #{side_p})))
(defn pad [x] x)
(defmemo pads [[s Side]] [] (return (pad s)))
(defmemo lambda-param [] [lambda] (return (* 2 lambda)))
(defmemo halved [] [n] (return (/ n 2)))
(defmemo calls-halved [] [] (return (halved [] 2)))

(defn- message [thunk]
  (try (thunk) nil (catch Exception e (some-> (err/root-memo-error e) ex-message))))

(deftest source-for-users
  (let [forms (filterv #(and (seq? %) (#{'Side 'tilt 'lean} (second %)))
                       (read-forms "test/memo/pygen_test.clj"))
        python (pygen/source [#'lean #'Side #'tilt])]
    (is (= (:python (pygen/transpile-forms (sort-by #(case (str (first %)) "defenum" 0 "defn" 1 2) forms)
                                           'memo.pygen-test))
           python)
        "the transpiler's module, its forms read from this file, in Python's order")
    (is (< (str/index-of python "class Side(IntEnum)") (str/index-of python "def tilt")
           (str/index-of python "def lean")))
    (testing "run in Python memo, it gives memo.clj's numbers"
      (if-not (py/available?)
        (println "Skipping memo.pygen/source in Python memo: no Python memo (memo-mcp)")
        (let [dir (doto (io/file (System/getProperty "java.io.tmpdir") (str "memo-pygen-" (System/nanoTime))) .mkdirs)
              f (io/file dir "lean_module.py")]
          (spit f (str "import sys\nsys.path.insert(0, " (pr-str (.getAbsolutePath (io/file py/memo-src))) ")\n"
                       python "\nimport json\nprint(json.dumps(lean(0.3).tolist()))\n"))
          (let [{:keys [exit out err]} (sh/sh py/venv-python "-I" (.getPath f))]
            (is (zero? exit) err)
            (is (ts/close? (edn/read-string (last (str/split-lines out))) (ts/host (lean 0.3))))))))))

(deftest source-refuses-loudly
  (is (= "memo.pygen/source cannot translate the vectorized function tilt-v"
         (message #(pygen/source [#'lean #'tilt-v]))))
  (is (= "memo.pygen/source takes vars of one namespace, not 2"
         (message #(pygen/source [#'lean #'clojure.core/inc]))))
  (is (= "memo.pygen/source takes vars" (message #(pygen/source ['lean]))))
  (testing "two names that become one Python name"
    (is (= "memo.pygen/source would give big-qm and big? one Python name, big_qm"
           (message #(pygen/source [#'big? #'big-qm]))))
    (is (= "memo.pygen/source would give side-p and side_p one Python name, side_p"
           (message #(pygen/source [#'local-clash #'side_p])))
        "a model's parameter and a var"))
  (testing "a name Python memo's models import (memo.lib's pad), passed or only read"
    (is (= "memo.pygen/source cannot translate pad: Python memo's models read their own pad"
           (message #(pygen/source [#'pads #'Side #'pad]))))
    (is (= "memo.pygen/source cannot translate pad: Python memo's models read their own pad"
           (message #(pygen/source [#'pads #'Side])))))
  (testing "a memo call without axes: Python memo has none"
    (is (= "memo.pygen/source cannot translate calls-halved"
           (message #(pygen/source [#'calls-halved #'halved])))))
  (testing "a Python keyword as a name"
    (is (= "memo.pygen/source cannot translate lambda: lambda is a Python keyword"
           (message #(pygen/source [#'lambda-param])))))
  (testing "a file edited since it was loaded: the var's line holds another form"
    (let [line (:line (meta #'lean))]
      (try
        (alter-meta! #'lean assoc :line (:line (meta #'tilt)))
        (is (= "The file memo/pygen_test.clj has changed since memo.pygen-test/lean was loaded"
               (message #(pygen/source [#'lean #'Side #'tilt]))))
        (finally (alter-meta! #'lean assoc :line line))))))

(deftest source-of-a-file-loaded-by-its-path
  ;; clojure -M -i models.clj and load-file give a var its file's path, not a
  ;; classpath resource
  (let [dir (doto (io/file (System/getProperty "java.io.tmpdir") (str "memo-pygen-load-" (System/nanoTime))) .mkdirs)
        f (io/file dir "loaded_models.clj")]
    (spit f (str "(ns memo.pygen-loaded-models (:require [memo.core :refer [defmemo]]))\n"
                 "(def Z [0 1])\n"
                 "(defmemo zf [[z Z]] [] (return (* 2 z)))\n"))
    (load-file (.getAbsolutePath f))
    (let [python (pygen/source [(resolve 'memo.pygen-loaded-models/zf) (resolve 'memo.pygen-loaded-models/Z)])]
      (is (str/includes? python "def zf[z: Z]():"))
      (is (str/includes? python "Z = ")))))
