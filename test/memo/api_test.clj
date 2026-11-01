(ns memo.api-test
  "What 1.0 freezes (DESIGN.md §1, the Handbook's \"Stability\"): the public
  vars of memo.core, memo.domains, memo.errors and memo.gen, without those
  marked ^:no-doc (internal); a memo's metadata keys under :memo/; a memo
  error's ex-data keys under :memo/. A change to any of them fails here, so
  that it is made on purpose: additions in a minor release, removals and
  renames only in a new major version. memo.clj's source uses only gen.clj's
  stable surface, so that gen.clj's own freeze covers what memo.clj needs."
  (:require [clojure.edn :as edn]
            [clojure.java.basis :as basis]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.gen]))

(def frozen
  ;; cost: added for 1.1 (M4's cost reflection, DESIGN.md §14.1)
  {'memo.core #{'defmemo 'memo? 'print-table 'program 'cost}
   'memo.domains #{'Bool 'defdomain 'defenum 'domain? 'enum? 'field 'field-keys 'fields 'labels
                   'members 'pack 'product-domain 'product? 'values}
   'memo.errors #{'*warnings?* 'format-error 'memo-error? 'root-memo-error}
   ;; estimated and log-estimate-variance are experimental (M3), outside the freeze
   'memo.gen #{'deterministic 'dist 'estimated 'log-estimate-variance}})

(defn- api [ns-sym]
  (into #{} (comp (remove (fn [[_ v]] (:no-doc (meta v)))) (map key)) (ns-publics ns-sym)))

(deftest the-public-vars-are-the-frozen-ones
  (doseq [[ns-sym vars] frozen]
    (is (= vars (api ns-sym)) (str ns-sym))))

(deftest every-public-var-has-a-docstring
  (doseq [ns-sym (keys frozen), v (api ns-sym)]
    (is (:doc (meta (ns-resolve ns-sym v))) (str ns-sym "/" v))))

(deftest experimental-vars-say-so
  (doseq [v ['estimated 'log-estimate-variance]]
    (is (.startsWith ^String (:doc (meta (ns-resolve 'memo.gen v))) "Experimental (M3)") (str v))))

(defenum Coin T H)

(defmemo flip [[c Coin]] [p]
  "A coin."
  (alice :chooses [c Coin] :wpp (if (== c 1) p (- 1 p)))
  (return (Pr (= alice.c c))))

(deftest a-memo's-metadata-under-memo-is-the-stable-set
  (is (= #{:memo/name :memo/doc :memo/params :memo/axes :memo/domain-names :memo/domains
           :memo/shape :memo/num-results}
         (set (filter #(= "memo" (namespace %)) (keys (meta flip))))))
  (testing "the var carries the docstring and the arguments"
    (is (= "A coin." (:doc (meta #'flip))))
    (is (= '([p] [p named]) (:arglists (meta #'flip))))))

(defn- gen-clj-names
  "The gen.clj namespaces and vars the source file f names, as [ns var]
  pairs, var nil for a namespace it requires: read as forms (no strings or
  comments), with f's namespace's aliases, and its refers."
  [f]
  (with-open [r (java.io.PushbackReader. (io/reader f))]
    (let [the-ns (the-ns (second (read r)))
          gen? #(str/starts-with? (str %) "gen.")
          forms (binding [*ns* the-ns *read-eval* false]
                  (doall (take-while #(not= ::eof %) (repeatedly #(read {:eof ::eof} r)))))
          ns-of (fn [s] (when-let [q (namespace s)]
                          (if-let [a (get (ns-aliases the-ns) (symbol q))] (ns-name a) (symbol q))))]
      (concat
       (for [a (vals (ns-aliases the-ns)) :when (gen? (ns-name a))] [(ns-name a) nil])
       (for [v (vals (ns-refers the-ns)) :let [n (ns-name (:ns (meta v)))] :when (gen? n)]
         [n (:name (meta v))])
       (for [s (tree-seq coll? seq forms) :when (and (symbol? s) (gen? (ns-of s)))]
         [(ns-of s) (symbol (name s))])))))

(deftest memo.clj-uses-only-gen.clj's-stable-surface
  ;; gen.clj's snapshot of its stable surface, from the gen.clj in use (the
  ;; pin's checkout, or the working copy under :dev)
  (let [root (get-in (basis/current-basis) [:libs 'gen.clj/gen.clj :deps/root])
        snapshot (edn/read-string (slurp (io/file root "test/fixtures/api-snapshot.edn")))]
    (doseq [f (file-seq (io/file "src")) :when (str/ends-with? (.getName f) ".clj")
            [n v] (distinct (gen-clj-names f))]
      (is (if v (contains? (get snapshot n) v) (contains? snapshot n))
          (str f ": " n (when v (str "/" v)) " is not in gen.clj's stable surface")))))

(deftest a-memo-error's-data-under-memo-is-the-stable-set
  (let [e (try (flip 0.5 {"p" 0.5}) nil (catch Exception e (err/root-memo-error e)))]
    (is (err/memo-error? e))
    (is (= #{:memo/error :memo/hint :memo/loc :memo/frame-ctxt}
           (set (filter #(= "memo" (namespace %)) (keys (ex-data e))))))))
