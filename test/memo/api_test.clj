(ns memo.api-test
  "What 1.0 freezes (DESIGN.md §1, the Handbook's \"Stability\"): the public
  vars of memo.core, memo.domains, memo.errors and memo.gen, without those
  marked ^:no-doc (internal); a memo's metadata keys under :memo/; a memo
  error's ex-data keys under :memo/. A change to any of them fails here, so
  that it is made on purpose: additions in a minor release, removals and
  renames only in a new major version."
  (:require [clojure.test :refer [deftest is testing]]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.gen]))

(def frozen
  {'memo.core #{'defmemo 'memo? 'print-table 'program}
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

(deftest a-memo-error's-data-under-memo-is-the-stable-set
  (let [e (try (flip 0.5 {"p" 0.5}) nil (catch Exception e (err/root-memo-error e)))]
    (is (err/memo-error? e))
    (is (= #{:memo/error :memo/hint :memo/loc :memo/frame-ctxt}
           (set (filter #(= "memo" (namespace %)) (keys (ex-data e))))))))
