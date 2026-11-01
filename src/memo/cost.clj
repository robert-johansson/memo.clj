(ns memo.cost
  "Cost reflection (DESIGN.md §14.1, M4): the measures Python memo reads with
  cost @, time @, space @ and size @. Internal; the language reads them
  through (cost @(f ...)) and its siblings, the host through memo.core.

  size: the number of nodes of a memo's syntax tree, counted as Python memo
  1.3.3's codegen.weigh counts its dataclasses (exact parity, checked against
  every model of the corpus: test/memo/cost_test.clj). A node counts 1, and
  each of its fields: a nested node by its own count, a list by its items (a
  node by its count, anything else 1), a dict by its keys and values, any
  other value 1. A source location is a dataclass of 4 fields, so 5; none,
  1. memo.clj's own keys (:surface…, :ellipsis?, an inline's :idx, an enum
  member's :member) are not Python's fields, and do not count. Two readings
  differ from Clojure's: Python parses -2 as unary minus over 2, which
  counts (negative-literal?), and +x as unary plus, which memo.parse keeps."
  {:no-doc true}
  (:require [memo.ast :as ast]))

(def ^:private python-fields
  "Each node's fields in Python memo's dataclasses (core.py), by memo.clj's
  keys, besides :loc (every node) and :static (every expression)."
  {;; expressions
   :lit [:value]
   :op [:op :args]
   :ffi [:name :args :kwargs]
   :memo [:name :which-retval :args :ids :kwargs]
   :choice [:id]
   :expect [:expr :reduction :warn]
   :entropy [:rvs]
   :kl [:p-who :p-id :q-who :q-id]
   :with [:who :expr]
   :imagine [:do :then]
   :cost [:name :args :kwargs :metric]
   :inline [:val]
   :posterior [:query :var]
   :predict [:expr]
   :util [:goal]
   ;; statements
   :pass []
   :choose [:who :choices :wpp :reduction]
   :observe [:who :id]
   :with-stmt [:who :stmt]
   :show [:who :target-who :target-id :source-who :source-id]
   :observes-factor [:who :what :how]
   :forall [:id :domain]
   :knows [:who :source-who :source-id]
   :snapshot [:who :alias]
   :variant [:who :variant]
   :trace [:who]
   :wants [:who :what :how]
   :guess [:who :id :target-who :target-id]
   :forgets-about [:who :ids]})


(defn- node? [x] (and (map? x) (contains? x :node)))

(declare weigh)

(defn- weigh-field
  "A field's count, as weigh counts a dataclass's field."
  [k v]
  (cond
    (node? v) (weigh v)
    ;; a dict of keyword arguments: its keys and values (memo.clj keeps it as
    ;; a map or as [name expr] pairs)
    (= k :kwargs) (reduce + 0 (map (fn [[_ x]] (+ 1 (if (node? x) (weigh x) 1))) v))
    (sequential? v) (reduce + 0 (map #(if (node? %) (weigh %) 1) v))
    (map? v) (reduce + 0 (map (fn [[_ x]] (+ 1 (if (node? x) (weigh x) 1))) v))
    :else 1))

(defn- negative-literal?
  "A literal Python reads as unary minus over a literal: -2 is
  UnaryOp(USub, 2) in Python's AST, a number below zero in Clojure's
  reader (so is -0.0; -0 reads as 0, and cannot be told apart)."
  [{:keys [node value member]}]
  (and (= :lit node) (number? value) (not member)
       (or (neg? value)
           (and (float? value) (zero? (double value)) (neg? (Math/copySign 1.0 (double value)))))))

(defn weigh
  "A node's count, as Python memo's codegen.weigh counts its dataclass."
  [{:keys [node loc] :as n}]
  (let [fields (or (get python-fields node)
                   (throw (ex-info (str "memo.cost: no Python fields for the node " node)
                                   {:node node})))
        own (+ 1
               (if loc 5 1)
               (if (ast/expr-nodes node) 1 0)           ; :static
               (reduce + 0 (map #(weigh-field % (get n %)) fields)))]
    ;; Python's EOp(NEG) around the literal: the node, its location, :static, :op
    (if (negative-literal? n)
      (+ own 1 (if loc 5 1) 1 1)
      own)))

(defn size
  "size @'s measure of a parsed memo: its statements' and returns' counts."
  [{:keys [stmts retvals]}]
  (reduce + 0 (map weigh (concat stmts retvals))))
