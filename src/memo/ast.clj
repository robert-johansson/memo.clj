(ns memo.ast
  "memo's AST. Nodes are maps, mirroring the dataclasses of Python memo's
  core.py (lifted from memo-clj after review).

  Expressions {:node … :static bool :loc {…}}:
    :lit :op :ffi :memo :choice :expect :entropy :kl :with :imagine :inline
    :posterior :predict :util
  Statements {:node … :loc {…}}:
    :pass :choose :observe :show :observes-factor :with-stmt :forall :knows
    :snapshot :variant :trace :wants :guess :forgets-about

  Agent names and choice ids are symbols; 'self is the reflexive agent name."
  (:require [clojure.walk :as walk]))

(def expr-nodes
  #{:lit :op :ffi :memo :choice :expect :entropy :kl :with :imagine :inline
    :posterior :predict :util})

(def stmt-nodes
  #{:pass :choose :observe :show :observes-factor :with-stmt :forall :knows
    :snapshot :variant :trace :wants :guess :forgets-about})

(def binary-ops
  #{:add :sub :mul :div :pow :mod :eq :neq :lt :lte :gt :gte :and :or :xor})

(def unary-ops #{:exp :abs :log :uadd :neg :inv})

(defn collect
  "External references of an AST: {:ffi-names #{} :memo-names #{} :domain-names #{}}."
  [ast]
  (let [acc (atom {:ffi-names #{} :memo-names #{} :domain-names #{}})]
    (walk/postwalk
     (fn [x]
       (when (map? x)
         (case (:node x)
           :ffi (swap! acc update :ffi-names conj (:name x))
           :memo (swap! acc update :memo-names conj (:name x))
           :choose (swap! acc update :domain-names into (map second (:choices x)))
           :forall (swap! acc update :domain-names conj (:domain x))
           nil))
       x)
     ast)
    @acc))
