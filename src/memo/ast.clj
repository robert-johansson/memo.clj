(ns memo.ast
  "memo's AST. Nodes are maps, mirroring the dataclasses of Python memo's
  core.py (lifted from memo-clj after review).

  Expressions {:node … :static bool :loc {…}}:
    :lit :op :ffi :memo :choice :expect :entropy :kl :with :imagine :inline
    :posterior :predict :util :cost
  Statements {:node … :loc {…}}:
    :pass :choose :observe :show :observes-factor :with-stmt :forall :knows
    :snapshot :variant :trace :wants :guess :forgets-about

  Agent names and choice ids are symbols; 'self is the reflexive agent name."
  (:require [clojure.walk :as walk]))

(def expr-nodes
  #{:lit :op :ffi :memo :choice :expect :entropy :kl :with :imagine :inline
    :posterior :predict :util :cost})

(def stmt-nodes
  #{:pass :choose :observe :show :observes-factor :with-stmt :forall :knows
    :snapshot :variant :trace :wants :guess :forgets-about})

(def binary-ops
  #{:add :sub :mul :div :pow :mod :eq :neq :lt :lte :gt :gte :and :or :xor})

(def unary-ops #{:exp :abs :log :uadd :neg :inv})

(defn assign-inline-ids
  "The AST with each inline escape hatch #{form} numbered (:idx in place of
  :form), in a fixed order, and the forms: [ast forms]."
  [ast]
  (let [forms (atom [])
        ast' (walk/postwalk
              (fn [x]
                (if (and (map? x) (= :inline (:node x)) (contains? x :form))
                  (let [idx (count @forms)]
                    (swap! forms conj (:form x))
                    (-> x (dissoc :form) (assoc :idx idx)))
                  x))
              ast)]
    [ast' @forms]))

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

(defn agents
  "The agents an AST names: {:agents #{name} :snapshots {alias agent}}, self
  excluded. A snapshot's alias belongs to the agent that took it."
  [ast]
  (let [acc (atom {:agents #{} :snapshots {}})
        add #(when (and (symbol? %) (not= 'self %)) (swap! acc update :agents conj %))]
    (walk/postwalk
     (fn [x]
       (when (map? x)
         (case (:node x)
           (:choose :observes-factor :with-stmt :knows :variant :trace :wants :forgets-about :with)
           (add (:who x))
           :show (run! add [(:who x) (:target-who x) (:source-who x)])
           :snapshot (do (add (:who x)) (swap! acc assoc-in [:snapshots (:alias x)] (:who x)))
           :kl (run! add [(:p-who x) (:q-who x)])
           :entropy (run! (comp add first) (:rvs x))
           :memo (run! (comp add second) (:ids x))
           nil)
         (when (= :knows (:node x)) (add (:source-who x))))
       x)
     ast)
    (update @acc :agents #(apply disj % (keys (:snapshots @acc))))))
