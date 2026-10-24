(ns memo.gen
  "memo.clj's bridge to gen.clj (M2; DESIGN.md §6): a memo as a gen.clj
  distribution, or as a deterministic generative function.

  (defenum Opt PASS FIX FLEE)
  (defmemo choose [[a Opt]] [beta]
    (agent :chooses [o Opt] :wpp (exp (* beta o)))
    (return (Pr (= agent.o a))))

  (def choice (dist choose))          ; a distribution constructor, as d/categorical is
  (gen [beta] (trace :a (choice beta)))   ; a draw is a value of Opt

  A mind's own choices are summed out exactly, as memo does: the trace holds
  only the draw. The table is checked at every call, log-prob is the log of
  the drawn cell, and a NaN gradient reaching a parameter is a loud error.

  A bounded mind (M3, experimental; memo.bounded) passes its budget in the
  call's trailing map, {:memo/budget ...}: exact over its samples, dist and
  deterministic take it as they take any call. Under a Monte Carlo budget
  the table is an estimate, and estimated gives the generative function whose
  density it estimates (gen.custom/estimated), for pseudo-marginal MCMC."
  (:refer-clojure :exclude [+ - * / < > <= >= == abs])
  (:require [gen.array :as array :refer [+ - * < > ==]]
            [gen.array.dtype :as dt]
            [gen.array.shape :as sh]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.dist.core :as dc :refer [defdist]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [memo.bounded :as bounded]
            [memo.domains :as domains]
            [memo.errors :as err]))

(defn- memo-meta
  "model's metadata, which must be a memo's."
  [who model]
  (let [m (meta model)]
    (when-not (:memo.impl/call m)
      (err/raise (str who " takes a memo (defmemo), not " (pr-str model))
                 {:hint (str "(" who " model) wraps a model defined with defmemo.")}))
    m))

;; ---------------------------------------------------------------------------
;; The NaN guard (DESIGN.md §13.10)
;; ---------------------------------------------------------------------------

(defn- guard
  "x, a parameter of memo mname, through an identity whose gradient rule
  checks the gradient arriving for it: NaN there is a loud error. Values
  that carry no gradient (host values, integers) pass as they are."
  [mname pname x]
  (if (and (array/array? x) (not (array/host? x)) (= :float (dt/kind (array/dtype x))))
    ((array/custom-vjp
      identity
      (fn [_ _ g]
        (when (array/any-true? (array/isnan g))
          (err/raise (str "NaN gradient with respect to parameter " pname " of memo " mname)
                     {:hint (str "The memo's values can be right while this gradient is NaN, as JAX's is "
                                 "(DESIGN.md §2.4). The usual cause is (exp (* alpha (log p))) where p "
                                 "is 0: log 0 is -inf, and the chain rule multiplies 0 by -inf. "
                                 "(** p alpha) has the same values and a finite gradient. Or an observed "
                                 "value has probability 0, so its log-prob is -Inf.")
                      :loc {:name mname}}))
        [g]))
     x)
    x))

(defn- guard-args
  "A memo call's arguments (positional, then perhaps a map by name), each
  parameter through the guard."
  [mname pnames args]
  (let [[pos kw] (if (map? (last args)) [(butlast args) (last args)] [args nil])
        pos (into [] (map-indexed (fn [i x] (guard mname (get pnames i i) x))) pos)]
    (cond-> pos
      kw (conj (into {} (map (fn [[k v]] [k (guard mname (name k) v)])) kw)))))

;; ---------------------------------------------------------------------------
;; The distribution
;; ---------------------------------------------------------------------------

(defn- eps
  "The machine epsilon of a float dtype."
  [dtype]
  (case dtype
    :float16 9.765625E-4
    :bfloat16 0.0078125
    :float32 (double (Math/ulp (float 1.0)))
    (Math/ulp 1.0)))

(defn- where-given
  "The given axes' values, for a message."
  [given values]
  (when (seq given)
    (str " for " (apply str (interpose ", " (map (fn [a v] (str a " = " v)) given values))))))

(defn- check-table!
  "probs, a memo's table over its sampled axes, flattened on its last axis
  (with leading batch axes in a batched run): no NaN, no negative cell, and
  each row summing to 1 within 1e-6 + n·eps (DESIGN.md §6.4). Every particle
  is read at once, also on a lazy backend, where it forces evaluation."
  [mname probs n given given-values]
  (let [at (where-given given given-values)
        dist-hint (str "A memo used as a distribution returns probabilities over the axes it samples, "
                       "such as (Pr (= alice.a a)) over [a A].")]
    (when (array/any-true? (array/isnan probs))
      (err/raise (str "The table of memo " mname " has NaN" at)
                 {:hint dist-hint :loc {:name mname}}))
    (when (array/any-true? (< probs 0))
      (err/raise (str "The table of memo " mname " has a negative cell" at ": it is not a distribution")
                 {:hint dist-hint :loc {:name mname}}))
    (let [sums (array/sum probs {:axis -1})
          tol (+ 1e-6 (* n (eps (array/dtype probs))))]
      (when (array/any-true? (> (array/abs (- sums 1.0)) tol))
        (err/raise (str "The table of memo " mname " does not sum to 1" at
                        (when (zero? (array/ndim sums)) (str ": it sums to " (array/item sums))))
                   {:hint (str "memo returns an all-zero table silently when an observation is impossible "
                               "or every :wpp weight is zero (DESIGN.md §2.4); a distribution cannot "
                               "be drawn from it. " dist-hint)
                    :loc {:name mname}})))))

(defdist ^:private memo-table
  "The distribution of a memo's table (dist's): probs over its n cells, the
  last axis; labels the cells' values, [n] for one sampled axis and [n k] for
  k; info, host data: the memo's name, k, and the support."
  [probs labels ^:host info]
  (sample [key shape]
    (let [idx (rng/categorical key (array/log probs) (sh/broadcast shape (pop (array/shape probs)))
                               {:backend (dc/backend-of probs)})]
      (array/take labels idx 0)))
  (log-prob [v]
    (let [{:keys [k model axes]} info
          v (dc/->array v)
          eq (if (= k 1)
               (== labels (array/expand-dims v -1))
               (array/all (== labels (array/expand-dims v -2)) {:axis -1}))]
      (when (array/any-true? (array/logical-not (array/any eq {:axis -1})))
        (err/raise (str "A value of memo " model "'s distribution that is not in its domain: "
                        (pr-str (array/->clj v)))
                   {:hint (str "Its values are those of its sampled axes' domains (" (apply str (interpose ", " axes))
                               "), not their indices: an enum member's value, or an element of a vector domain."
                               (when (clojure.core/> k 1) (str " With " k " sampled axes, a value is the vector of their values.")))
                    :loc {:name model}}))
      ;; the drawn cell, picked, then logged: another cell's zero has no gradient
      (array/log (array/sum (array/where eq probs 0.0) {:axis -1}))))
  (support [] (:support info))
  (discrete? [] true)
  (value-shape [] (if (= (:k info) 1) [] [(:k info)])))

(defn- cells
  "The tuples of values of the cells of axes with these value lists, in
  row-major order."
  [value-lists]
  (reduce (fn [acc vs] (vec (for [a acc v vs] (conj a v)))) [[]] value-lists))

(defn- index-in
  "The index of host value x in values, or nil."
  [values x]
  (first (keep-indexed (fn [i v] (when (clojure.core/== v x) i)) values)))

(defn- check-options
  "opts, a builder's options map: only the keys allowed, a loud error
  otherwise (a misspelt option would be ignored)."
  [who model opts allowed]
  (when-not (or (nil? opts) (map? opts))
    (err/raise (str who " takes its options as a map, not " (pr-str opts))))
  (when-let [k (first (remove allowed (keys opts)))]
    (err/raise (str who ": unknown option " (pr-str k))
               {:hint (str "Its options are " (apply str (interpose ", " (sort allowed))) ".")
                :loc {:name (:memo/name (meta model))}})))

(defn- trailing-options
  "The call options in args' trailing map, or nil."
  [args]
  (when (map? (last args)) (select-keys (last args) [:memo/budget :memo/limit :memo/key])))

(defn- distribution-of
  "From a memo's arguments (the given axes' values first, as a vector, when
  opts has :given) to its distribution (memo-table): the construction of
  dist, checked once; the result a function of args and the extra call
  options (a key) to merge into their trailing map."
  [who model {:keys [given] :as opts}]
  (let [m (memo-meta who model)
        _ (check-options who model opts #{:given :return})
        mname (:memo/name m)
        axes (:memo/axes m)
        nres (:memo/num-results m)
        ;; an axis named by a symbol or a keyword
        given (mapv #(if (keyword? %) (symbol (name %)) %) given)
        _ (when (empty? axes)
            (err/raise (str "Memo " mname " has no axes: it is not a distribution")
                       {:hint (str who " samples a memo's axes; a memo without axes returns a number.")
                        :loc {:name mname}}))
        _ (doseq [g given]
            (when-not (some #{g} axes)
              (err/raise (str who ": " g " is not an axis of memo " mname)
                         {:hint (str "Its axes are " (apply str (interpose ", " axes)) ".")
                          :loc {:name mname}})))
        _ (when (not= (count given) (count (set given)))
            (err/raise (str who ": an axis given twice, " (pr-str given)) {:loc {:name mname}}))
        sampled (vec (remove (set given) axes))
        _ (when (empty? sampled)
            (err/raise (str who ": every axis of memo " mname " is given, so none is sampled")
                       {:loc {:name mname}}))
        ret (:return opts)
        _ (cond
            (and (clojure.core/> nres 1) (nil? ret))
            (err/raise (str "Memo " mname " has " nres " returns: name the distribution's with :return")
                       {:hint (str "(" who " " mname " {:return 0}) is the first return's.") :loc {:name mname}})
            (and (some? ret) (not (and (integer? ret) (clojure.core/<= 0 ret (dec nres)))))
            (err/raise (str who ": :return " (pr-str ret) " is not a return of memo " mname
                            ", which has " nres)
                       {:loc {:name mname}}))
        position (into {} (map-indexed (fn [i a] [a i])) axes)
        values (into {} (map (fn [a d] [a (domains/values d)]) axes (:memo/domains m)))
        k (count sampled)
        n (reduce clojure.core/* (map (comp count values) sampled))
        labels (if (= k 1) (values (first sampled)) (cells (map values sampled)))
        info {:model mname :k k :axes sampled
              :support (vec (distinct labels))}
        ;; given axes from the last, so the earlier ones keep their positions
        given-desc (sort-by (comp clojure.core/- position) given)
        pnames (:memo/params m)]
    (fn [args extra]
      (let [[gvals args] (if (seq given)
                           (let [g (first args)]
                             (when-not (and (sequential? g) (= (count g) (count given)))
                               (err/raise (str "The distribution of memo " mname " given " (pr-str given)
                                               " takes their values first, as a vector")
                                          {:hint (str "As in a memo call: (d [" (apply str (interpose " " given))
                                                      "] args ...).")
                                           :loc {:name mname}}))
                             [(vec g) (rest args)])
                           [nil args])
            args (if (seq extra)
                   (if (map? (last args)) (concat (butlast args) [(merge (last args) extra)]) (concat args [extra]))
                   args)
            t (apply model (guard-args mname pnames args))
            t (if (clojure.core/> nres 1) (array/take t ret 0) t)
            gidx (zipmap given
                         (map (fn [a v]
                                (let [x (if (array/array? v) (array/item v) v)]
                                  (or (index-in (values a) x)
                                      (err/raise (str "The value " (pr-str x) " of given axis " a " of memo "
                                                      mname " is not in its domain")
                                                 {:hint (str "Its values are " (pr-str (values a)) ".")
                                                  :loc {:name mname}}))))
                              given gvals))
            t (reduce (fn [t a] (array/take t (gidx a) (position a))) t given-desc)
            probs (array/reshape t [n])]
        (check-table! mname probs n given gvals)
        (memo-table probs labels info)))))

(defn dist
  "A distribution constructor for model, a memo whose return is a
  probability over its axes: called with the memo's parameters, as the memo
  is, it gives a gen.clj distribution (as (d/categorical p) does).

  - A draw is a value of the sampled axes' domains: an enum member's value,
    an element of a vector domain; with several sampled axes, the vector of
    their values, in the order the axes are declared.
  - log-prob is the log of the drawn cell. A value outside the domain is an
    error.
  - The table is checked at every call: no NaN, no negative cell, each
    distribution in it summing to 1 within 1e-6 + n·eps.
  - A NaN gradient reaching a parameter is an error naming it.
  - (Experimental, M3.) A bounded mind's budget, in the call's trailing
    map, is passed on: the table is exact over its samples. A Monte Carlo budget is an error: its
    table is an estimate (see estimated).

  opts:
  - :given, a vector of the memo's axes (symbols or keywords) given as
    values, not sampled. The constructor then takes their values first, as
    a vector in :given's order, as a memo call does:
    ((dist L1 {:given '[u]}) [u] alpha).
  - :return, which return is the distribution, for a memo with several."
  ([model] (dist model {}))
  ([model opts]
   (let [d (distribution-of "dist" model opts)
         mname (:memo/name (meta model))]
     (fn [& args]
       (when-let [b (:memo/budget (trailing-options args))]
         (when (bounded/monte-carlo? (bounded/normalize-budget mname b))
           (err/raise (str "The budget of this call to memo " mname " declares Monte Carlo, so its table is an estimate")
                      {:hint (str "A distribution needs exact probabilities. Use (memo.gen/estimated " mname
                                  "), a generative function whose density is the estimate, or drop :monte-carlo.")
                       :loc {:name mname}})))
       (d args nil)))))

(defn estimated
  "Experimental (M3): outside what 1.0 freezes.

  model, a memo whose return is a probability over its axes, as a gen.clj
  generative function whose density is estimated (gen.custom/estimated;
  DESIGN.md §7): for a bounded mind under a Monte Carlo budget, given in the
  call's trailing map, as {:memo/budget {'alice {:samples 8 :monte-carlo
  true}}}. Called with the memo's arguments, as dist's distributions are.

  - Its one choice, at :value, is a value of the memo's sampled axes, as
    dist's draws; its return value is that value.
  - Its density, for a value, is the cell of the memo's table computed with
    the operation's key as :memo/key: an unbiased estimate of the bounded
    agent's choice probability (log ξ, the log of the estimate). simulate
    splits its key in two, draws a table with one and the value from that
    table with the other, and weighs the value in that table: the joint
    sampler of the value and the samples.
  - Pseudo-marginal MH and PMMH (gen.clj's MH on a model that splices it)
    then target the bounded model's exact posterior.
  - Without a Monte Carlo budget the table is exact, and so is the density.
  - The estimate is unbiased when the memo's table is linear in the
    bounded agent's choice probabilities (it returns them, or marginals of
    them). An observer conditioning on the choice afterwards makes the
    table a ratio of estimates, which is not: then pseudo-marginal MCMC on
    it is not exact.

  opts: :given and :return, as dist's."
  ([model] (estimated model {}))
  ([model opts]
   (let [d (distribution-of "estimated" model opts)
         mname (:memo/name (meta model))
         at (fn [args key]
              (when (contains? (trailing-options args) :memo/key)
                (err/raise (str "estimated: the call to memo " mname " gives its own :memo/key")
                           {:hint "The generative function's operations give the key; drop :memo/key."
                            :loc {:name mname}}))
              (d args {:memo/key key}))]
     (custom/estimated
      {:addresses [:value]
       :estimate (fn [args choices key] (dc/log-prob (at args key) (cm/get-choice choices :value)))
       :simulate (fn [args key]
                   (let [[k-table k-value] (rng/split key 2)
                         dist (at args k-table)
                         v (dc/sample dist k-value [])]
                     {:choices (cm/choicemap :value v) :log-estimate (dc/log-prob dist v)}))
       :retval (fn [_ choices] (cm/get-choice choices :value))}))))

(defn log-estimate-variance
  "Experimental (M3): outside what 1.0 freezes.

  How noisy an estimated density's log-likelihood is for one participant:
  the sum of log ξ over its trials, each [args value] (args as gf takes them,
  value its observed :value), drawn :draws times (default 100) with keys from
  :key. Returns {:mean :variance :sd :draws}. Pseudo-marginal MH mixes well
  while :sd is about 1 to 1.7 (Doucet, Pitt, Deligiannidis & Kohn, 2015);
  more draws per trial (the budget's :draws) lower it."
  [gf trials {:keys [draws key] :or {draws 100}}]
  (when-not key
    (err/raise "log-estimate-variance needs a :key" {:hint "e.g. {:draws 100 :key (gen.random/key 0)}"}))
  (let [totals (mapv (fn [k]
                       (reduce clojure.core/+
                               (map (fn [[args v] kk]
                                      (double (array/item (:weight (gfi/assess gf args (cm/choicemap :value v) kk)))))
                                    trials (rng/split k (count trials)))))
                     (rng/split key draws))
        mean (clojure.core// (reduce clojure.core/+ totals) draws)
        variance (if (clojure.core/> draws 1)
                   (clojure.core// (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % mean) 2) totals)) (dec draws))
                   0.0)]
    {:mean mean :variance variance :sd (Math/sqrt variance) :draws draws}))

;; ---------------------------------------------------------------------------
;; The deterministic generative function
;; ---------------------------------------------------------------------------

(defn deterministic
  "model, a memo, as a gen.custom/deterministic generative function whose
  return value is its table (one return of it with :return, for a memo
  with several): for splicing a mind at an address. Its parameters pass
  through the NaN guard, as dist's do. Called outside a model, it is the
  memo. Unlike dist's, its values are not checked for lost mass (an
  impossible observation gives 0, silently, as in memo; memoclj-wbhh).
  opts: :return."
  ([model] (deterministic model {}))
  ([model {:keys [return] :as opts}]
   (let [m (memo-meta "deterministic" model)
         _ (check-options "deterministic" model opts #{:return})
         mname (:memo/name m)
         nres (:memo/num-results m)
         _ (when (and (some? return) (not (and (integer? return) (clojure.core/<= 0 return (dec nres)))))
             (err/raise (str "deterministic: :return " (pr-str return) " is not a return of memo " mname
                             ", which has " nres)
                        {:loc {:name mname}}))
         pnames (:memo/params m)]
     (custom/deterministic
      {:apply (fn [args]
                (let [t (apply model (guard-args mname pnames args))]
                  (if (and (some? return) (clojure.core/> nres 1)) (array/take t return 0) t)))}))))
