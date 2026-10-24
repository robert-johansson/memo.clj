(ns memo.bounded
  "Bounded minds (M3; DESIGN.md §7), experimental: outside what 1.0 freezes.

  A memo called with a budget, keyed by agent, evaluates that agent's queries
  from samples instead of from its exact beliefs:

    (choose 2.0 {:memo/budget {'agent {:samples 8}}})

  Every query the agent evaluates (E, Pr, Var, H, KL, and the expectation
  inside EU and Predict), in every frame it owns (its own wherever it is
  modeled, its imagined worlds, its future selves and snapshots), is computed
  on k samples of the variables the query depends on, in place of its belief:
  a plug-in estimate. The observer marginalizes the samples at the agent's
  choice, P(a | x) = E_ω[π(a | estimates(ω), x)], exactly, by enumerating the
  samples' count vectors (or by Monte Carlo, when the budget declares it).

  A budget is {agent spec}, an agent a symbol (or keyword), a spec a map:
    :samples k        each query from k samples
    :algorithm a      :one-and-done (the default: samples from the agent's
                      current belief, Vul et al. 2014) or :rejection
                      (samples from its prior, kept when consistent with
                      what it observed)
    :per-option true  every option of a choice has samples of its own (by
                      default options share them, unless the agent's belief
                      about what it samples depends on the option)
    :options k        each of its :chooses considers k options, drawn
                      uniformly without replacement
    :monte-carlo true the samples drawn (with :draws R draws, default 1) in
                      place of enumerated; the call then takes a :memo/key
    :limit n          the enumerated states allowed at one choice (default
                      10^6; the call's :memo/limit sets it for every agent
                      without a :limit of its own)

  This namespace holds the budget's validation, the combinatorics, and the
  lowering of the bounded IR instructions (:sample-weight, :sample-belief,
  :subsets), which memo.lower runs; the analysis (memo.analyze) emits them."
  (:refer-clojure :exclude [+ - * / < > <= >= == abs])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * / > == where]]
            [gen.array.tape :as tape]
            [gen.random :as rng]
            [memo.errors :as err]))

;; -----------------------------------------------------------------------------
;; the budget

(def default-limit
  "The enumerated states allowed at one choice, unless a budget says otherwise
  (DESIGN.md §7.2, the owner's decision of 2026-10-24)."
  1000000)

(def algorithms
  "How a bounded agent draws its samples."
  #{:one-and-done :rejection})

(def ^:private spec-keys [:samples :algorithm :per-option :options :monte-carlo :draws :limit])

(def ^:private budget-hint
  (str "A budget is {agent spec}, e.g. {:memo/budget {'alice {:samples 8}}}, a spec taking "
       ":samples, :algorithm (:one-and-done or :rejection), :per-option, :options, "
       ":monte-carlo, :draws and :limit."))

(defn- whole?
  "x is a positive whole number (a long, or a double such as 1e7)."
  [x]
  (and (number? x) (clojure.core/> x 0) (clojure.core/== x (Math/floor (double x)))
       (clojure.core/< (double x) 9.2e18)))

(defn limit-value
  "A limit given as a number, as a long; a loud error otherwise."
  [what x]
  (when-not (whole? x)
    (err/raise (str what " is " (pr-str x) ", not a positive whole number")
               {:hint "The limit is the number of enumerated states allowed at one choice, e.g. 1e7."}))
  (long x))

(defn- agent-name [mname k]
  (let [s (cond (and (symbol? k) (nil? (namespace k))) k
                (and (keyword? k) (nil? (namespace k))) (symbol (name k))
                :else (err/raise (str "A budget's agent is " (pr-str k) ", not an agent's name")
                                 {:hint (str "Name an agent as a symbol or a keyword: {'alice {:samples 8}} or "
                                             "{:alice {:samples 8}}.")
                                  :loc {:name mname}}))]
    (when (= 'self s)
      (err/raise "self cannot be given a budget"
                 {:hint "self is how an agent refers to itself; name the agent." :loc {:name mname}}))
    (when (= 'observer s)
      (err/raise "The observer cannot be given a budget"
                 {:hint (str "The observer is the memo's own frame, the scientist's: its queries define the "
                             "table. Bound an agent it models.")
                  :loc {:name mname}}))
    s))

(defn- spec [mname who s]
  (let [bad (fn [msg & [hint]]
              (err/raise (str "The budget of " who " " msg) {:hint (or hint budget-hint) :loc {:name mname}}))]
    (when-not (map? s) (bad (str "is " (pr-str s) ", not a map")))
    (when-let [k (first (remove (set spec-keys) (keys s)))]
      (bad (str "has an unknown key " (pr-str k))))
    (let [{:keys [samples algorithm per-option options monte-carlo draws limit]} s]
      (when (and (some? samples) (not (and (integer? samples) (pos? samples))))
        (bad (str "has :samples " (pr-str samples) ", not a positive integer")))
      (when (and (some? options) (not (and (integer? options) (pos? options))))
        (bad (str "has :options " (pr-str options) ", not a positive integer")))
      (when (and (nil? samples) (nil? options))
        (bad "bounds nothing" "A budget gives :samples (estimates from samples), :options (options considered), or both."))
      (when (some? algorithm)
        (when (nil? samples) (bad "has an :algorithm but no :samples" "An algorithm draws samples: give :samples k."))
        (when-not (algorithms algorithm)
          (bad (str "has :algorithm " (pr-str algorithm) ", not one of :one-and-done, :rejection"))))
      (when (some? per-option)
        (when-not (boolean? per-option) (bad (str "has :per-option " (pr-str per-option) ", not true or false")))
        (when (nil? samples) (bad "has :per-option but no :samples" "Options share or own samples: give :samples k.")))
      (when (and (some? monte-carlo) (not (boolean? monte-carlo)))
        (bad (str "has :monte-carlo " (pr-str monte-carlo) ", not true or false")))
      (when (some? draws)
        (when-not (and (integer? draws) (pos? draws)) (bad (str "has :draws " (pr-str draws) ", not a positive integer")))
        (when-not (true? monte-carlo)
          (bad "has :draws without :monte-carlo true" "Draws are Monte Carlo's: add :monte-carlo true.")))
      {:samples samples
       :algorithm (when samples (or algorithm :one-and-done))
       :per-option (boolean per-option)
       :options options
       :monte-carlo (boolean monte-carlo)
       :draws (or draws 1)
       :limit (when (some? limit) (limit-value (str "The :limit of " who "'s budget") limit))})))

(defn normalize-budget
  "budget, {agent spec}, validated: agents as symbols, each spec with every
  key, defaults filled in. nil for an empty or nil budget (the exact memo).
  Loud errors name what is wrong."
  [mname budget]
  (cond
    (nil? budget) nil
    (not (map? budget))
    (err/raise (str "The budget is " (pr-str budget) ", not a map") {:hint budget-hint :loc {:name mname}})
    (empty? budget) nil
    :else
    (let [named (map (fn [[k v]] [(agent-name mname k) v]) budget)]
      (when-let [a (some (fn [[a n]] (when (clojure.core/> n 1) a)) (frequencies (map first named)))]
        (err/raise (str "The budget names " a " twice")
                   {:hint (str "A symbol and a keyword name the same agent ('" a " and :" a "): give it one "
                               "spec.")
                    :loc {:name mname}}))
      (into (sorted-map-by #(compare (str %1) (str %2)))
            (map (fn [[a s]] [a (spec mname a s)]))
            named))))

(defn monte-carlo?
  "Does a normalized budget declare Monte Carlo for any agent?"
  [budget]
  (boolean (some :monte-carlo (vals budget))))

(defn limit-of
  "The limit for an agent's spec: its own :limit, else the call's, else the
  default (10^6)."
  [spec call-limit]
  (or (:limit spec) call-limit default-limit))

;; -----------------------------------------------------------------------------
;; combinatorics (host)

(defn binomial
  "n choose r, exactly."
  [n r]
  (if (or (neg? r) (clojure.core/> r n))
    0N
    (let [r (min r (clojure.core/- n r))]
      (loop [i 0 acc 1N]
        (if (= i r) acc (recur (inc i) (clojure.core// (clojure.core/* acc (clojure.core/- n i)) (inc i))))))))

(defn states
  "The count vectors of k samples over m categories: C(k + m - 1, m - 1)."
  [k m]
  (binomial (clojure.core/+ k m -1) (dec m)))

(defn power
  "b^e, exactly (b a bigint or long, e a non-negative integer)."
  [b e]
  (reduce clojure.core/* 1N (repeat e (bigint b))))

(defn- compositions [k m]
  (if (= m 1)
    [[k]]
    (vec (for [c (range k -1 -1), tail (compositions (clojure.core/- k c) (dec m))] (into [c] tail)))))

(def ^:private host-cache (atom {}))

(defn- host-cached
  "(build), cached under k on the host, the cache starting over past 16
  entries (an enumeration is up to the limit's size, 10^6 vectors by
  default, and a fit may try many budgets)."
  [k build]
  (if-let [[_ v] (find @host-cache k)]
    v
    (let [v (build)]
      (swap! host-cache (fn [m] (assoc (if (clojure.core/>= (count m) 16) {} m) k v)))
      v)))

(defn count-vectors
  "Every way k samples fall into m categories, as count vectors, in a fixed
  order (the first category's count descending)."
  [k m]
  (host-cached [:count-vectors k m] #(compositions k m)))

(defn- log-factorials [k]
  (host-cached [:log-factorials k] #(vec (reductions clojure.core/+ 0.0 (map (fn [i] (Math/log (double i))) (range 1 (inc k)))))))

(defn- log-coefficients
  "log k! / Π c_j! for each count vector."
  [k cvs]
  (let [lf (log-factorials k)]
    (mapv (fn [c] (clojure.core/- (lf k) (reduce clojure.core/+ (map lf c)))) cvs)))

(defn- combinations
  "Every k-subset of 0..n-1, as sorted vectors, lexicographically."
  [n k]
  (letfn [(go [start k] (if (zero? k) [[]] (vec (for [i (range start (inc (clojure.core/- n k))), t (go (inc i) (dec k))] (into [i] t)))))]
    (go 0 k)))

;; -----------------------------------------------------------------------------
;; layout (dim d is array axis -1-d, as in memo.lower)

(defn- axis [d] (clojure.core/- -1 d))

(defn- pad [x r]
  (let [s (array/shape x)
        k (clojure.core/- r (count s))]
    (if (pos? k) (array/reshape x (into (vec (repeat k 1)) s)) x)))

(defn- dims-shape
  "A shape of rank r with size (sizes d) on the axis of each dim d, 1 elsewhere."
  [r sizes]
  (mapv (fn [i] (get sizes (clojure.core/- r 1 i) 1)) (range r)))

(defn- front-perm [dims r]
  (let [front (mapv #(clojure.core/+ r (axis %)) dims)]
    (into front (remove (set front)) (range r))))

(defn- to-front
  "x (rank r) with the axes of dims first, in that order, the others after."
  [x dims r]
  (array/transpose x (front-perm dims r)))

(defn- from-front
  "y with its first axes those of dims (to-front's layout), back on the dims."
  [y dims r]
  (let [perm (front-perm dims r)
        inv (reduce (fn [v [i p]] (assoc v p i)) (vec (repeat r 0)) (map-indexed vector perm))]
    (array/transpose y inv)))

(defn- host-array
  "A float array of shape from row-major host values."
  [values shape]
  (array/reshape (array/array (double-array values)) shape))

(defn- index-array [values]
  (array/array (double-array values) {:dtype :int32}))

;; -----------------------------------------------------------------------------
;; constants, built once per backend (an array belongs to one)

(def ^:private constants (atom {}))

(defn- constant
  "(build), cached under k for the backend in effect: the host arrays of the
  count vectors and draws, which depend only on the budget and the sizes.
  The cache starts over past 32 entries (a batch backend is one per batch
  shape, and an exact budget's arrays reach tens of megabytes)."
  [k build]
  (let [key [array/*backend* k]]
    (if-let [[_ v] (find @constants key)]
      v
      (let [v (build)]
        (swap! constants (fn [m] (assoc (if (clojure.core/>= (count m) 32) {} m) key v)))
        v))))

;; -----------------------------------------------------------------------------
;; the multinomial probabilities, with gradients at zero categories

(defn- pmf
  "P[..., s] = coef_s Π_j p[..., j]^C_sj, the probability of each count vector
  C_s (host, S of them) under the category probabilities p (categories on the
  last axis), computed in log space. Its gradient is written out so that a
  category of probability 0 keeps the right one: dP_s/dp_j there is coef_s
  Π_{i≠j} p_i^C_si when C_sj = 1, and 0 when C_sj > 1 (through log space it
  would be NaN)."
  [p k cvs]
  (let [ps (array/shape p)
        m (peek ps)
        n (reduce clojure.core/* 1 (pop ps))
        S (count cvs)
        {:keys [ct c pos-t ones-c lc]}
        (constant [:pmf k m]
                  #(let [flat (vec (apply concat cvs))
                         at (fn [s j] (nth flat (clojure.core/+ (clojure.core/* s m) j)))]
                     {:ct (host-array (for [j (range m), s (range S)] (at s j)) [m S])
                      :c (host-array flat [S m])
                      :pos-t (host-array (for [j (range m), s (range S)] (if (pos? (at s j)) 1.0 0.0)) [m S])
                      :ones-c (host-array (map (fn [x] (if (clojure.core/== 1 x) 1.0 0.0)) flat) [S m])
                      :lc (host-array (log-coefficients k cvs) [1 S])}))
        parts (fn [p2]
                (let [lp (array/log (where (> p2 0) p2 1.0))
                      l (+ (array/matmul lp ct) lc)
                      nbad (array/matmul (where (== p2 0) 1.0 0.0) pos-t)]
                  [l nbad]))
        f (fn [p]
            (let [p2 (array/reshape p [n m])
                  [l nbad] (parts p2)]
              (array/reshape (where (> nbad 0) 0.0 (array/exp l)) (conj (pop ps) S))))
        vjp (fn [[p] out g]
              (let [p2 (array/reshape p [n m])
                    [l nbad] (parts p2)
                    pv (array/reshape out [n S])
                    g2 (array/reshape g [n S])
                    inner (/ (array/matmul (* g2 pv) c) (where (> p2 0) p2 1.0))
                    edge (array/matmul (* g2 (where (== nbad 1) (array/exp l) 0.0)) ones-c)]
                [(array/reshape (where (> p2 0) inner edge) ps)]))]
    ((array/custom-vjp f vjp) p)))

;; -----------------------------------------------------------------------------
;; sampled beliefs

(defn- digits
  "The digits of 0 .. S^n - 1 in base S, n per row, the first most significant."
  [S n]
  (let [total (long (power S n))]
    (vec (for [w (range total)]
           (loop [c (dec n) w w out (vec (repeat n 0))]
             (if (neg? c) out (recur (dec c) (quot w S) (assoc out c (rem w S)))))))))

(defn- check-mixing
  "x (a value of the dims layout) does not vary along the dims the analysis
  found it does not, :shared: dims the agent knows and its reasoning mixes,
  but which have no samples of their own. A belief varying there would need
  independent samples."
  [x {:keys [shared]} what]
  (let [r (array/ndim x)
        s (array/shape x)]
    (doseq [d shared
            :when (clojure.core/< d r)]
      (when (not= 1 (nth s (clojure.core/+ r (axis d))))
        (err/raise (str "memo.clj's analysis missed that " what " varies along dim " d)
                   {:user false :hint "Its samples would be shared where they must be independent."})))))

(defn- belief
  "The belief p (the marginal over the sampled dims, on the dims layout, or a
  number) laid out as [1 n m rest...]: n the cells of the dims with samples of
  their own, m those of the sampled dims, rest the other axes. Returns
  {:r :front :rest :mass [1 n 1 rest] :cats [n rest... m']}, cats the
  categories of one draw: the normalized belief (:one-and-done), or that
  times the evidence z and a last category, rejection, of 1 - z
  (:rejection)."
  [p z {:keys [sample sample-sizes indep indep-sizes omega algorithm] :as instr}]
  (let [p (if (number? p) (array/array (double p)) p)
        _ (check-mixing p instr "a bounded agent's belief")
        _ (when z (check-mixing z instr "a bounded agent's evidence"))
        r (max (inc (reduce max omega (concat sample indep))) (array/ndim p) (if z (array/ndim z) 0))
        n (reduce clojure.core/* 1 indep-sizes)
        m (reduce clojure.core/* 1 sample-sizes)
        front (into [omega] (concat indep sample))
        sizes (zipmap (concat indep sample) (concat indep-sizes sample-sizes))
        a (to-front (* (pad p r) (array/ones (dims-shape r sizes))) front r)
        rest-shape (vec (drop (count front) (array/shape a)))
        a (array/reshape a (into [1 n m] rest-shape))
        mass (array/sum a {:axis 2 :keep-dims? true})
        _ (when (array/any-true? (array/< a 0.0))
            (err/raise "A bounded agent's belief has a negative probability: it cannot be sampled"
                       {:hint (str "Its beliefs came from negative weights (a :wpp, or an :observes-event "
                                   "weight, below 0). memo computes with them exactly, but samples need "
                                   "probabilities: keep the weights non-negative.")}))
        has-mass (> mass 0)
        phat (where has-mass (/ a (where has-mass mass 1.0)) (/ 1.0 m))
        cats (if (= algorithm :rejection)
               (let [zs (if z
                          (let [zf (to-front (* (pad z r) (array/ones (dims-shape r (zipmap indep indep-sizes)))) front r)]
                            (array/broadcast-to (array/reshape zf (into [1 n 1] (drop (count front) (array/shape zf))))
                                                (into [1 n 1] rest-shape)))
                          (array/ones (into [1 n 1] rest-shape)))]
                 (when (array/any-true? (> zs (clojure.core/+ 1.0 1e-6)))
                   (err/raise "A rejection sampler's acceptance probability is above 1"
                              {:hint (str "With :algorithm :rejection, an agent keeps a sample with the probability "
                                          "of what it observed: an :observes-event weight must not exceed 1.")}))
                 (when (array/any-true? (array/< zs 0.0))
                   (err/raise "A rejection sampler's acceptance probability is below 0"
                              {:hint (str "With :algorithm :rejection, an agent keeps a sample with the probability "
                                          "of what it observed: an :observes-event weight must not be negative.")}))
                 (array/concatenate [(* zs phat) (array/clip (- 1.0 zs) 0.0 1.0)] 2))
               phat)
        mprime (nth (array/shape cats) 2)
        k (count rest-shape)]
    {:r r :front front :rest rest-shape :mass mass :m m :n n
     ;; [1 n m' rest...] -> [n rest... m']
     :cats (array/transpose (array/reshape cats (into [n mprime] rest-shape))
                            (-> [0] (into (map #(clojure.core/+ 2 %) (range k))) (conj 1)))}))

(defn- back
  "y, laid out [ω n m more...] (more: the rest axes, or 1s), with its n and m
  axes split into the dims' sizes (or 1s), back on the dims layout."
  [y {:keys [r front]} indep-sizes sample-sizes]
  (let [s (array/shape y)]
    (from-front (array/reshape y (-> [(first s)] (into indep-sizes) (into sample-sizes) (into (drop 3 s))))
                front r)))

(defn- empirical
  "The empirical belief of each count vector: counts over the first m
  categories divided by k (:one-and-done), or by the samples kept
  (:rejection; 0 when none is kept)."
  [cvs m k algorithm]
  (mapv (fn [c]
          (let [c (subvec c 0 m)
                total (if (= algorithm :rejection) (reduce clojure.core/+ c) k)]
            (mapv #(if (zero? total) 0.0 (clojure.core// (double %) total)) c)))
        cvs))

(defn- sample-space
  "The sampled cells m, the categories of one draw m' (m, or m + 1 with
  rejection's), and, for the exact lowering, the count vectors (a delay:
  Monte Carlo never enumerates them)."
  [{:keys [k algorithm sample-sizes]}]
  (let [m (reduce clojure.core/* 1 sample-sizes)
        mprime (if (= algorithm :rejection) (inc m) m)]
    {:m m :mprime mprime :cvs (delay (count-vectors k mprime))}))

(defn sample-weight
  "The :sample-weight instruction: the probability of each enumerated draw,
  every set of count vectors (one per cell of the dims with samples of their
  own), on the draw's dim ω and the belief's contexts."
  [p z {:keys [k indep-sizes sample-sizes] :as instr}]
  (let [{:keys [cvs mprime]} (sample-space instr)
        cvs @cvs
        {:keys [n cats rest] :as b} (belief p z instr)
        S (count cvs)
        probs (pmf cats k cvs)                           ; [n rest... S]
        idx (constant [:digits S n]
                      #(let [ds (digits S n)] (mapv (fn [c] (index-array (map (fn [d] (nth d c)) ds))) (range n))))
        per-cell (for [c (range n)]
                   (array/take (array/take probs c 0) (idx c) -1))
        w (reduce * per-cell)                            ; [rest... S^n]
        w (array/transpose w (into [(count rest)] (range (count rest))))]
    (back (array/reshape w (into [(first (array/shape w)) 1 1] rest))
          b (repeat (count indep-sizes) 1) (repeat (count sample-sizes) 1))))

(defn- belief-of-draws
  "q from the empirical beliefs e [ω n m] (host or array) and the mass."
  [e {:keys [mass n m rest algorithm] :as b} indep-sizes sample-sizes]
  (let [w (first (array/shape e))
        e (array/reshape e (into [w n m] (repeat (count rest) 1)))
        q (if (= algorithm :rejection) e (* e mass))]
    (back q b indep-sizes sample-sizes)))

(defn sample-belief
  "The :sample-belief instruction: the agent's empirical belief over the
  sampled dims for each draw on dim ω, enumerated (every set of count vectors)
  or, with :draws, drawn with key."
  [p z {:keys [k indep-sizes sample-sizes algorithm draws] :as instr} key]
  (let [{:keys [m mprime cvs]} (sample-space instr)
        b (assoc (belief p z instr) :algorithm algorithm)
        {:keys [n cats rest]} b]
    (if-not draws
      (let [e (constant [:empirical k m mprime algorithm n]
                        #(let [cvs @cvs
                               emp (empirical cvs m k algorithm)
                               ds (digits (count cvs) n)]
                           (host-array (for [d ds, c (range n), x (nth emp (nth d c))] x) [(count ds) n m])))]
        (belief-of-draws e b indep-sizes sample-sizes))
      (do
        (when (some tape/tracked? [p z])
          (err/raise "Monte Carlo samples cannot be differentiated through the agent's belief"
                     {:hint (str "The agent's belief depends on a parameter being differentiated, and the samples "
                                 "drawn from it carry no gradient. Use the exact lowering (no :monte-carlo), or "
                                 "fit without gradients (pseudo-marginal MH).")}))
        (when-not key
          (err/raise "A Monte Carlo budget needs a key: call with :memo/key"
                     {:hint "(f args... {:memo/budget {...} :memo/key (gen.random/key 0)})"}))
        (let [logits (array/log (array/stop-gradient cats))               ; [n rest... m']
              idx (rng/categorical key logits (into [draws k n] rest))   ; [R k n rest...]
              onehot (where (== (array/expand-dims idx -1) (array/arange mprime)) 1.0 0.0)
              counts (array/sum onehot {:axis 1})                       ; [R n rest... m']
              kept (array/slice counts -1 0 m)
              e (if (= algorithm :rejection)
                  (let [t (array/sum kept {:axis -1 :keep-dims? true})]
                    (where (> t 0) (/ kept (where (> t 0) t 1.0)) 0.0))
                  (/ kept (double k)))
              nr (count rest)
              ;; [R n rest... m] -> [R n m rest...]
              e (array/transpose e (-> [0 1] (conj (clojure.core/+ 2 nr)) (into (map #(clojure.core/+ 2 %) (range nr)))))
              q (if (= algorithm :rejection) e (* e (array/reshape (:mass b) (into [1 n 1] rest))))]
          (back q b indep-sizes sample-sizes))))))

(declare subsets*)

(defn subsets
  "The :subsets instruction: for each draw on dim σ, which options are
  considered (true) among the cells of the chosen dims, every k-subset
  enumerated or, with :draws, drawn uniformly with key."
  [{:keys [omega dims sizes k draws] :as instr} key]
  (if-not draws
    (constant [:subsets omega dims sizes k] #(subsets* instr nil))
    (subsets* instr key)))

(defn- subsets* [{:keys [omega dims sizes k draws]} key]
  (let [n (reduce clojure.core/* 1 sizes)
        rows (if-not draws
               (combinations n k)
               (do (when-not key
                     (err/raise "A Monte Carlo budget needs a key: call with :memo/key"
                                {:hint "(f args... {:memo/budget {...} :memo/key (gen.random/key 0)})"}))
                   (mapv (fn [u] (vec (sort (take k (sort-by u (range n))))))
                         (mapv vec (array/->clj (rng/uniform key [draws n]))))))
        mask (host-array (for [row rows, :let [in (set row)], i (range n)] (if (in i) 1.0 0.0)) (into [(count rows)] sizes))
        r (inc (reduce max omega dims))
        front (into [omega] dims)]
    (array/!= (from-front (array/reshape mask (into (array/shape mask) (repeat (clojure.core/- r (count front)) 1))) front r)
              0.0)))

;; -----------------------------------------------------------------------------
;; inspection

(defn- compile-for [who model budget opts report-only]
  (let [f (:memo/compile (meta model))]
    (when-not f
      (err/raise (str who " takes a memo (defmemo), not " (pr-str model))
                 {:hint (str "(" who " model {'alice {:samples 8}})")}))
    (f budget (:limit opts) report-only)))

(defn report
  "The bounded lowering's sizes for model under budget, and for the memos it
  calls (which the budget reaches too): one entry per choice that averages
  over samples, {:memo :agent :line :states :limit :monte-carlo :sources},
  each source {:kind (:samples or :options) :states ...}, :categories the
  values one sample can take (one more under :rejection, for a rejected
  sample) and :sets the sets of samples (one per option, where each has its
  own). A choice above its limit is reported, marked :over-limit true, where
  a call would throw. Under Monte Carlo, :states is the number of draws.
  opts: :limit (the call's)."
  ([model budget] (report model budget {}))
  ([model budget opts]
   (loop [todo [model] seen #{} out []]
     (if-let [m (peek todo)]
       (if (seen m)
         (recur (pop todo) seen out)
         (let [entries (map #(assoc % :memo (:memo/name (meta m)))
                            (:bounded-report (:compiled (compile-for "report" m budget opts true))))]
           (recur (into (pop todo) (sort-by (comp str :memo/name meta) ((:memo/callees (meta m)))))
                  (conj seen m) (into out entries))))
       out))))

(defn program
  "model's IR program under budget, as text (memo.core/program's, bounded)."
  ([model budget] (program model budget {}))
  ([model budget opts]
   (let [{:keys [instrs results axes]} (:compiled (compile-for "program" model budget opts false))]
     (str/join "\n" (concat [(str "; axes " (pr-str axes))]
                            (map-indexed (fn [i instr] (format "%4d  %s" i (pr-str instr))) instrs)
                            [(str "; results " (pr-str results))])))))
