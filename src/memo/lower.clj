(ns memo.lower
  "The exact lowering: an IR program (memo.ir) run on gen.array, so on every
  gen.clj backend, with gen.array's gradients.

  Layout. Dim d is array axis -1-d, as in Python memo: the values along dim d
  have shape [n 1 … 1] with d trailing ones, and everything broadcasts by
  position. Shapes are static: a value's extent follows from the program.

  Python memo computes with JAX, and where JAX gives other values than
  gen.array, the lowering follows JAX:
  - On booleans, + is logical or and * logical and: JAX (and NumPy) keep
    them bool, where gen.array converts booleans to integers (Julia's rule,
    so (array/+ true true) is 2). Booleans have no - in JAX: there it is a
    TypeError, here a loud error.
  - &, |, ^ and not are logical on booleans and on integers that are 0 or
    1, where bitwise is logical; as in JAX, the result is a boolean when
    every operand is one, an integer otherwise. Python memo computes them
    bitwise on other integers; gen.array has no bitwise operators, so those
    are a loud error.
  - nan_to_num maps NaN to 0 and ±inf to ± the dtype's largest value
    (DESIGN.md §2.4): Python memo's 0/0 = 0, and the 0 · log 0 = 0 of its
    entropies.
  - ** on host numbers (literals and parameters) is Python's, as Python
    memo computes them outside JAX: (** 2 -1) is 0.5. An integer array to a
    negative integer power, which JAX rejects (and returns garbage for
    under jit), is a loud error.
  - The gradient of abs at 0 is 1, JAX's, where gen.array's is 0."
  (:refer-clojure :exclude [+ - * / < > <= >= == abs])
  (:require [gen.array :as array :refer [+ - * / < > <= >= == exp log abs where]]
            [gen.array.batch :as batch]
            [gen.array.tape :as tape]
            [gen.random :as rng]
            [memo.bounded :as bounded]
            [memo.domains :as domains]
            [memo.errors :as err]
            [memo.ir :as ir]))

;; -----------------------------------------------------------------------------
;; layout

(defn- axis
  "The array axis of dim d."
  [d]
  (clojure.core/- -1 d))

(defn- pad
  "x with leading size-1 axes up to rank r."
  [x r]
  (let [s (array/shape x)
        k (clojure.core/- r (count s))]
    (if (pos? k) (array/reshape x (into (vec (repeat k 1)) s)) x)))

(defn- along
  "A domain's values as an array along dim d. Integers the backend holds in
  32 bits (MLX) must fit them: Python memo raises an OverflowError there."
  [values d]
  (let [a (array/array values)]
    (when (= :int32 (array/dtype a))
      (when-let [v (first (filter #(not (clojure.core/<= Integer/MIN_VALUE % Integer/MAX_VALUE)) values))]
        (err/raise (str "Domain value " v " is out of bounds for int32")
                   {:hint (str "This backend holds a domain's integers in 32 bits, and Python memo "
                               "raises an OverflowError here (the JVM backend's are 64 bits). Use "
                               "doubles, e.g. [0.0 1e10].")})))
    (array/reshape a (into [(count values)] (repeat d 1)))))

(defn- reduce-dims [f x dims]
  (if (empty? dims)
    x
    (f (pad x (inc (apply max dims))) {:axis (mapv axis dims) :keep-dims? true})))

(defn- swap-axes [x i j]
  (let [r (array/ndim x)
        i (clojure.core/+ r i)
        j (clojure.core/+ r j)]
    (array/transpose x (assoc (vec (range r)) i j j i))))

(defn- posterior-align
  "x's extent along dim var moved onto dim knw; where x already varies along
  knw, its diagonal (core.py's collapse_diagonal), leaving var size 1."
  [x knw var]
  (let [x (pad x (inc (max knw var)))
        r (array/ndim x)
        n (nth (array/shape x) (clojure.core/+ r (axis knw)))]
    (if (= 1 n)
      (swap-axes x (axis var) (axis knw))
      (let [idx (array/reshape (array/arange n)
                               (assoc (vec (repeat r 1)) (clojure.core/+ r (axis knw)) n))]
        (array/take-along-axis x idx (axis var))))))

;; -----------------------------------------------------------------------------
;; values, with JAX's semantics where they differ

(def ^:private float-max
  {:float64 Double/MAX_VALUE
   :float32 (double Float/MAX_VALUE)
   :float16 65504.0
   :bfloat16 3.3895313892515355E38})

(defn- nan-to-num [x]
  (if-let [m (float-max (array/dtype x))]
    (where (array/isnan x) 0.0 (where (array/isinf x) (where (> x 0) m (clojure.core/- m)) x))
    x))

(defn- bool? [x] (= :bool (array/dtype x)))

(defn- unit-literal? [x] (and (number? x) (or (clojure.core/== x 0) (clojure.core/== x 1))))

(defn- as-number
  "x, with booleans as the integers 0 and 1."
  [x]
  (if (bool? x) (where x 1 0) x))

(defn- as-bool [x]
  (cond (bool? x) x
        (number? x) (not (clojure.core/== x 0))
        :else (array/!= x 0)))

(defn- integral? [x] (if (number? x) (integer? x) (not (float-max (array/dtype x)))))

(defn- unit-valued?
  "Whether x is a boolean, the literal 0 or 1, or integers that are all 0 or
  1 (read on the host)."
  [x]
  (cond (bool? x) true
        (number? x) (unit-literal? x)
        (integral? x) (array/all-true? (array/logical-or (== x 0) (== x 1)))
        :else false))

(defn- as-int
  "A logical result as JAX gives it from integer operands: 1 and 0."
  [r]
  (if (boolean? r) (if r 1 0) (where r 1 0)))

(defn- python-pow
  "x ** y on host numbers, as Python computes it."
  [x y]
  (cond
    (and (zero? x) (neg? y))
    (err/raise "zero to a negative power"
               {:hint "Python memo raises a ZeroDivisionError here."})

    (and (neg? x) (not (integer? y)) (Double/isFinite (double y))
         (not= (double y) (Math/floor (double y))))
    (err/raise (str "A negative number to a fractional power, (** " x " " y "), is complex")
               {:hint "Python memo returns a complex number here, which memo.clj does not compute."})

    (and (integer? x) (integer? y) (neg? y)) (Math/pow (double x) (double y))
    :else (array/pow x y)))

(defn- pow [x y]
  (cond
    (and (number? x) (number? y)) (python-pow x y)

    ;; JAX's integer_pow: x ** 0 is 1, with gradient 0 (gen.array's
    ;; y x^(y-1) is NaN at x = 0)
    (and (integer? y) (zero? y)) (array/ones-like x)

    (and (integral? x) (integral? y)
         (if (number? y) (neg? y) (array/any-true? (< y 0))))
    (err/raise "Integers cannot be raised to negative powers"
               {:hint (str "JAX rejects an integer array to a negative integer power, and under "
                           "jit Python memo returns garbage (3 ** -1 is 2111105451). Use a float, "
                           "e.g. (** x -1.0).")})

    :else (array/pow x y)))

(defn- abs* [x]
  (if (and (not (number? x)) (float-max (array/dtype x)))
    ;; JAX's gradient, 1 at 0 (gen.array's is 0); + 0.0 makes -0.0 0.0
    (where (>= x 0) (+ x 0.0) (array/negate x))
    (abs x)))

(defn- no-bool [op x]
  (when (bool? x)
    (err/raise (str "Unsupported operand type for " op ": a boolean")
               {:hint (str "Python memo computes with JAX, which has no " op " on booleans "
                           "(a TypeError there). Convert first, e.g. (if b 1 0).")})))

(defn- logical [op f x y]
  (if (and (unit-valued? x) (unit-valued? y))
    (let [r (f (as-bool x) (as-bool y))]
      (if (and (bool? x) (bool? y)) r (as-int r)))
    (err/raise (str "memo.clj computes " op " on booleans and on 0 and 1 only")
               {:hint (str "Python memo computes " op " bitwise on other integers, which "
                           "gen.array cannot do. Compare first, e.g. (= x 1).")})))

(defn- python-scalar? [x] (or (number? x) (boolean? x)))

(defn- python-int [x] (if (boolean? x) (if x 1 0) x))

(declare run-op run-op*)

(defn- python-op
  "op on host values (parameters and literals), as Python memo computes them,
  as Python numbers: a boolean is the integer 0 or 1 in arithmetic, and
  division by zero is an error. nil when op is not one of those."
  [op [x y] what]
  (cond
    (and (#{:add :sub :mul :neg :uadd :abs} op) (or (boolean? x) (boolean? y)))
    (run-op op (if (some? y) [(python-int x) (python-int y)] [(python-int x)]) what)

    (and (#{:div :mod} op) (some? y) (clojure.core/== 0 (python-int y)))
    (err/raise "division by zero"
               {:hint (str "Python memo computes parameters and literals as Python numbers, and "
                           "raises a ZeroDivisionError here.")})))

(defn- run-op [op [x y :as args] what]
  (or (when (and (python-scalar? x) (or (nil? y) (python-scalar? y)))
        (python-op op args what))
      (run-op* op args what)))

(defn- run-op* [op [x y] what]
  (case op
    :add (if (and (bool? x) (bool? y)) (array/logical-or x y) (+ x y))
    :mul (if (and (bool? x) (bool? y)) (array/logical-and x y) (* x y))
    :sub (do (no-bool "-" x) (no-bool "-" y) (- x y))
    :div (/ (as-number x) (as-number y))
    :pow (pow (as-number x) (as-number y))
    :mod (array/mod (as-number x) (as-number y))
    :eq (== (as-number x) (as-number y))
    :neq (array/!= (as-number x) (as-number y))
    :lt (< (as-number x) (as-number y))
    :lte (<= (as-number x) (as-number y))
    :gt (> (as-number x) (as-number y))
    :gte (>= (as-number x) (as-number y))
    :and (logical (or what "and") array/logical-and x y)
    :or (logical "or" array/logical-or x y)
    :xor (logical "xor" array/!= x y)
    :exp (exp (as-number x))
    :log (log (as-number x))
    :abs (if (bool? x) x (abs* x))
    :uadd x
    :neg (do (no-bool "unary -" x) (array/negate x))
    :inv (cond
           (bool? x) (array/logical-not x)
           (unit-valued? x) (as-int (if (number? x) (clojure.core/== x 0) (array/logical-not (as-bool x))))
           :else (err/raise "memo.clj computes not on booleans and on 0 and 1 only"
                            {:hint (str "Python memo computes it as True ^ x, bitwise on "
                                        "other integers, which gen.array cannot do. Compare "
                                        "first, e.g. (= x 0).")}))))

;; -----------------------------------------------------------------------------
;; memo calls

(defn- check-which [n which loc]
  (if (= 1 n)
    (when (some? which)
      (err/raise (str "You are calling a memo model with only one return value, so you "
                      "do not have to specify which return value you want.") {:loc loc}))
    (do (when (nil? which)
          (err/raise (str "You are calling a memo model that has multiple return values, "
                          "but you have not specified which return value you want. Try "
                          "writing (model 0 [...] ...) to get the first return value.")
                     {:loc loc}))
        (when-not (clojure.core/< -1 which n)
          (err/raise "Your memo model does not have enough return values." {:loc loc})))))

(defn- same-values? [a b]
  (and (= (count a) (count b)) (every? true? (map clojure.core/== a b))))

(defn- check-domains
  "The axes a memo call passes against the callee's: their domains must have
  the same values, in order (Python memo compares their names, so another
  module's X of other values passes, misaligned)."
  [callee caller callee-values caller-values loc]
  (when (clojure.core/> (count callee) (count caller))
    (err/raise "Not enough arguments to memo call!" {:loc loc}))
  (when (clojure.core/> (count caller) (count callee))
    (err/raise "Too many arguments to memo call!" {:loc loc}))
  (doseq [[i t s tv sv] (map vector (range) callee caller callee-values caller-values)]
    (when-not (same-values? tv sv)
      (err/raise (str "Domain mismatch in memo call argument " (inc i) ": " t " != " s ".")
                 {:hint (str "The memo called ranges over " t " = " (pr-str tv) ", the choice "
                             "passed over " s " = " (pr-str sv) ": an axis takes a choice over the "
                             "same values, in the same order.")
                  :loc loc}))))

(defn- place
  "A table whose axis i belongs on dim (dims i), laid out on the dims."
  [table dims]
  (if (empty? dims)
    table
    (let [k (count dims)
          r (inc (apply max dims))
          x (array/reshape table (into (vec (array/shape table)) (repeat (clojure.core/- r k) 1)))
          at (into {} (map-indexed (fn [i d] [(clojure.core/- r 1 d) i]) dims))
          perm (loop [p 0 filler k out []]
                 (cond
                   (= p r) out
                   (contains? at p) (recur (inc p) filler (conj out (at p)))
                   :else (recur (inc p) (inc filler) (conj out filler))))]
      (array/transpose x perm))))

(defn- instr-key
  "The key of an instruction's draws in this call (M3's Monte Carlo): the
  call's :key folded with the instruction's position."
  [ctx {:keys [out]}]
  (when-let [k (get-in ctx [:call :key])]
    (rng/fold-in k (get (:positions ctx) out))))

(defn- callee-opts
  "What a memo call inside a bounded call passes on: the budget and the limit,
  and a key of its own (instr-key); nil for an exact call."
  [ctx instr]
  (when-let [c (:call ctx)]
    (cond-> (assoc (select-keys c [:budget :limit]) :propagated true)
      (:key c) (assoc :key (instr-key ctx instr)))))

(defn- call-memo [{:keys [resolve-memo domain-values] :as ctx} env
                  {:keys [name which args kwargs guards dims domains] :as instr}]
  (let [f (resolve-memo name)
        {:memo/keys [num-results domain-names shape call] callee-domains :memo/domains} (meta f)
        _ (check-which num-results which {:name (:name ctx)})
        _ (check-domains domain-names domains (mapv domains/values callee-domains)
                         (mapv domain-values domains) {:name (:name ctx)})
        table (if (every? #(array/truthy? (get env %)) guards)
                (let [res (call (mapv #(get env %) args)
                                (into {} (map (fn [[k s]] [k (get env s)])) kwargs)
                                (callee-opts ctx instr))]
                  (if (some? which) (array/take res which 0) res))
                (array/zeros shape))]
    (place table dims)))

;; -----------------------------------------------------------------------------
;; host functions

(defn- broadcast-shape
  "The NumPy broadcast of shapes (right-aligned)."
  [shapes]
  (let [r (reduce max 0 (map count shapes))]
    (if (zero? r)
      []
      (apply mapv (fn [& ds]
                    (let [big (distinct (remove #(= 1 %) ds))]
                      (when (clojure.core/> (count big) 1)
                        (err/raise (str "Arguments of a host function do not broadcast: "
                                        (pr-str shapes)) {:user false}))
                      (or (first big) 1)))
             (map #(into (vec (repeat (clojure.core/- r (count %)) 1)) %) shapes)))))

(defn- elements
  "x's values at every element of shape, on the host, in row-major order."
  [x shape]
  (if (empty? shape)
    [(array/item x)]
    (let [x (if (array/host? x) (array/array x) x)]
      (array/->clj (array/reshape (array/broadcast-to (pad x (count shape)) shape)
                                  [(reduce clojure.core/* shape)])))))

(defn- failure
  "What an exception from user code says: for a ClassCastException, which
  quotes class and loader names, what it means here."
  [e]
  (if (instance? ClassCastException e)
    "a gen.array value reached a function that takes numbers"
    (or (ex-message e) (.getSimpleName (class e)))))

(defn- not-scalar [name]
  (err/raise (str "The function " name "(...) is not scalar-in-scalar-out. memo can "
                  "only handle host functions that take scalars as input and return "
                  "a single scalar as output.")))

(defn- host-number
  "A host number as Python memo would have it: a Ratio or a BigDecimal is a
  double (Python's division gives a float), and an integer beyond 64 bits is
  a loud error (Python memo's is an OverflowError)."
  [what x]
  (cond
    (or (ratio? x) (decimal? x)) (double x)
    (and (integer? x) (not (instance? Long x)))
    (if (clojure.core/<= Long/MIN_VALUE x Long/MAX_VALUE)
      (long x)
      (err/raise (str what " gave " x ", an integer beyond 64 bits")
                 {:hint "Python memo cannot take it either (an OverflowError). Use a double."}))
    :else x))

(defn- scalar [name r]
  (cond
    (number? r) (host-number (str "The host function " name) r)
    (boolean? r) r
    (and (array/array? r) (empty? (array/shape r))) (array/item r)
    :else (not-scalar name)))

(defn- tabulate
  "f at every element of its dynamic arguments, on the host (Python memo's
  vmap of a function of scalars): any Clojure function works, and none is
  differentiated."
  [name f pos statics kwargs]
  (let [slots (concat (map vector pos statics) (map (fn [[_ v s]] [v s]) kwargs))
        dyn (keep (fn [[v s]] (when-not s v)) slots)
        shape (broadcast-shape (map array/shape dyn))
        cols (mapv #(elements % shape) dyn)
        n (reduce clojure.core/* 1 shape)
        call (fn [i]
               (let [vals (first (reduce (fn [[out d] [v s]]
                                           (if s
                                             [(conj out v) d]
                                             [(conj out (nth (nth cols d) i)) (inc d)]))
                                         [[] 0] slots))
                     args (subvec vals 0 (count pos))
                     kw (zipmap (map (comp keyword first) kwargs) (subvec vals (count pos)))
                     call-args (cond-> args (seq kwargs) (conj kw))]
                 (scalar name
                         (try (apply f call-args)
                              (catch Exception e
                                (throw (if (err/memo-error? e)
                                         e
                                         (err/memo-error
                                          (str "The host function " name " failed on "
                                               (pr-str call-args) ": "
                                               (or (ex-message e) (.getSimpleName (class e))))
                                          {:hint (str "It is called on host numbers, one element "
                                                      "at a time, with its static arguments whole.")
                                           :cause e}))))))))]
    (if (empty? shape)
      (call 0)
      (array/reshape (array/array (mapv call (range n))) shape))))

(defn- samples
  "x's values at the row-major indices idxs of shape (x broadcast to it), on
  the host: only those elements are read."
  [x shape idxs]
  (let [s (array/shape x)]
    (if (empty? s)
      (vec (repeat (count idxs) (array/->clj x)))
      (let [r (count shape)
            s (into (vec (repeat (clojure.core/- r (count s)) 1)) s)
            strides (vec (reverse (reductions clojure.core/* 1 (reverse (rest s)))))
            flat (fn [i]
                   (loop [d (dec r), i i, at 0]
                     (if (neg? d)
                       at
                       (recur (dec d) (quot i (nth shape d))
                              (if (= 1 (nth s d))
                                at
                                (clojure.core/+ at (clojure.core/* (rem i (nth shape d)) (nth strides d))))))))]
        (array/->clj (array/take (array/reshape x [(array/size x)])
                                 (array/array (mapv flat idxs) {:dtype :int32})))))))

(defn- spaced
  "Up to k indices of n, evenly spaced from the first to the last."
  [n k]
  (let [k (min n k)]
    (vec (distinct (map #(long (Math/floor (clojure.core/* % (/ (dec n) (double (max 1 (dec k)))))))
                        (range k))))))

(defn- spot-check
  "A vectorized function's result r against f on host numbers, at up to 32
  evenly spaced elements of its dynamic arguments (all of them, for most
  memo domains). An if or case on a value inside f takes one branch for a
  whole array, silently; this makes that loud, as gen.clj spot-checks its
  batched runs. Values are read on the host, primal values for those being
  differentiated, and only at those elements: read whole, demo-empowerment's
  Tr would be four times 89M host numbers. Static arguments are passed as
  given (an array parameter as its array), primal values for those being
  differentiated."
  [name f pos statics kws r]
  (let [as-given (fn [v] (if (tape/tracked? v) (array/stop-gradient v) v))
        slots (concat (map vector pos statics) (map (fn [[_ v s]] [v s]) kws))
        dyn (keep (fn [[v s]] (when-not s v)) slots)
        shape (broadcast-shape (map array/shape dyn))
        n (reduce clojure.core/* 1 shape)
        idxs (spaced n 32)
        cols (mapv #(samples % shape idxs) dyn)
        got (samples r shape idxs)
        num (fn [v] (double (if (boolean? v) (if v 1 0) v)))]
    (doseq [[j i] (map-indexed vector idxs)]
      (let [vals (first (reduce (fn [[out d] [v s]]
                                  (if s [(conj out (as-given v)) d] [(conj out (nth (nth cols d) j)) (inc d)]))
                                [[] 0] slots))
            args (subvec vals 0 (count pos))
            kw (zipmap (map (comp keyword first) kws) (subvec vals (count pos)))
            want (num (scalar name (apply f (cond-> args (seq kws) (conj kw)))))
            have (num (nth got j))]
        (when-not (or (= want have) (and (Double/isNaN want) (Double/isNaN have))
                      (clojure.core/<= (Math/abs (clojure.core/- want have))
                                       (clojure.core/* 1e-9 (clojure.core/+ 1.0 (Math/abs want)))))
          (err/raise (str "The vectorized function " name " gives other values on whole arrays "
                          "than on single values (" have " where " want " at element " i ")")
                     {:hint (str "Inside a function marked ^:memo/vectorized, branch with gen.array's "
                                 "where, not if or case: an if on an array takes one branch for every "
                                 "element. Or remove the mark, so that memo.clj calls it on each value "
                                 "(and cannot differentiate it).")}))))))

(defn- batched
  "x without its gradient tracking, when it is a value of a run under
  gen.array's batch tracer that differs across particles; else nil."
  [x]
  (let [x (if (tape/tracked? x) (array/stop-gradient x) x)]
    (when (batch/batched? x) x)))

(defn- batch-axes
  "The batch axes of a batched value b."
  [b]
  (let [s (array/shape (batch/value b))]
    (subvec s 0 (clojure.core/- (count s) (count (array/shape b))))))

(defn- particle
  "x's value in particle p of a batch of batch axes bs (x where it is shared
  by every particle), read primal."
  [x bs p]
  (if-let [b (batched x)]
    (array/take (array/reshape (batch/value b) (into [(reduce clojure.core/* 1 bs)] (array/shape b))) p 0)
    x))

(defn- host-call
  "The :ffi instruction. A function marked :memo/vectorized (written with
  gen.array's operators, elementwise) is called once on whole arrays, and so
  differentiated; any other is tabulated on the host, and a value being
  differentiated reaching it is a loud error (Python memo gives a silent zero
  gradient there). A memo called this way (one without axes) is called
  directly, under its path conditions, as a memo call is."
  [{:keys [fns resolve-memo] :as ctx} env {:keys [name args statics kwargs guards] :as instr}]
  (let [v (get fns name)
        f (if v @v (resolve-memo name))
        vectorized? (or (:memo/vectorized (meta v)) (:memo/vectorized (meta f)))
        ;; a vectorized function computes with gen.array: an array parameter
        ;; given as a vector reaches it as an array (Python memo's are JAX's)
        given (if vectorized? #(if (sequential? %) (array/array %) %) identity)
        pos (mapv #(given (get env %)) args)
        kws (mapv (fn [[k s static]] [k (given (get env s)) static]) kwargs)
        kw-map (zipmap (map (comp keyword first) kws) (map second kws))
        direct #(apply f (cond-> pos (seq kws) (conj kw-map)))
        vals (concat pos (map second kws))
        dyn-shape (broadcast-shape (map array/shape (concat (keep (fn [[x s]] (when-not s x))
                                                                  (map vector pos statics))
                                                            (keep (fn [[_ x s]] (when-not s x)) kws))))]
    (cond
      (and (:memo/call (meta f)) (every? true? statics) (every? #(nth % 2) kws))
      (if (every? #(array/truthy? (get env %)) guards)
        ((:memo/call (meta f)) pos kw-map (callee-opts ctx instr))
        (array/zeros dyn-shape))

      vectorized?
      ;; elementwise: its result has at most the arguments' extent
      (let [r (try (direct)
                   (catch Exception e
                     (throw (if (err/memo-error? e)
                              e
                              (err/memo-error
                               (str "The vectorized function " name " failed on whole arrays: "
                                    (failure e))
                               {:hint (str "A function marked ^:memo/vectorized is called once on "
                                           "gen.array arrays: write it with gen.array's operators "
                                           "(array/exp, not Math/exp), or remove the mark, so that "
                                           "memo.clj calls it on each value.")
                                :cause e})))))
            _ (when-not (or (number? r) (boolean? r) (array/array? r) (tape/tracked? r))
                (err/raise (str "The vectorized function " name " returned " (pr-str r)
                                ", not an array")
                           {:hint (str "A function marked ^:memo/vectorized returns a gen.array "
                                       "array (or a number), elementwise over its arguments.")}))
            rs (array/shape r)]
        (when-not (and (clojure.core/<= (count rs) (count dyn-shape))
                       (every? true? (map #(or (= 1 %1) (= %1 %2)) (rseq rs) (rseq dyn-shape))))
          (not-scalar name))
        ;; on an eager backend; a lazy one (MLX) would evaluate here
        (when (array/concrete? r)
          (if-let [b (some batched (cons r vals))]
            ;; under gen.array's batch tracer (gen.batched), evenly spaced
            ;; particles, each checked as its own run would be (memoclj-m2jw)
            (let [bs (batch-axes b)
                  at (fn [x p] (particle x bs p))]
              (array/with-backend (batch/inner-backend b)
                (doseq [p (spaced (reduce clojure.core/* 1 bs) 8)]
                  (spot-check name f (mapv #(at % p) pos) statics
                              (mapv (fn [[k x s]] [k (at x p) s]) kws) (at r p)))))
            (spot-check name f pos statics kws r)))
        r)

      (some tape/tracked? vals)
      (err/raise (str "Cannot differentiate through the host function " name)
                 {:hint (str "Its arguments depend on a parameter being differentiated, and "
                             name " runs on host numbers. Write it with gen.array's operators "
                             "(branching with where) and mark it ^:memo/vectorized, or give it "
                             "a gradient with gen.array/custom-vjp and mark that. (Python memo "
                             "gives a silent zero gradient here.)")})

      :else (tabulate name f pos statics kws))))

;; -----------------------------------------------------------------------------
;; programs

(defn- inline-value
  "The value of a #{...} form, f of the parameters' values: a number (as
  Python memo would have it), a boolean or a gen.array array; anything, for
  a form only passed to host functions and memo calls (args?)."
  [f params args?]
  (let [form (str "#{" (pr-str (:memo/form (meta f))) "}")
        traced? (some tape/tracked? params)
        v (try (apply f params)
               (catch Exception e
                 (throw (if (err/memo-error? e)
                          e
                          (err/memo-error
                           (str "The inline form " form " failed: " (failure e))
                           {:hint (if traced?
                                    (str "Its parameters are being differentiated, so they reach it "
                                         "as gen.array values: compute with gen.array's operators, "
                                         "e.g. #{(array/* 2 a)}, not clojure.core's.")
                                    "It is Clojure, run with the memo's parameters.")
                            :cause e})))))]
    (cond
      (number? v) (host-number form v)
      (or (boolean? v) (array/array? v) (tape/tracked? v) args?) v
      :else (err/raise (str "The inline form " form " gave " (pr-str v) ", not a number")
                       {:hint (str "An inline form's value is used as a number (or a boolean, or "
                                   "a gen.array array).")}))))

(defn- step [ctx env instr]
  (let [arg #(get env %)]
    (assoc env (:out instr)
           (case (:i instr)
             :const (:value instr)
             :param (get env (:name instr))
             :domain (along (:values instr) (:dim instr))
             :ones (array/ones (into [(:size instr)] (repeat (:dim instr) 1)))
             :op (run-op (:op instr) (mapv arg (:args instr)) (:what instr))
             :where (let [[c t f] (map arg (:args instr))] (where c t f))
             :sum (reduce-dims array/sum (arg (:arg instr)) (:dims instr))
             :max (reduce-dims array/max (arg (:arg instr)) (:dims instr))
             :nan-to-num (nan-to-num (arg (:arg instr)))
             :posterior-align (posterior-align (arg (:arg instr)) (:knw instr) (:var instr))
             :swap-dims (let [{:keys [d1 d2]} instr]
                          (swap-axes (pad (arg (:arg instr)) (inc (max d1 d2))) (axis d1) (axis d2)))
             :alias (arg (:arg instr))
             :call-memo (call-memo ctx env instr)
             :ffi (host-call ctx env instr)
             :sample-belief (bounded/sample-belief (arg (:arg instr)) (some-> (:evidence instr) arg)
                                                   (mapv arg (:weights instr)) instr (instr-key ctx instr))
             :sample-weight (bounded/sample-weight (arg (:arg instr)) (some-> (:evidence instr) arg)
                                                   (mapv arg (:weights instr)) instr)
             :subsets (bounded/subsets instr (instr-key ctx instr))
             :inline (inline-value (nth (:inlines ctx) (:idx instr)) (map env (:param-order ctx))
                                   (contains? (:inline-args ctx) (:out instr)))))))

(defn- table
  "A result value as the table over the memo's axes: the axes in declaration
  order, broadcast to their full sizes (Python memo leaves size 1 the axes a
  result does not depend on)."
  [x {:keys [n-dims forall shape]}]
  (let [axis-dims (set (map first forall))
        x (pad x n-dims)
        x (array/squeeze x (vec (for [d (range n-dims) :when (not (axis-dims d))] (axis d))))
        x (if (clojure.core/> (array/ndim x) 1) (array/transpose x) x)]
    (if (empty? shape) x (array/broadcast-to x shape))))

(defn- calls-first
  "instrs with every memo call, and the instructions it reads (statics: its
  arguments and path conditions), moved to the front, each group in its own
  order. Python memo hoists its memo calls out of the kernel the same way,
  so a recursion holds none of the tables of the levels above it: without
  this, demo-mdp's Q(1000) keeps each level's tables alive across 1000
  nested calls and the JVM backend runs out of memory."
  [instrs]
  (let [by-out (into {} (map (juxt :out identity)) instrs)
        hoisted (loop [todo (vec (keep #(when (= :call-memo (:i %)) (:out %)) instrs))
                       seen #{}]
                  (if-let [s (peek todo)]
                    (if (seen s)
                      (recur (pop todo) seen)
                      (recur (into (pop todo) (some-> (by-out s) ir/inputs)) (conj seen s)))
                    seen))]
    (into (filterv #(hoisted (:out %)) instrs) (remove #(hoisted (:out %))) instrs)))

(defn- drops
  "For each of instrs, the values to drop once it has run: those it reads
  for the last time, and its own when nothing reads it. The results are
  kept. An eager backend holds every value kept in memory: kept to the end,
  demo-empowerment's values peak at 10.7 GB in float64, dropped after their
  last use at 2.9 GB."
  [instrs results]
  (let [kept (set results)
        last-use (into {} (for [[k instr] (map-indexed vector instrs), s (ir/inputs instr)] [s k]))
        at-last-use (reduce-kv (fn [v s k] (cond-> v (not (kept s)) (update k conj s)))
                               (vec (repeat (count instrs) [])) last-use)]
    (reduce (fn [v [k {s :out}]] (cond-> v (not (or (kept s) (last-use s))) (update k conj s)))
            at-last-use (map-indexed vector instrs))))

(defn- constant
  "The outputs of instrs whose values do not depend on the call: literals,
  domains, ones, and what memo's operators compute from them alone. Never a
  parameter, an inline form, a memo call or a host function: their values
  may change with nothing in the memo changed (a redefined helper, an atom
  it reads), so they are computed at every call."
  [instrs]
  (reduce (fn [const {:keys [i out] :as instr}]
            (if (case i
                  (:const :domain :ones) true
                  (:param :inline :call-memo :ffi) false
                  ;; Monte Carlo draws differ at every call (M3)
                  (:sample-belief :sample-weight :subsets) (and (not (:draws instr))
                                                                (every? const (ir/inputs instr)))
                  (every? const (ir/inputs instr)))
              (conj const out)
              const))
          #{} instrs))

(defn program
  "The function computing a compiled memo's result: from a map of parameter
  values (and, for a bounded program, the call's options {:budget :limit
  :key}, passed on to the memos it calls) to its table, or, for several
  returns, their tables stacked on a leading axis. ctx: :resolve-memo, from
  a memo's name to the memo, at call time (so a memo may call itself);
  :name, the memo's; :domain-values, from its domains' names to their
  values; :fns, :inlines, :param-order. Memo calls run first (calls-first),
  and no value is kept past its last use (drops). What does not depend on
  the call (constant) is computed once per backend, at its first call
  there."
  [compiled ctx]
  (let [{:keys [results]} compiled
        all (calls-first (:instrs compiled))
        const (constant all)
        const-instrs (filterv #(const (:out %)) all)
        instrs (filterv #(not (const (:out %))) all)
        ;; the constants read by the rest, or results
        kept-consts (into (set (filter const results))
                          (comp (mapcat ir/inputs) (filter const)) instrs)
        dead (drops instrs results)
        ;; #{...} forms read only as arguments of host functions and memo
        ;; calls, which may take any value (a vector, say)
        readers (reduce (fn [m instr] (reduce #(update %1 %2 (fnil conj #{}) (:i instr)) m (ir/inputs instr)))
                        {} all)
        ctx (assoc ctx
                   :inline-args (set (for [{:keys [i out]} all
                                           :when (and (= :inline i) (not (some #{out} results))
                                                      (every? #{:ffi :call-memo} (readers out)))]
                                       out))
                   :positions (into {} (map-indexed (fn [i {:keys [out]}] [out i])) all))
        ;; by backend, since an array belongs to one (the JVM's and MLX's
        ;; meet in one process); a batch backend is one per batch shape, so
        ;; the cache starts over past four
        cache (atom {})
        constants (fn []
                    (let [b array/*backend*]
                      (or (get @cache b)
                          (let [env (select-keys (reduce #(step ctx %1 %2) {} const-instrs) kept-consts)]
                            (swap! cache (fn [m] (assoc (if (clojure.core/>= (count m) 4) {} m) b env)))
                            env))))]
    (fn run
      ([params] (run params nil))
      ([params call]
       (let [ctx (assoc ctx :call call)
             env (reduce (fn [env [instr dead]] (apply dissoc (step ctx env instr) dead))
                         (merge (constants) params) (map vector instrs dead))
             tables (mapv #(table (get env %) compiled) results)]
         (if (= 1 (count tables)) (first tables) (array/stack tables)))))))
