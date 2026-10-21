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
  - &, |, ^ and not are logical on booleans and on the literals 0 and 1.
    Python memo computes them bitwise on other integers; gen.array has no
    bitwise operators, so those are a loud error.
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
            [gen.array.tape :as tape]
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
  "A domain's values as an array along dim d."
  [values d]
  (array/reshape (array/array values) (into [(count values)] (repeat d 1))))

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

(defn- as-bool [x] (if (bool? x) x (not (clojure.core/== x 0))))

(defn- integral? [x] (if (number? x) (integer? x) (not (float-max (array/dtype x)))))

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
  (if (and (or (bool? x) (unit-literal? x)) (or (bool? y) (unit-literal? y)))
    (f (as-bool x) (as-bool y))
    (err/raise (str "memo.clj computes " op " on booleans only")
               {:hint (str "Python memo computes " op " bitwise on integers, which gen.array "
                           "cannot do. Compare first, e.g. (= x 1).")})))

(defn- run-op [op [x y]]
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
    :and (logical "and" array/logical-and x y)
    :or (logical "or" array/logical-or x y)
    :xor (logical "xor" array/!= x y)
    :exp (exp (as-number x))
    :log (log (as-number x))
    :abs (if (bool? x) x (abs* x))
    :uadd x
    :neg (do (no-bool "unary -" x) (array/negate x))
    :inv (if (or (bool? x) (unit-literal? x))
           (array/logical-not (as-bool x))
           (err/raise "memo.clj computes not on booleans only"
                      {:hint (str "Python memo computes it as True ^ x, bitwise on "
                                  "integers, which gen.array cannot do. Compare first, "
                                  "e.g. (= x 0).")}))))

;; -----------------------------------------------------------------------------
;; memo calls

(defn- check-which [n which]
  (if (= 1 n)
    (when (some? which)
      (err/raise (str "You are calling a memo model with only one return value, so you "
                      "do not have to specify which return value you want.")))
    (do (when (nil? which)
          (err/raise (str "You are calling a memo model that has multiple return values, "
                          "but you have not specified which return value you want. Try "
                          "writing (model 0 [...] ...) to get the first return value.")))
        (when-not (clojure.core/< -1 which n)
          (err/raise "Your memo model does not have enough return values.")))))

(defn- check-domains [callee caller]
  (when (clojure.core/> (count callee) (count caller))
    (err/raise "Not enough arguments to memo call!"))
  (when (clojure.core/> (count caller) (count callee))
    (err/raise "Too many arguments to memo call!"))
  (doseq [[i t s] (map vector (range) callee caller)]
    (when (not= t s)
      (err/raise (str "Domain mismatch in memo call argument " (inc i) ": " t " != " s ".")))))

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

(defn- call-memo [resolve-memo env {:keys [name which args kwargs guards dims domains]}]
  (let [f (resolve-memo name)
        {:memo/keys [num-results domain-names shape call]} (meta f)
        _ (check-which num-results which)
        _ (check-domains domain-names domains)
        table (if (every? #(array/truthy? (get env %)) guards)
                (let [res (call (mapv #(get env %) args)
                                (into {} (map (fn [[k s]] [k (get env s)])) kwargs))]
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
      (vec (flatten (array/->clj (array/broadcast-to (pad x (count shape)) shape)))))))

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

(defn- spot-check
  "A vectorized function's result r against f on host numbers, at up to 32
  evenly spaced elements of its dynamic arguments (all of them, for most
  memo domains). An if or case on a value inside f takes one branch for a
  whole array, silently; this makes that loud, as gen.clj spot-checks its
  batched runs. Values are read on the host, primal values for those being
  differentiated, and only at those elements: read whole, demo-empowerment's
  Tr would be four times 89M host numbers."
  [name f pos statics kws r]
  (let [on-host (fn [v] (if (or (tape/tracked? v) (array/array? v)) (array/->clj v) v))
        slots (concat (map vector pos statics) (map (fn [[_ v s]] [v s]) kws))
        dyn (keep (fn [[v s]] (when-not s v)) slots)
        shape (broadcast-shape (map array/shape dyn))
        n (reduce clojure.core/* 1 shape)
        k (min n 32)
        idxs (vec (distinct (map #(long (Math/floor (clojure.core/* % (/ (dec n) (double (max 1 (dec k)))))))
                                 (range k))))
        cols (mapv #(samples % shape idxs) dyn)
        got (samples r shape idxs)
        num (fn [v] (double (if (boolean? v) (if v 1 0) v)))]
    (doseq [[j i] (map-indexed vector idxs)]
      (let [vals (first (reduce (fn [[out d] [v s]]
                                  (if s [(conj out (on-host v)) d] [(conj out (nth (nth cols d) j)) (inc d)]))
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

(defn- host-call
  "The :ffi instruction. A function marked :memo/vectorized (written with
  gen.array's operators, elementwise) is called once on whole arrays, and so
  differentiated; any other is tabulated on the host, and a value being
  differentiated reaching it is a loud error (Python memo gives a silent zero
  gradient there). A memo called this way (one without axes) is called
  directly, under its path conditions, as a memo call is."
  [{:keys [fns resolve-memo]} env {:keys [name args statics kwargs guards]}]
  (let [v (get fns name)
        f (if v @v (resolve-memo name))
        pos (mapv #(get env %) args)
        kws (mapv (fn [[k s static]] [k (get env s) static]) kwargs)
        kw-map (zipmap (map (comp keyword first) kws) (map second kws))
        direct #(apply f (cond-> pos (seq kws) (conj kw-map)))
        vals (concat pos (map second kws))
        dyn-shape (broadcast-shape (map array/shape (concat (keep (fn [[x s]] (when-not s x))
                                                                  (map vector pos statics))
                                                            (keep (fn [[_ x s]] (when-not s x)) kws))))]
    (cond
      (and (:memo/call (meta f)) (every? true? statics) (every? #(nth % 2) kws))
      (if (every? #(array/truthy? (get env %)) guards)
        (direct)
        (array/zeros dyn-shape))

      (or (:memo/vectorized (meta v)) (:memo/vectorized (meta f)))
      ;; elementwise: its result has at most the arguments' extent
      (let [r (direct)
            rs (array/shape r)]
        (when-not (and (clojure.core/<= (count rs) (count dyn-shape))
                       (every? true? (map #(or (= 1 %1) (= %1 %2)) (rseq rs) (rseq dyn-shape))))
          (not-scalar name))
        ;; on an eager backend; a lazy one (MLX) would evaluate here
        (when (array/concrete? r)
          (spot-check name f pos statics kws r))
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

(defn- step [ctx env instr]
  (let [arg #(get env %)]
    (assoc env (:out instr)
           (case (:i instr)
             :const (:value instr)
             :param (get env (:name instr))
             :domain (along (:values instr) (:dim instr))
             :ones (array/ones (into [(:size instr)] (repeat (:dim instr) 1)))
             :op (run-op (:op instr) (mapv arg (:args instr)))
             :where (let [[c t f] (map arg (:args instr))] (where c t f))
             :sum (reduce-dims array/sum (arg (:arg instr)) (:dims instr))
             :max (reduce-dims array/max (arg (:arg instr)) (:dims instr))
             :nan-to-num (nan-to-num (arg (:arg instr)))
             :posterior-align (posterior-align (arg (:arg instr)) (:knw instr) (:var instr))
             :swap-dims (let [{:keys [d1 d2]} instr]
                          (swap-axes (pad (arg (:arg instr)) (inc (max d1 d2))) (axis d1) (axis d2)))
             :alias (arg (:arg instr))
             :call-memo (call-memo (:resolve-memo ctx) env instr)
             :ffi (host-call ctx env instr)
             :inline (let [f (nth (:inlines ctx) (:idx instr))
                           v (apply f (map env (:param-order ctx)))]
                       (if (number? v) (host-number (str "#{" (pr-str (:memo/form (meta f))) "}") v) v))))))

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

(defn program
  "The function computing a compiled memo's result: from a map of parameter
  values to its table, or, for several returns, their tables stacked on a
  leading axis. ctx: :resolve-memo, from a memo's name to the memo, at call
  time (so a memo may call itself). Memo calls run first (calls-first), and
  no value is kept past its last use (drops)."
  [compiled ctx]
  (let [{:keys [results]} compiled
        instrs (calls-first (:instrs compiled))
        dead (drops instrs results)]
    (fn [params]
      (let [env (reduce (fn [env [instr dead]] (apply dissoc (step ctx env instr) dead))
                        params (map vector instrs dead))
            tables (mapv #(table (get env %) compiled) results)]
        (if (= 1 (count tables)) (first tables) (array/stack tables))))))
