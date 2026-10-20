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
    entropies."
  (:refer-clojure :exclude [+ - * / < > <= >= == abs])
  (:require [gen.array :as array :refer [+ - * / < > <= >= == exp log abs where]]
            [memo.errors :as err]))

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
    :pow (array/pow (as-number x) (as-number y))
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
    :abs (if (bool? x) x (abs x))
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
             :call-memo (call-memo (:resolve-memo ctx) env instr)))))

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

(defn program
  "The function computing a compiled memo's result: from a map of parameter
  values to its table, or, for several returns, their tables stacked on a
  leading axis. ctx: :resolve-memo, from a memo's name to the memo, at call
  time (so a memo may call itself)."
  [compiled ctx]
  (let [{:keys [instrs results]} compiled]
    (fn [params]
      (let [env (reduce (partial step ctx) params instrs)
            tables (mapv #(table (get env %) compiled) results)]
        (if (= 1 (count tables)) (first tables) (array/stack tables))))))
