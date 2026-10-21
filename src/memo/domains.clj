(ns memo.domains
  "Domains, the finite sets that choices range over. A domain is a vector of
  numbers (Python memo's lists and arrays), an enum made by `defenum`
  (Python's IntEnum), or a product made by `defdomain` (Python memo's
  `domain(...)`). A choice's value is a domain value, never an index.

  Outside defmemo an enum is a function from a member's name to its value,
  `(Utt :SOME)`. Inside defmemo, `Utt.SOME` is read as data and resolved to
  its value when the macro expands (DESIGN.md §8): no var has a dot in its
  name, which JVM Clojure would read as a Java class.

  A product domain's members are the integers 0 .. n-1, each packing one
  value of every field (mixed radix, the first field most significant, as
  Python memo packs them). `(S :x s)` is member s's field x, on a number or
  on a gen.array array (so in a ^:memo/vectorized host function too);
  `(S {:x 1 :y 2})` is the member with those fields; `(fields S s)` is all
  of s's fields, in order (Python's `S._tuple(s)`)."
  (:require [gen.array :as array]))

(declare member-value)

(defrecord EnumDomain [name members values aliases]
  clojure.lang.IFn
  (invoke [this member] (member-value this member))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn enum? [d] (instance? EnumDomain d))

(defn member-value
  "The value of an enum's member, named by a keyword or a symbol (an alias
  too)."
  [e member]
  (let [m (symbol (name member))
        i (.indexOf ^java.util.List (:members e) (get (:aliases e) m m))]
    (when (neg? i)
      (throw (ex-info (str "Enum " (:name e) " has no member " (name member))
                      {:enum (:name e) :member member :members (:members e)})))
    (nth (:values e) i)))

(declare field pack)

(defrecord ProductDomain [name keys sizes places size]
  clojure.lang.IFn
  (invoke [this m] (pack this m))
  (invoke [this k z] (field this k z))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn product? [d] (instance? ProductDomain d))

(defn- place [d k]
  (or (get (:places d) k)
      (throw (ex-info (str "Domain " (:name d) " has no field " (pr-str k))
                      {:domain (:name d) :field k :fields (:keys d)}))))

(defn field
  "Field k of member z of product domain d: a number for a number, an array
  for an array."
  [d k z]
  (let [p (place d k) m (get (:sizes d) k)]
    (if (number? z)
      (mod (quot (long z) p) m)
      (array/mod (array/floor-div z p) m))))

(defn pack
  "The member of product domain d with the fields of m, a map from every
  field's keyword to its value."
  [d m]
  (when-not (= (set (keys m)) (set (:keys d)))
    (throw (ex-info (str "Domain " (:name d) " packs exactly its fields " (pr-str (:keys d)))
                    {:domain (:name d) :fields (:keys d) :given (vec (keys m))})))
  (reduce + (map (fn [k] (* (get m k) (place d k))) (:keys d))))

(defn fields
  "The fields of member z of product domain d, in order (Python memo's
  d._tuple(z))."
  [d z]
  (mapv #(field d % z) (:keys d)))

(defn product-domain
  "A product domain named name, from [keyword size] pairs in order. Each size
  is an integer above 1, as Python memo requires."
  [name pairs]
  (doseq [[k n] pairs]
    (when-not (and (keyword? k) (integer? n) (> n 1))
      (throw (ex-info (str "defdomain " name ": each field is a keyword and a size above 1, not "
                           (pr-str k) " " (pr-str n))
                      {:field k :size n}))))
  (when (not= (count (set (map first pairs))) (count pairs))
    (throw (ex-info (str "defdomain " name ": repeated field") {:fields (map first pairs)})))
  (let [ks (mapv first pairs)
        sizes (into {} pairs)
        places (into {} (map vector (rseq ks) (reductions * 1 (map sizes (rseq ks)))))]
    (->ProductDomain name ks sizes places (reduce * (vals sizes)))))

(defn values
  "The values of a domain, as a vector of numbers."
  [d]
  (cond
    (enum? d) (:values d)
    (product? d) (vec (range (:size d)))
    (and (sequential? d) (every? number? d)) (vec d)
    :else (throw (ex-info (str "Not a domain: " (pr-str d))
                          {:hint "A domain is a vector of numbers, an enum (defenum) or a product (defdomain)."}))))

(defn labels
  "Display labels of a domain's members: enum names, or the values."
  [d]
  (if (enum? d) (mapv str (:members d)) (mapv str (values d))))

(defmacro defenum
  "(defenum Utt SILENT SOME ALL) defines Utt, an enum with values 0, 1, 2.
  Members may give explicit values: (defenum R [GREEN 5] [RED 9]). As in
  Python's IntEnum, a member with the value of an earlier one is its alias:
  (defenum D [P 1] [Q 1] [R 2]) has the members P and R, and D.Q is P."
  [name & members]
  (let [pairs (vec (map-indexed (fn [i m] (if (vector? m) [(first m) (second m)] [m i]))
                                members))]
    (when (not= (count (set (map first pairs))) (count pairs))
      (throw (ex-info (str "defenum " name ": repeated member") {:members (map first pairs)})))
    (let [canonical (reduce (fn [acc [m v]] (if (some #(= v (second %)) acc) acc (conj acc [m v])))
                            [] pairs)
          by-value (into {} (map (fn [[m v]] [v m])) canonical)
          aliases (into {} (keep (fn [[m v]] (when (not= m (by-value v)) [m (by-value v)]))) pairs)]
      `(def ~name (->EnumDomain '~name '~(mapv first canonical) ~(mapv second canonical) '~aliases)))))

(defmacro defdomain
  "(defdomain S :x 3 :y 4) defines S, a product domain of 12 members whose
  fields x and y range over 0..2 and 0..3 (Python memo's
  S = domain(x=3, y=4)). Sizes are expressions, e.g. (count (values Loc))."
  [name & fields]
  (when (odd? (count fields))
    (throw (ex-info (str "defdomain " name ": fields come in pairs, :field size") {:fields fields})))
  `(def ~name (product-domain '~name ~(mapv vec (partition 2 fields)))))

(defenum Bool NO YES)

(def builtins
  "Domains every defmemo can name without defining them, as Python memo's
  models can (its generated code imports memo.lib's Bool). A namespace's own
  var of the same name comes first."
  {'Bool #'Bool})
