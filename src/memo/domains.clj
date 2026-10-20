(ns memo.domains
  "Domains, the finite sets that choices range over. A domain is a vector of
  numbers (Python memo's lists and arrays) or an enum made by `defenum`
  (Python's IntEnum). A choice's value is a domain value, never an index.

  Outside defmemo an enum is a function from a member's name to its value,
  `(Utt :SOME)`. Inside defmemo, `Utt.SOME` is read as data and resolved to
  its value when the macro expands (DESIGN.md §8): no var has a dot in its
  name, which JVM Clojure would read as a Java class.")

(declare member-value)

(defrecord EnumDomain [name members values]
  clojure.lang.IFn
  (invoke [this member] (member-value this member))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn enum? [d] (instance? EnumDomain d))

(defn member-value
  "The value of an enum's member, named by a keyword or a symbol."
  [e member]
  (let [i (.indexOf ^java.util.List (:members e) (symbol (name member)))]
    (when (neg? i)
      (throw (ex-info (str "Enum " (:name e) " has no member " (name member))
                      {:enum (:name e) :member member :members (:members e)})))
    (nth (:values e) i)))

(defn values
  "The values of a domain, as a vector of numbers."
  [d]
  (cond
    (enum? d) (:values d)
    (and (sequential? d) (every? number? d)) (vec d)
    :else (throw (ex-info (str "Not a domain: " (pr-str d))
                          {:hint "A domain is a vector of numbers or an enum (defenum)."}))))

(defn labels
  "Display labels of a domain's members: enum names, or the values."
  [d]
  (if (enum? d) (mapv str (:members d)) (mapv str (values d))))

(defmacro defenum
  "(defenum Utt SILENT SOME ALL) defines Utt, an enum with values 0, 1, 2.
  Members may give explicit values: (defenum R [GREEN 5] [RED 9])."
  [name & members]
  (let [pairs (vec (map-indexed (fn [i m] (if (vector? m) [(first m) (second m)] [m i]))
                                members))]
    (when (not= (count (set (map first pairs))) (count pairs))
      (throw (ex-info (str "defenum " name ": repeated member") {:members (map first pairs)})))
    `(def ~name (->EnumDomain '~name '~(mapv first pairs) ~(mapv second pairs)))))
