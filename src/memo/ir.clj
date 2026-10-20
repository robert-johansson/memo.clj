(ns memo.ir
  "The IR: a program of instructions over dims, emitted by the analysis
  (memo.analyze) and run by a lowering (memo.lower). It is data, not code.

  Dims. Dim d is the d-th choice made anywhere in the model, the observer's
  axes first. A value has extent along the dims it depends on and size 1
  along the others. (The exact lowering puts dim d at array axis -1-d, as
  Python memo does.)

  Single assignment. Every instruction defines a new symbol, :out, used
  only after it is defined. memo.clj never rebinds a value: Python memo
  1.3.2's rebinding of an observed choice is what lets a snapshot see its
  owner's later observations (DESIGN.md §2.4).

  Instructions:
    {:i :const  :out s :value n}             a number
    {:i :param  :out s :name p}              the parameter p
    {:i :domain :out s :dim d :values [v…]}  a domain's values along dim d
    {:i :ones   :out s :dim d :size n}       ones along dim d
    {:i :op     :out s :op k :args [s…]}     elementwise; k in memo.ast's ops
    {:i :where  :out s :args [c t f]}        elementwise choice
    {:i :sum    :out s :arg s :dims [d…]}    sum over dims (kept, size 1)
    {:i :max    :out s :arg s :dims [d…]}    maximum over dims (kept, size 1)
    {:i :nan-to-num :out s :arg s}           NaN → 0, ±inf → ± the dtype's max
    {:i :posterior-align :out s :arg s :knw d :var d}
        the marginal over dim var, re-indexed by the known dim knw: var's
        extent moves onto knw (or, where the value already varies along
        knw, its diagonal is taken)
    {:i :swap-dims :out s :arg s :d1 d :d2 d}  dims d1 and d2 exchanged
    {:i :alias :out s :arg s}                  the same value, a new name
    {:i :call-memo :out s :name f :which k :args [s…] :kwargs [[p s]…]
     :guards [s…] :dims [d…] :domains [dom…]}
        the memo f's table (its return k, when it has several), with its axes
        on the dims given; called only when every guard (a path condition)
        is true, and zeros otherwise
    {:i :ffi :out s :name f :args [s…] :statics [bool…]
     :kwargs [[k s static?]…] :guards [s…]}
        the host function f, of scalars, at every element of its dynamic
        arguments (the static ones passed whole), as Python memo's vmap
    {:i :inline :out s :idx k}                 the k-th #{…} form's value"
  (:require [clojure.string :as str]))

(def kinds
  #{:const :param :domain :ones :op :where :sum :max :nan-to-num :posterior-align
    :swap-dims :alias :call-memo :ffi :inline})

(defn inputs
  "The symbols an instruction reads."
  [instr]
  (case (:i instr)
    (:const :param :domain :ones :inline) []
    :ffi (concat (:args instr) (map second (:kwargs instr)) (:guards instr))
    (:op :where) (:args instr)
    (:sum :max :nan-to-num :posterior-align :swap-dims :alias) [(:arg instr)]
    :call-memo (concat (:args instr) (map second (:kwargs instr)) (:guards instr))))

(defn check
  "The problems of a program, as strings: unknown kinds, symbols read before
  they are defined, symbols defined twice, results never defined. [] when the
  program is well formed."
  [{:keys [instrs results]}]
  (loop [i 0, defined #{}, problems []]
    (if (= i (count instrs))
      (into problems
            (keep #(when-not (defined %) (str "result " % " is never defined")))
            results)
      (let [{kind :i out :out :as instr} (nth instrs i)
            problems (cond-> problems
                       (not (kinds kind))
                       (conj (str i ": unknown instruction " kind))
                       (defined out)
                       (conj (str i ": " out " is defined twice")))
            problems (if (kinds kind)
                       (into problems
                             (keep #(when-not (defined %)
                                      (str i ": " % " is read before it is defined")))
                             (inputs instr))
                       problems)]
        (recur (inc i) (conj defined out) problems)))))

(defn format-program
  "A compiled program as text, one instruction per line."
  [{:keys [instrs results axes]}]
  (str/join "\n"
            (concat [(str "; axes " (pr-str axes))]
                    (map-indexed (fn [i instr] (format "%4d  %s" i (pr-str instr))) instrs)
                    [(str "; results " (pr-str results))])))
