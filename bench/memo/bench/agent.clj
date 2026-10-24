(ns memo.bench.agent
  "The speed research's benchmark (bean memoclj-c0xj; DESIGN.md §6.4,
  question 3): GenGDT's memo agent, its shape copied from
  gengdt.memo-agent and gengdt.synthetic-agent (GenGDT's working copy,
  2026-10-23), since memo.clj cannot depend on GenGDT.

  The mind, choose, is called once per trial of an Unfold of n cycles,
  and again at every evaluation inference makes. The same life runs with
  four minds, so their differences say where the time goes:

    :memo-dist    (mg/dist choose): the guard, memo's lowering, the checks
    :memo-plain   (d/categorical (choose ...)): memo without memo.gen
    :hand         the same table in gen.array's operators, no memo
    :fixed        a constant table: gen.clj's own cost, no mind

  clojure -M:bench -m memo.bench.agent [n]         on the JVM backend
  clojure -M:bench:mlx -m memo.bench.agent [n]     on MLX (one GPU JVM at a time)"
  (:require [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.gen :as mg]))

;; ---------------------------------------------------------------------------
;; The synthetic agent's content (gengdt.synthetic-agent), in host code
;; ---------------------------------------------------------------------------

(def PASS 0)
(def FIX 1)
(def FLEE 2)

(def theta {:beta 4.0 :q 0.6 :v 2.0 :kappa 2.0 :sd 1.0})
(def world {FIX 0.9 FLEE 0.2})

(defn satisfaction ^double [s] (- 3.0 (double s)))

(defn initial-state [{:keys [kappa]}]
  {:s 2 :counts (vec (repeat 3 (vec (repeat 4 [(/ kappa 2) (/ kappa 2)]))))})

(defn expectancy [counts o s]
  (let [[succ fail] (get-in counts [o s])]
    (/ succ (+ succ fail))))

(defn outcome-probs [world s o]
  (if (= o PASS)
    (assoc [0.0 0.0 0.0 0.0] s 1.0)
    (let [p (get world o)]
      (-> [0.0 0.0 0.0 0.0] (assoc (dec s) p) (assoc s (- 1 p))))))

(defn learn [counts s o s2]
  (if (= o PASS) counts (update-in counts [o s (if (= s2 (dec s)) 0 1)] inc)))

(defn next-state [{:keys [counts]} s o s2]
  {:s (if (zero? s2) 2 s2) :counts (learn counts s o s2)})

;; ---------------------------------------------------------------------------
;; The mind (gengdt.memo-agent)
;; ---------------------------------------------------------------------------

(defenum Opt PASS FIX FLEE)
(defenum Bool NO YES)

(defn ^:memo/vectorized eu
  [o v s m-fix m-flee]
  (let [sat (a/- 3.0 s)
        m (a/where (a/== o 1) m-fix m-flee)]
    (a/* v (a/where (a/== o 0) sat (a/+ (a/* m (a/- 4.0 s)) (a/* (a/- 1.0 m) sat))))))

(defn available [o fix-on flee-on]
  (cond (= o 0) 1 (= o 1) fix-on :else flee-on))

(defmemo choose [[a Opt]] [beta q v s m-fix m-flee]
  (agent :chooses [fixon Bool] :wpp (+ (* fixon q) (* (- 1 fixon) (- 1 q))))
  (agent :chooses [fleeon Bool] :wpp (+ (* fleeon q) (* (- 1 fleeon) (- 1 q))))
  (agent :chooses [o Opt] :wpp (* (available o fixon fleeon) (exp (* beta (eu o v s m-fix m-flee)))))
  (return (Pr (= agent.o a))))

(def ^:private options (a/array [0 1 2]))
(def ^:private access-masks (a/array [[1.0 0.0 0.0] [1.0 1.0 0.0] [1.0 0.0 1.0] [1.0 1.0 1.0]]))

(defn choose-by-hand
  "choose's table in gen.array's operators: the four access patterns (FIX
  and FLEE each in mind with probability q), softmax(beta · EU) over the
  options in mind, summed over the patterns. s, m-fix and m-flee may have
  a leading batch axis ([B 1]): the tables of B trials, [B 3]."
  [beta q v s m-fix m-flee]
  (let [w (a/exp (a/* beta (eu options v s m-fix m-flee)))
        mw (a/* access-masks (a/expand-dims w -2))
        p (a// mw (a/sum mw {:axis -1 :keep-dims? true}))
        nq (a/- 1.0 q)
        pattern (a/stack [(a/* nq nq) (a/* q nq) (a/* nq q) (a/* q q)])]
    (a/sum (a/* (a/expand-dims pattern -1) p) {:axis -2})))

(def minds
  "Each mind: from choose's arguments to a distribution over Opt."
  {:memo-dist (mg/dist choose)
   :memo-plain (fn [& args] (d/categorical (apply choose args)))
   :hand (fn [& args] (d/categorical (apply choose-by-hand args)))
   :fixed (let [t (a/array [0.2 0.5 0.3])] (fn [& _] (d/categorical t)))})

;; ---------------------------------------------------------------------------
;; The person across cycles, and the scientist's model (fit-beta-v)
;; ---------------------------------------------------------------------------

(defn- host [x] (a/item x))

(defn report-mean [{:keys [v]} counts s o s2]
  (a/- (a/* v (satisfaction s2))
       (eu o v s (expectancy counts FIX s) (expectancy counts FLEE s))))

(defn model
  "life and fit-beta-v with this mind."
  [mind]
  (let [cycle-gf (gen [_t st theta world]
                   (let [{:keys [s counts]} st
                         o (host (trace :a (mind (:beta theta) (:q theta) (:v theta) s
                                                 (expectancy counts FIX s)
                                                 (expectancy counts FLEE s))))
                         s2 (host (trace :s2 (d/categorical (outcome-probs world s o))))]
                     (trace :y (d/normal (report-mean theta counts s o s2) (:sd theta)))
                     (next-state st s o s2)))
        life (comb/unfold cycle-gf)]
    {:life life
     :fit-beta-v (gen [n theta world]
                   (let [beta (trace :beta (d/uniform 0.5 10.0))
                         v (trace :v (d/uniform 0.5 5.0))]
                     (splice :life life n (initial-state theta) (assoc theta :beta beta :v v) world)))}))

(defn observations
  "n cycles of the agent (the memo mind), as fit-beta-v's constraints."
  [n key]
  (let [{:keys [life]} (model (:memo-dist minds))
        tr (gfi/simulate life [n (initial-state theta) theta world] key)]
    (reduce cm/merge cm/EMPTY
            (for [t (range n) k [:a :s2 :y]]
              (cm/choicemap [:life t k] (host (tr [t k])))))))

;; ---------------------------------------------------------------------------
;; Timing
;; ---------------------------------------------------------------------------

(defn measure
  "The time per call in ms of each thunk of cases (a map; a thunk reads its
  result on the host): rounds in which every case runs reps calls in turn,
  so that no case gains from running after the others while the JIT warms
  code they share. Warm rounds for warm-ms, then rounds timed; the median
  and the minimum over the timed rounds."
  [cases {:keys [warm-ms rounds reps] :or {warm-ms 5000 rounds 15 reps 1}}]
  (let [round (fn [acc]
                (reduce-kv (fn [acc k f]
                             (let [t0 (System/nanoTime)]
                               (dotimes [_ reps] (f))
                               (update acc k conj (/ (- (System/nanoTime) t0) 1e6 reps))))
                           acc cases))
        end (+ (System/nanoTime) (* warm-ms 1000000))]
    (while (< (System/nanoTime) end) (round {}))
    (into (sorted-map)
          (map (fn [[k ts]] (let [ts (sort ts)] [k {:median (nth ts (quot (count ts) 2)) :min (first ts)}])))
          (nth (iterate round (zipmap (keys cases) (repeat []))) rounds))))

(defn- args-at
  "choose's arguments at the agent's start, with beta and v as arrays (as
  the scientist's trace holds them)."
  []
  (let [{:keys [s counts]} (initial-state theta)]
    [(a/array 3.0) (:q theta) (a/array 1.5) s (expectancy counts FIX s) (expectancy counts FLEE s)]))

(defn per-call
  "One mind call (with its log-prob), and one log-prob's gradient in beta
  and v, per mind: {[mind what] time}."
  []
  (let [args (args-at)]
    (measure (into {}
                   (for [[k mind] minds
                         [what f] {:call #(host (dc/log-prob (apply mind args) 1))
                                   :grad (let [g (a/grad (fn [beta v]
                                                           (dc/log-prob (apply mind (assoc args 0 beta 2 v)) 1))
                                                         {:argnums [0 1]})]
                                           #(host (first (g (args 0) (args 2)))))}]
                     [[k what] f]))
             {:reps 200})))

(defn whole-life
  "fit-beta-v over n cycles, per mind: the exact likelihood (generate), the
  gradient in beta and v (choice-gradients) and an HMC step (L = 10):
  {[mind what] time}."
  [n]
  (let [obs (observations n (rng/key 21))
        cs (cm/merge obs (cm/choicemap :beta 3.0 :v 1.5))
        args [n theta world]
        s (sel/select :beta :v)]
    (measure (into {}
                   (for [[k mind] minds
                         :let [{:keys [fit-beta-v]} (model mind)
                               tr (:trace (gfi/generate fit-beta-v args cs (rng/key 0)))]
                         [what f] {:generate #(host (:weight (gfi/generate fit-beta-v args cs (rng/key 0))))
                                   :gradient #(host (cm/get-choice (:gradients (grad/choice-gradients tr s)) :v))
                                   :hmc #(host ((:trace (mcmc/hmc tr s {:L 10 :eps 0.15} (rng/key 7))) :v))
                                   :hmc-compile #(host ((:trace (mcmc/hmc tr s {:L 10 :eps 0.15 :compile? true} (rng/key 7))) :v))}]
                     [[k what] f]))
             {:warm-ms 10000 :rounds (parse-long (System/getProperty "bench.rounds" "10"))})))

(defn trial-inputs
  "The mind's inputs at each of n observed cycles, and the observed option:
  for the scientist, who sees every choice and outcome, the states are
  known before any mind is called. [{:s :m-fix :m-flee :a}]."
  [obs n]
  (loop [t 0 st (initial-state theta) out []]
    (if (= t n)
      out
      (let [{:keys [s counts]} st
            o (long (host (cm/get-choice obs [:life t :a])))
            s2 (long (host (cm/get-choice obs [:life t :s2])))]
        (recur (inc t) (next-state st s o s2)
               (conj out {:s s :m-fix (expectancy counts FIX s) :m-flee (expectancy counts FLEE s) :a o}))))))

(defn batched
  "The choices' log-likelihood over n trials, the mind alone: a call per
  trial against one call on the trials' inputs as arrays ([n 1]), with its
  gradient in beta and v. memo has no batched parameters, so only the
  hand-written mind has a batched row. Under gen.array's batch tracer memo
  runs batched (since memoclj-m2jw), but at memo.clj's gen.clj pin a
  gradient cannot reach through a batched run (genclj-hwk1)."
  [n]
  (let [ins (trial-inputs (observations n (rng/key 21)) n)
        col (fn [k] (a/reshape (a/array (mapv (comp double k) ins)) [n 1]))
        [ss mfs mls] (map col [:s :m-fix :m-flee])
        as (a/array (mapv :a ins))
        q (:q theta)
        per-trial (fn [mind]
                    (fn [beta v]
                      (reduce a/+ (map (fn [{:keys [s m-fix m-flee a]}]
                                         (dc/log-prob (mind beta q v s m-fix m-flee) a))
                                       ins))))
        hand-batched (fn [beta v]
                       (let [t (choose-by-hand beta q v ss mfs mls)]
                         (a/sum (a/log (a/take-along-axis t (a/reshape as [n 1]) -1)))))
        fs {:memo-dist (per-trial (:memo-dist minds))
            :hand (per-trial (:hand minds))
            :hand-batched hand-batched}
        beta (a/array 3.0) v (a/array 1.5)]
    (println "log-likelihood of the choices:" (into (sorted-map) (map (fn [[k f]] [k (host (f beta v))])) fs))
    (measure (into {}
                   (concat
                    (for [[k f] fs] [[k :value] #(host (f beta v))])
                    (for [[k f] fs
                          :let [g (a/grad f {:argnums [0 1]})]]
                      [[k :gradient] #(host (first (g beta v)))])))
             {:reps 5})))

(defn compiled
  "The choices' log-likelihood over n trials, the mind alone, as a function
  of arrays only, through array/compile (one fused MLX graph per input
  structure; on the JVM, the function itself) or not, for its value and its
  value-and-grad in beta and v. Each function runs once uncompiled first, so
  memo's constants exist before a trace."
  [n]
  (let [ins (trial-inputs (observations n (rng/key 21)) n)
        col (fn [k] (a/array (mapv (comp double k) ins)))
        [ss mfs mls] (map col [:s :m-fix :m-flee])
        as (mapv :a ins)
        q (:q theta)
        per-trial (fn [table]
                    (fn [beta v ss mfs mls]
                      (reduce a/+ (map (fn [t o]
                                         (a/log (a/take (table beta q v (a/take ss t) (a/take mfs t) (a/take mls t)) o)))
                                       (range n) as))))
        asb (a/reshape (a/array as) [n 1])
        fs {:memo (per-trial choose)
            :hand (per-trial choose-by-hand)
            :hand-batched (fn [beta v ss mfs mls]
                            (let [t (choose-by-hand beta q v (a/reshape ss [n 1]) (a/reshape mfs [n 1])
                                                    (a/reshape mls [n 1]))]
                              (a/sum (a/log (a/take-along-axis t asb -1)))))}
        beta (a/array 3.0) v (a/array 1.5)
        cases (into {}
                    (for [[k f] fs
                          :let [vg (a/value-and-grad f {:argnums [0 1]})
                                _ (f beta v ss mfs mls)]
                          [what g] {:value f :compiled-value (a/compile f)
                                    :gradient (comp second vg) :compiled-gradient (a/compile (comp second vg))}]
                      [[k what] (if (#{:value :compiled-value} what)
                                  #(host (g beta v ss mfs mls))
                                  #(host (first (g beta v ss mfs mls))))]))]
    (println "values:" (into (sorted-map) (for [[[k w] f] cases :when (#{:value :compiled-value} w)] [[k w] (f)])))
    (measure cases {:reps 5})))

(defn- print-table
  "{[mind what] time} as a table of medians (the minimum in parentheses)."
  [title r]
  (let [whats (distinct (map second (keys r)))]
    (println title)
    (println (apply str (format "  %-12s" "") (map #(format "%22s" (name %)) whats)))
    (doseq [k (distinct (map first (keys r)))]
      (println (apply str (format "  %-12s" (name k))
                      (map #(let [{:keys [median min]} (r [k %])]
                              (format "%22s" (format "%.4f (%.4f)" median min)))
                           whats))))))

(defn -main [& [n]]
  (let [n (if n (parse-long n) 150)]
    (println "backend" (System/getProperty "gen.backend" "jvm") "dtype" (a/dtype (a/array 1.0)) "n" n)
    (let [args (args-at)]
      (println "choose      " (a/->clj (apply choose args)))
      (println "by hand     " (a/->clj (apply choose-by-hand args))))
    (print-table "per call (ms)" (per-call))
    (print-table (str "the choices of " n " trials, the mind alone (ms)") (batched n))
    (print-table (str "the same, compiled or not (ms)") (compiled n))
    (when-not (System/getProperty "bench.skip-life")
      (print-table (str "a life of " n " cycles (ms)") (whole-life n)))
    (shutdown-agents)))
