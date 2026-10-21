(ns memo.handbook-test
  "The parity corpus, part 2: memo's Handbook (memo-mcp/memo/Handbook.md),
  entry by entry. handbook.edn lists every entry (a code block, heading/k, or
  a table row that starts with code, heading/code) and the outcomes of the
  Python twins in test/fixtures/handbook.py, each of which ports some
  entries. Each Python twin has a twin of the same name here, a defmemo or a
  form that must not compile; every entry is ported by a twin, covered by a
  case of another fixture, or has no counterpart in memo.clj, with the
  reason."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

;; handbook.py's module

(def X [0 1 2])
(def Y [0 1 2])
(def N [0 1 2 3])
(def A [0 1 2])
(def Bias [0.1 0.3 0.5 0.7 0.9])
(def Signed [-2 -1 0 1 2])
(defenum Action WAIT JUMP RUN)
(def Actions Action)
(defenum Roles HOST GUEST COOK)
(defenum Stores BAKERY BANK CAFE)
(defenum Bool FALSE TRUE)
(defenum Loc BASKET BOX)

(defn utility [a] (nth [0.0 1.0 3.0] a))
(defn joint [x y] (+ 1.0 (* x y)))
(defn is-open [s] (not (== s 1)))
(defn f [x] (Math/cos x))
(defn g2 [x y] (Math/exp (- (Math/pow (- x y) 2))))
(defn ber-pmf [k p] (if (== k 1) p (- 1 p)))                ; jax.scipy.stats.bernoulli.pmf

;; Anatomy of a memo

(defmemo model [[x X] [y Y]] [a b (c 3.14)]
  " This is a model of ____! "
  (alice :chooses [x X] :wpp (exp (* a x)))
  (bob :chooses [y Y] :wpp (exp (* b y c)))
  (return (Pr (= alice.x x) (= bob.y y))))

;; chooses

(defmemo chooses-softmax [[a Actions]] [β]
  (bob :chooses [a Actions] :wpp (exp (* β (utility a))))
  (return (Pr (= bob.a a))))

(defmemo chooses-bernoulli [[x Bool]] []
  (alice :chooses [x Bool] :wpp (ber-pmf x 0.9))
  (return (Pr (= alice.x x))))

(defmemo chooses-joint [[x X] [y Y]] []
  (bob :chooses [x X] [y Y] :wpp (joint x y))
  (return (Pr (= bob.x x) (= bob.y y))))

(defmemo chooses-to-maximize [[a Actions]] []
  (bob :chooses [a Actions] :to-maximize (utility a))
  (return (Pr (= bob.a a))))

(defmemo chooses-to-minimize [[a Actions]] []
  (bob :chooses [a Actions] :to-minimize (utility a))
  (return (Pr (= bob.a a))))

(defmemo chooses-to-be [[a Actions]] []
  (bob :chooses [a Actions] :to-be #{Action.JUMP})
  (return (Pr (= bob.a a))))

(defmemo chooses-to-be-twice [[a1 Actions] [a2 Actions]] []
  (bob :chooses [a1 Actions] [a2 Actions] :to-be #{Action.JUMP})   ; Bob jumps twice
  (return (Pr (= bob.a1 a1) (= bob.a2 a2))))

(defmemo alias-given [[r Roles]] []
  (bob :given [r Roles] :wpp 1)
  (return (Pr (= bob.r r))))

(defmemo alias-draws [[r Roles]] []
  (bob :draws [r Roles] :wpp (+ r 1))
  (return (Pr (= bob.r r))))

(defmemo alias-assigned [[r Roles]] []
  (bob :assigned [r Roles] :wpp (+ r 1))
  (return (Pr (= bob.r r))))

(defmemo alias-guesses [[r Roles]] []
  (bob :guesses [r Roles] :wpp (+ r 1))
  (return (Pr (= bob.r r))))

(defmemo chooses-such-that [[s Stores]] []
  (bob :chooses [s Stores] :such-that (is-open s))
  (return (Pr (= bob.s s))))

(defmemo chooses-uniformly [[s Stores]] []
  (bob :chooses [s Stores] :uniformly)
  (return (Pr (= bob.s s))))

;; thinks, observes, knows

(defmemo thinks-two-agents [] []
  (bob :thinks
    (alice :chooses [x X] :wpp 1)
    (charlie :chooses [y Y] :wpp (+ y 1)))
  (return (as bob (E (+ alice.x charlie.y)))))

(defmemo observes-axis [[y X]] []
  (bob :thinks (alice :chooses [x X] :wpp 1) (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :observes alice.x :is y)
  (return (as bob (E alice.z))))

(defmemo observes-own-choice-dotted [] []
  (bob :thinks (alice :chooses [x X] :wpp 1) (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :chooses [y X] :wpp (+ y 1))
  (bob :observes alice.x :is bob.y)
  (return (E (as bob (E alice.z)))))

(defmemo observes-other-agent [] []
  (charlie :chooses [y X] :wpp (+ y 1))
  (bob :thinks (alice :chooses [x X] :wpp 1) (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :observes alice.x :is charlie.y)
  (return (E (as bob (E alice.z)))))

(defmemo observes-that-bias [] []
  (bob :thinks (coin :chooses [bias Bias] :wpp 1))
  (bob :observes-that (> coin.bias 0.5))
  (return (as bob (E coin.bias))))

(defmemo observes-event-bias [] []
  (bob :thinks (coin :chooses [bias Bias] :wpp 1))
  (bob :observes-event :wpp coin.bias)
  (return (as bob (E coin.bias))))

(defmemo knows-two [[x X]] []
  (alice :chooses [y Y] :wpp (+ y 1))
  (bob :knows x alice.y)
  (bob :chooses [z X] :wpp (+ 1 (* 2 (= z (% (+ x alice.y) 3)))))
  (return (E bob.z)))

(defmemo knows-short [] []
  (alice :chooses [y Y] :wpp (+ y 1))
  (bob :knows alice.y)
  (return (E (as bob (E alice.y)))))

(defmemo knows-expanded [] []
  (alice :chooses [y Y] :wpp (+ y 1))
  (bob :thinks (alice :chooses [y Y] :wpp 1))
  (bob :observes alice.y :is alice.y)
  (return (E (as bob (E alice.y)))))

;; snapshots, wants, forgets-about, has-theory, inspect

(defmemo snapshot-aliases [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice younger_alice)
  (return (as alice (+ (as past_alice (E bob.x)) (as younger_alice (Pr (= bob.x 2)))))))

(defmemo snapshot-unaffected [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E bob.x)))))

(defmemo wants-to-win [[a A]] []
  (alice :thinks (bob :chooses [score N] :wpp 1))
  (alice :wants win (> score bob.score))
  (alice :chooses [a A] :to-maximize (EU win))
  (alice :given [score N] :wpp (exp (- (** (- score a 1) 2))))
  (return (Pr (= alice.a a))))

(defmemo forgets-about-not-yet [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :to-be x)
  (return (+ (E alice.y) (Var alice.y))))

(defmemo forgets-about-x [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :to-be x)
  (alice :forgets-about x)
  (return (+ (E alice.y) (Var alice.y))))

(defmemo forgets-about-three [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :wpp (+ 1 x))
  (alice :chooses [z Bool] :wpp (+ 1 y))
  (alice :chooses [w Bool] :to-be z)
  (alice :forgets-about x y z)
  (return (E alice.w)))

(defmemo child-thinks [] []
  (child :has-theory :child)
  (child :thinks (sally :chooses [loc Loc] :wpp (+ 1 loc)))
  (return (as child (E sally.loc))))

(defmemo ^{:variant :child} child-variant-chooses [] []
  (child :chooses [loc Loc] :wpp (+ 1 loc))
  (return (E child.loc)))

(def inspect-alice-form
  '(defmemo inspect-alice [] []
     (alice :chooses [x X] :wpp 1)
     (alice :inspect)
     (return (E alice.x))))

(def inspect-alice-printed
  "What (alice :inspect) printed while inspect-alice compiled."
  (with-out-str (eval inspect-alice-form)))

;; expressions

(defmemo literal [] []
  (return 3.14))

(defmemo parameters [] [a b c]
  (return (+ a (* b c))))

(defmemo op-add [[x X] [y Y]] [] (return (+ x y)))
(defmemo op-sub [[x X] [y Y]] [] (return (- x y)))
(defmemo op-mul [[x X] [y Y]] [] (return (* x y)))
(defmemo op-div [[x X] [y Y]] [] (return (/ x (+ y 1))))
(defmemo op-pow [[x X] [y Y]] [] (return (** x (+ y 0.5))))
(defmemo op-mod [[x Signed] [y Y]] [] (return (% x (+ y 1))))
(defmemo op-mod-float [[x Signed]] [] (return (% (+ x 0.5) 2)))
(defmemo op-eq [[x X] [y Y]] [] (return (= x y)))
(defmemo op-neq [[x X] [y Y]] [] (return (not= x y)))
(defmemo op-lt [[x X] [y Y]] [] (return (< x y)))
(defmemo op-lte [[x X] [y Y]] [] (return (<= x y)))
(defmemo op-gt [[x X] [y Y]] [] (return (> x y)))
(defmemo op-gte [[x X] [y Y]] [] (return (>= x y)))
(defmemo op-and [[x X] [y Y]] [] (return (and (> x 0) (> y 1))))
(defmemo op-or [[x X] [y Y]] [] (return (or (> x 0) (> y 1))))
(defmemo op-xor [[x X] [y Y]] [] (return (xor (> x 0) (> y 1))))
(defmemo op-xor-int [[x X] [y Y]] [] (return (xor x y)))
(defmemo op-neg [[x X]] [] (return (- x)))
(defmemo op-uplus [[x X]] [] (return (+ x)))
(defmemo op-inv [[x X]] [] (return (not (> x 1))))
(defmemo op-inv-int [[x X]] [] (return (not x)))
(defmemo fn-exp [[x X]] [] (return (exp x)))
(defmemo fn-log [[x X]] [] (return (log (+ x 1))))
(defmemo fn-abs [[x Signed]] [] (return (abs x)))

(defmemo conditional [[x X]] [a b]
  (return (if (> x 1) a b)))

(defmemo custom-function [[x X]] []
  (alice :chooses [y Y] :wpp (+ 1 (f y)))
  (return (+ (E alice.y) (f x))))

(defmemo own-choices [[x X] [y Y]] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (Pr (= alice.x x) (= alice.y y))))

(defmemo dot-notation [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (+ alice.x alice.y))))

(defmemo bracket-each [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (+ (as alice x) (as alice y)))))

(defmemo bracket-whole [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (as alice (+ x y)))))

(defmemo expectation [] []
  (alice :chooses [x X] :wpp (+ x 1))
  (bob :chooses [z X] :wpp 1)
  (return (E (+ alice.x bob.z))))

(defmemo variance [] []
  (alice :chooses [y Signed] :wpp (+ 1 (* y y)))
  (return (Var (* alice.y 2))))

(defmemo probability [] []
  (alice :chooses [y Signed] :wpp (+ 3 y))
  (return (Pr (>= alice.y 0))))

(defmemo joint-probability [] []
  (a :chooses [x Signed] :wpp 1)
  (b :chooses [y Signed] :wpp (+ 3 y))
  (return (Pr (> a.x 0) (< b.y 2))))

(defmemo entropy [] []
  (alice :chooses [x X] :wpp 1)
  (bob :knows alice.x)
  (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y alice.x))))
  (return (H alice.x bob.y)))

(defmemo kl-divergence [] []
  (alice :chooses [x X] :wpp (+ x 1))
  (bob :chooses [y X] :wpp 1)
  (return (KL alice.x bob.y)))

(defmemo query-variance [] []
  (alice :chooses [x Signed] :wpp 1)
  (return (Var (as alice (* (abs x) 2)))))

(defmemo query-known [] []
  (bob :chooses [y N] :wpp (+ y 1))
  (alice :knows bob.y)
  (return (E (as alice (= bob.y 3)))))

(defmemo imagine-observation [] []
  (alice :thinks
    (bob :chooses [x N] :wpp 1)
    (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y (% x 3))))))
  (return (imagine
           (bob :chooses [y Y] :wpp 1)
           (alice :observes bob.y :is bob.y)
           (E (as alice (Pr (= bob.x 3)))))))

;; calling other memos

(defmemo f3 [[x X]] [a b c]
  (alice :chooses [x X] :wpp (exp (+ (* a x) b)))
  (return (* (Pr (= alice.x x)) c)))

(defmemo g-calls-f [] []
  (alice :chooses [x X] :wpp (f3 [x] 1.0 0.0 3.1))
  (return (E alice.x)))

(defmemo g-forwards [] [a b c]
  (alice :chooses [x X] :wpp (f3 [x] ...))
  (return (E alice.x)))

(defmemo g2-returns [[x X] [y Y]] [t]
  (return (+ x (* y t)))
  (return (* x y)))

(defmemo g2-first [] []
  (alice :chooses [x X] [y Y] :wpp (g2-returns 0 [x y] 3.14))
  (return (E (+ alice.x alice.y))))

;; the inline escape hatch

(defmemo inline-member [] []
  (return #{Action.WAIT}))

(defmemo inline-shift [] [x]
  (return #{(bit-shift-left x 4)}))

;; running and configuring

(defmemo executed [[x X] [y Y]] [a b]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (+ (Pr (= alice.x x)) (* b y)))
  (return (E alice.x)))

(defmemo ^:cache cached [] [a]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (E alice.x)))

(defmemo differentiable [] [a b]
  (alice :chooses [x X] :wpp (exp (+ (* a x) (* b x x))))
  (return (E alice.x)))

;; twins that must not compile

(def compile-forms
  {:observes-own-choice
   '(defmemo observes-own-choice [] []
      (bob :thinks (alice :chooses [x X] :wpp 1) (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
      (bob :chooses [y X] :wpp (+ y 1))
      (bob :observes alice.x :is y)
      (return (E (as bob (E alice.z)))))
   :observes-literal
   '(defmemo observes-literal [] []
      (bob :thinks (alice :chooses [x X] :wpp 1))
      (bob :observes alice.x :is 7)
      (return 0))
   :forgets-about-then-x
   '(defmemo forgets-about-then-x [] []
      (alice :chooses [x Bool] :uniformly)
      (alice :chooses [y Bool] :to-be x)
      (alice :forgets-about x)
      (return (E alice.x)))
   :child-thinks-of-thinking
   '(defmemo child-thinks-of-thinking [] []
      (child :has-theory :child)
      (child :thinks (sally :thinks (ball :chooses [loc Loc] :wpp 1)))
      (return 0))
   :child-variant-thinks
   '(defmemo ^{:variant :child} child-variant-thinks [] []
      (child :thinks (sally :chooses [loc Loc] :wpp 1))
      (return 0))
   :query-unknown
   '(defmemo query-unknown [] []
      (alice :thinks (bob :chooses [y N] :wpp 1))
      (return (as alice (= bob.y 3))))
   :imagine-as-written
   '(defmemo imagine-as-written [] []
      (alice :thinks
        (bob :chooses [x N] :wpp 1)
        (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y (% x 3))))))
      (return (imagine
               (bob :chooses [y Y] :wpp 1)
               (alice :observes bob.y :is bob.y)
               (as alice (Pr (= bob.x 3))))))
   :inline-agent
   '(defmemo inline-agent [] []
      (alice :chooses [x X] :wpp 1)
      (return #{alice.x}))})

(def memo-clj-messages
  "Where memo.clj's compile error differs from Python memo's on purpose."
  {})

(def divergences
  "Where memo.clj does not do what Python memo does, on purpose: memo.clj's
  error, at compile time or when called, and why."
  {:op-xor-int {:call-error "memo.clj computes xor on booleans only"
                :why "no bitwise operators (DESIGN.md §5): Python memo's ^ on integers is bitwise"}
   :op-inv-int {:call-error "memo.clj computes not on booleans only"
                :why (str "no bitwise operators (DESIGN.md §5): Python memo's ~x is True ^ x, so ~2 is 3, "
                          "not the bitwise inversion the Handbook describes")}
   :inline-agent {:compile-error "Cannot reference alice.x inside the inline escape hatch #{...}"
                  :why (str "an inline form is Clojure, compiled with its defmemo, so the Handbook's rule "
                            "(no memo constructs inside braces) is checked there; Python memo's NameError "
                            "comes when the memo is called")}})

(def covered-by
  "Entries a case of another fixture ports: [fixture section case]."
  {})

(def not-applicable
  "Entries with no counterpart in memo.clj, and why."
  {"cost-reflection/1" "cost reflection (time @, space @, size @) is M4 (DESIGN.md §1)"
   "executing-a-memo/3" (str "return_pandas and return_xarray give Python data frames; memo.clj returns "
                             "gen.array arrays (gen.array/->clj reads them)")
   "executing-a-memo/4" "save_comic draws the model with Graphviz: not in memo.clj"
   "decorator-options/2" "debug_trace prints each memo call with its timing: not in memo.clj"})

;; the checks

(def fixture (ts/fixture "handbook.edn"))

(defn- twin [k] (some-> (ns-resolve 'memo.handbook-test (symbol (name k))) deref))

(defn- call-args [{:keys [args kwargs]}]
  (cond-> (vec args) (seq kwargs) (conj kwargs)))

(defn- table-rows
  "A printed table's rows of cells (its | lines)."
  [s]
  (for [line (str/split-lines s) :when (str/starts-with? line "|")]
    (mapv str/trim (str/split (subs line 1 (dec (count line))) #"\|"))))

(defn- same-cell? [a b]
  (let [x (parse-double a) y (parse-double b)]
    (if (and x y) (ts/close? x y) (= a b))))

(defn- same-table? [python ours]
  (let [p (table-rows python) o (table-rows ours)]
    (and (= (count p) (count o))
         (every? (fn [[pr or]] (and (= (count pr) (count or)) (every? true? (map same-cell? pr or))))
                 (map vector p o)))))

(defn- thrown [thunk] (try (thunk) nil (catch Throwable e e)))

(defn- check-call [k {:keys [kind argnums value grad shape table error] :as c}]
  (testing (str kind " " (pr-str (call-args c)))
    (let [m (twin k)
          call #(apply m (call-args c))]
      (cond
        (get-in divergences [k :call-error])
        (let [e (thrown call)]
          (is (err/memo-error? e))
          (is (= (get-in divergences [k :call-error]) (ex-message e))))

        error
        (is (= (get memo-clj-messages k error) (some-> (thrown call) ex-message)))

        (= "grads" kind)
        (let [[v g] (apply (array/value-and-grad m {:argnums argnums}) (call-args c))]
          (is (ts/close? value (ts/host v)))
          (if (sequential? grad)
            (is (every? true? (map #(ts/grad-close? %1 (ts/host %2)) grad g)) (str "jax " grad))
            (is (ts/grad-close? grad (ts/host g)) (str "jax " grad " memo.clj " (ts/host g)))))

        :else
        (let [r (call)]
          (is (= shape (array/shape r)))
          (is (ts/close? value (ts/host r)) (str "python " (pr-str value) "\nmemo.clj " (pr-str (ts/host r))))
          (when (= "table" kind)
            (let [ours (with-out-str (memo/print-table m r))]
              (is (same-table? table ours) (str "python\n" table "memo.clj\n" ours)))))))))

(deftest every-twin
  (doseq [{k :name :keys [outcome message calls doc]} (:twins fixture)]
    (testing (name k)
      (cond
        (= "ce" outcome)
        (is (= (get memo-clj-messages k message)
               (some-> (ts/compile-error 'memo.handbook-test (compile-forms k)) ex-message)))

        (get-in divergences [k :compile-error])
        (is (= (get-in divergences [k :compile-error])
               (some-> (ts/compile-error 'memo.handbook-test (compile-forms k)) ex-message)))

        :else
        (do (when doc
              (is (= (str/trim doc) (str/trim (:memo/doc (meta (twin k)))))))
            (doseq [c calls] (check-call k c)))))))

(deftest what-inspect-prints
  (let [python (:compile-output (first (filter #(= :inspect-alice (:name %)) (:twins fixture))))
        rest-lines #(rest (str/split-lines %))]
    (testing "the first line names the source: a file and line there, the defmemo here"
      (is (str/starts-with? inspect-alice-printed "** Tracing alice in defmemo inspect-alice")))
    (is (= (rest-lines python) (rest-lines inspect-alice-printed)))))

(deftest a-cached-memo-returns-its-table
  (is (identical? (cached 0.5) (cached 0.5))))

(deftest every-entry-is-accounted-for
  (let [ids (map :id (:entries fixture))
        ported (set (mapcat :entries (:twins fixture)))]
    (is (= 79 (count ids)) "Handbook.md at memo 1.3.3")
    (doseq [id ids]
      (is (or (ported id) (covered-by id) (not-applicable id)) (str id " is not accounted for")))
    (doseq [[id [file section case]] covered-by]
      (is (some #{case} (map :model (section (ts/fixture file)))) (str id ": no " case " in " file)))
    (is (every? (set ids) (concat (keys covered-by) (keys not-applicable)))
        "a reason for an entry the Handbook does not have")
    (doseq [{k :name} (:twins fixture)]
      (is (or (twin k) (compile-forms k)) (str (name k) " has no twin")))
    (is (every? (set (map :name (:twins fixture)))
                (concat (keys compile-forms) (keys divergences) (keys memo-clj-messages)))
        "a form, divergence or message for a twin handbook.py does not have")))
