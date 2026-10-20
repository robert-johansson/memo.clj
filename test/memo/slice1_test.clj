(ns memo.slice1-test
  "M1, slice 1: one frame, a wpp choice, Pr and E. Every model here has a twin
  of the same name in test/fixtures/generate.py, and slice1.edn holds Python
  memo's results for it."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [memo.core :refer [defmemo]]
            [memo.domains :refer [defenum]]
            [memo.errors :as err]
            [memo.test-support :as ts]))

(alter-var-root #'err/*warnings?* (constantly false))

(defenum Coin HEADS TAILS)
(def N [0 1 2 3 4 5])

(defmemo coin-flip [[c Coin]] [p]
  (alice :chooses [c Coin] :wpp (if (= c Coin.HEADS) p (- 1 p)))
  (return (Pr (= alice.c c))))

(defmemo mean-roll [] [beta]
  (alice :chooses [n N] :wpp (exp (* beta n)))
  (return (E alice.n)))

(defmemo roll-table [[n N]] [beta]
  (alice :chooses [n N] :wpp (exp (* beta n)))
  (return (Pr (= alice.n n)))
  (return (E alice.n)))

(defmemo roll-stats [] [beta]
  (alice :chooses [n N] :wpp (exp (* beta n)))
  (return (Var alice.n))
  (return (Pr (> alice.n 2)))
  (return (E (* alice.n alice.n))))

(defmemo argmax-pick [[n N]] []
  (alice :chooses [n N] :to-maximize (- (abs (- n 2.5))))
  (return (Pr (= alice.n n))))

(defmemo argmin-pick [[n N]] []
  (alice :chooses [n N] :to-minimize (** (- n 2) 2))
  (return (Pr (= alice.n n))))

(defmemo two-choices [[x Coin] [y Coin]] [p]
  (alice :chooses [x Coin] [y Coin] :wpp (* (if (= x y) p (- 1 p)) (+ 1 x)))
  (return (Pr (= alice.x x) (= alice.y y))))

(defmemo two-agents [] [p]
  (alice :chooses [x Coin] :wpp (if (= x Coin.HEADS) p (- 1 p)))
  (bob :chooses [y Coin] :wpp 1)
  (return (Pr (= alice.x bob.y))))

(defmemo zero-mass [] []
  (alice :chooses [n N] :wpp (* 0 n))
  (return (+ (E alice.n) (Pr (= alice.n 0)) 10)))

(defmemo const-return [[n N] [c Coin]] []
  (return 1.0))

(defmemo axis-return [[n N] [c Coin]] []
  (return (+ (* n 10) c)))

(defmemo such-that-pick [] []
  (alice :chooses [n N] :such-that (< n 3))
  (return (E alice.n)))

(defmemo uniform-pick [] []
  (alice :chooses [n N] :uniformly)
  (return (E alice.n)))

(defmemo to-be-pick [] []
  (alice :chooses [n N] :to-be 4)
  (return (E alice.n)))

(defmemo bool-sum [] []
  (alice :chooses [n N] :wpp (+ (= n 0) (< n 2)))
  (return (Pr (= alice.n 0))))

(defmemo default-param [] [(beta 0.5)]
  (alice :chooses [n N] :wpp (exp (* beta n)))
  (return (E alice.n)))

(defmemo exp-overflow [] [beta]
  (alice :chooses [n N] :wpp (* 1.0 (exp (* beta n))))
  (return (+ (E alice.n) (Pr (= alice.n 5)))))

(defmemo static-ite [] [p]
  (alice :chooses [n N] :wpp (if (> p 0) (+ n 1) (- 6 n)))
  (return (E alice.n)))

(defmemo ops-pick [[n N]] []
  (alice :chooses [n N] :wpp (+ (xor (< n 3) (= (mod n 2) 0))
                                (* (not (= n 5)) 0.5)
                                (/ (** n 2) 10)))
  (return (Pr (= alice.n n))))

(def fixture (ts/fixture "slice1.edn"))

(deftest results-match-python-memo
  (is (= "711e4cdde9cb1764815c17a0b3204122bc3873d3" (get-in fixture [:oracle :commit])))
  (doseq [{:keys [model args shape value]} (:cases fixture)]
    (testing (str (name model) " " args)
      (let [f @(ns-resolve 'memo.slice1-test (symbol (name model)))
            result (try (apply f args) (catch Exception e e))]
        (if (instance? Exception result)
          (is false (str "threw: " (ex-message result)))
          (do (is (= shape (array/shape result)))
              (is (ts/close? value (ts/host result))
                  (str "python " (pr-str value) "\nmemo.clj " (pr-str (ts/host result))))))))))

(def error-forms
  {:return-uncertain
   '(defmemo return-uncertain [] []
      (alice :chooses [n N] :uniformly)
      (return alice.n))
   :unknown-choice
   '(defmemo unknown-choice [] []
      (alice :chooses [n N] :wpp m)
      (return 1))
   :repeated-choice
   '(defmemo repeated-choice [] []
      (alice :chooses [n N] :uniformly)
      (alice :chooses [n N] :uniformly)
      (return 1))
   :name-conflict
   '(defmemo name-conflict [] [n]
      (alice :chooses [n N] :uniformly)
      (return 1))
   :other-agent-unknown
   '(defmemo other-agent-unknown [] []
      (bob :chooses [m N] :uniformly)
      (alice :chooses [n N] :wpp bob.m)
      (return 1))})

(deftest compile-errors-match-python-memo
  (doseq [[k message] (:errors fixture)]
    (testing (name k)
      (is (= message (some-> (ts/compile-error 'memo.slice1-test (error-forms k)) ex-message))))))

(deftest gradients
  (testing "d/dbeta E[n] is Var[n] in this exponential family, and matches finite differences"
    (let [g ((array/grad mean-roll) 0.5)
          h 1e-3
          fd (/ (- (ts/host (mean-roll (+ 0.5 h))) (ts/host (mean-roll (- 0.5 h)))) (* 2 h))]
      (is (ts/close? (first (ts/host (roll-stats 0.5))) (ts/host g)))
      ;; central differences: O(h^2) truncation, and float32 rounding on MLX
      (is (< (Math/abs (- fd (ts/host g))) 1e-3))))
  (testing "through a parameter in a static condition and the posterior shortcut"
    (let [g ((array/grad (fn [p] (array/sum (array/* (coin-flip p) (array/array [1.0 0.0]))))) 0.3)]
      (is (ts/close? 1.0 (ts/host g))))))

(deftest divergences-from-python-memo
  (testing "self cannot name an agent (Python memo 1.3.2 accepts it; DESIGN.md §8)"
    (is (= "`self` cannot be the name of an agent"
           (ex-message (ts/compile-error 'memo.slice1-test
                                         '(defmemo self-agent [] []
                                            (self :chooses [n N] :uniformly)
                                            (return 1))))))))

(deftest loud-errors
  (testing "booleans have no - in JAX, so a model that subtracts them fails loudly"
    (let [f (do (ts/compile-error 'memo.slice1-test
                                  '(defmemo bool-minus [] []
                                     (alice :chooses [n N] :wpp (- (< n 3) (= n 0)))
                                     (return (E alice.n))))
                @(ns-resolve 'memo.slice1-test 'bool-minus))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Unsupported operand type for -" (f)))))
  (testing "a construct that is not built yet says so"
    (is (re-find #"not supported in memo.clj yet"
                 (ex-message (ts/compile-error 'memo.slice1-test
                                               '(defmemo later [] []
                                                  (alice :snapshots-self-as past-alice)
                                                  (return 1)))))))
  (testing "a parameter must be a scalar"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"was not a numeric scalar"
                          (mean-roll (array/array [0.1 0.2]))))))

(deftest outside-defmemo
  (is (= 1 (Coin :TAILS)))
  (is (= [0 1] (:values Coin))))
