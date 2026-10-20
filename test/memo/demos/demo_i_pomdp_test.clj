(ns memo.demos.demo-i-pomdp-test
  "memo's demo/demo-i-pomdp.ipynb: an interactive POMDP, the trust game with
  guilt, over a product domain of two rounds of history."
  (:require [clojure.test :refer [deftest]]
            [memo.core :refer [defmemo]]
            [memo.domains :as domains :refer [defdomain]]
            [memo.test-support :as ts]))

(def mult 3.0)
(def F [0 1])                 ; actions are fractions :)
(def Fractions [0.2 0.8])
(def G [0.0 0.7])             ; guilts

(defn payout-investor [fi ft]   ; fi: the investor's fraction, ft: the trustee's
  (+ (- 1 (nth Fractions (long fi))) (* mult (nth Fractions (long fi)) (nth Fractions (long ft)))))

(defn payout-trustee [fi ft]
  (* mult (nth Fractions (long fi)) (- 1 (nth Fractions (long ft)))))

(defdomain H                  ; histories: just 2 rounds of history (3 rounds of game)
  :i1 (count F) :t1 (count F)  ; 1st round, investor and trustee
  :i2 (count F) :t2 (count F)) ; 2nd round

(defn reludiff [x y] (max (- x y) 0))
(defn is-init [h] (== h 0))

(defn Tr
  "Step the game: h becomes h_ with moves (fi, ft) at round r."
  [r h fi ft h_]
  (let [z (assoc (domains/fields H h) (* 2 (long r)) (long fi) (inc (* 2 (long r))) (long ft))]
    (== h_ (H (zipmap (:keys H) z)))))

(defmemo ^:cache hist [[ig G] [tg G] [h H]] [r level beta]
  ;; the prior over histories h at the start of round r >= 1, given guilts ig, tg
  (world :knows ig tg)
  ;; start with h at round r - 1
  (world :chooses [h H] :wpp (if (> r 1) (hist [ig tg h] (- r 1) level beta) (is-init h)))
  ;; the investor and trustee move
  (world :chooses [fi F] :wpp (exp (* beta (investor [h ig fi] (- r 1) level beta))))
  (world :chooses [ft F] :wpp (exp (* beta (trustee [h tg fi ft] (- r 1) level beta))))
  ;; h is updated
  (world :chooses [h_ H] :wpp (Tr (- r 1) h fi ft h_))
  (return (Pr (= world.h_ h))))

(defmemo ^:cache investor [[h H] [ig G] [fi F]] [r level beta]
  ;; the investor's Q-function, given h and its own guilt
  (investor :knows ig)
  (investor :thinks
    (trustee :knows ig)
    (trustee :chooses [tg G] :wpp 1)
    (trustee :chooses [h H] :wpp (+ (if (and (> r 0) (> level 0)) (hist [ig tg h] r (- level 1) beta) (is-init h))
                                    1e-3)))
  (investor :observes trustee.h :is h)   ; now the investor believes something about trustee.tg
  (investor :knows fi)
  (return (as investor
              (imagine   ; the Q-value having chosen fi
               (trustee :knows fi)
               (trustee :chooses [ft F] :wpp (if (> level 0) (exp (* beta (trustee [h tg fi ft] r (- level 1) beta))) 1))
               (trustee :chooses [h_ H] :wpp (Tr r h fi ft h_))
               (trustee :chooses [fi_ F] :wpp (if (and (< r 2) (> level 0))
                                                (exp (* beta (investor [h_ ig fi_] (+ r 1) level beta)))
                                                1))
               (E (+ (- (payout-investor fi trustee.ft)
                        (* ig (reludiff (payout-investor fi trustee.ft) (payout-trustee fi trustee.ft))))
                     (if (and (< r 2) (> level 0)) (investor [trustee.h_ ig trustee.fi_] (+ r 1) level beta) 0)))))))

(defmemo ^:cache trustee [[h H] [tg G] [fi F] [ft F]] [r level beta]
  ;; the trustee's Q-function, given h, its own guilt and the investor's last move
  (trustee :knows tg)
  (trustee :thinks
    (investor :knows tg)
    (investor :chooses [ig G] :wpp 1)
    (investor :chooses [h H] :wpp (+ (if (and (> r 0) (> level 0)) (hist [ig tg h] r (- level 1) beta) (is-init h))
                                     1e-3))
    (investor :chooses [fi F] :wpp (if (> level 0) (exp (* beta (investor [h ig fi] r (- level 1) beta))) 1)))
  (trustee :observes investor.h :is h)
  (trustee :observes investor.fi :is fi)
  (trustee :knows ft)
  (return (as trustee
              (imagine   ; the Q-value having chosen ft
               (investor :knows ft)
               (investor :chooses [h_ H] :wpp (Tr r h fi ft h_))
               (investor :chooses [fi_ F] :wpp (if (and (< r 2) (> level 0))
                                                 (exp (* beta (investor [h_ ig fi_] (+ r 1) level beta)))
                                                 1))
               (investor :chooses [ft_ F] :wpp (if (and (< r 2) (> level 0))
                                                 (exp (* beta (trustee [h_ tg fi_ ft_] (+ r 1) level beta)))
                                                 1))
               (E (+ (- (payout-trustee investor.fi ft)
                        (* tg (reludiff (payout-trustee investor.fi ft) (payout-investor investor.fi ft))))
                     (if (and (< r 2) (> level 0))
                       (trustee [investor.h_ tg investor.fi_ investor.ft_] (+ r 1) level beta)
                       0)))))))

(deftest demo-i-pomdp
  (ts/check-demo 'memo.demos.demo-i-pomdp-test "demo-i-pomdp"))
