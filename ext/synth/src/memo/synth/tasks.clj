(ns memo.synth.tasks
  "The task suite: a system prompt (a compact memo.clj reference with three
  worked examples, none of them a task), and nine tasks, each with an exact
  name and signature, calls to grade, and a reference solution whose tables
  are the expected values."
  (:require [memo.synth.check :as check]))

(def reference
  "memo.clj's language, the part of every system prompt after its framing."
  "DOMAINS. A domain is a var: a vector of numbers, (def N [0 1 2 3]), or an enum, (defenum Coin HEADS TAILS) whose members are the integers 0, 1, ... Inside a defmemo write a member as Coin.TAILS; elsewhere (Coin :TAILS). `Bool` is predefined: Bool.NO is 0, Bool.YES is 1.

A MODEL: (defmemo name [[axis Domain] ...] [param (param default) ...] statement ... (return expr)). The result is a table over the axes, in order; several returns stack on a leading axis.

STATEMENTS (agent first, then a keyword):
  (alice :chooses [x X] :wpp e)    x with probability proportional to e. Also :to-maximize e, :to-minimize e (argmax/argmin, ties split evenly), :to-be e, :such-that cond, :uniformly. Aliases: :given :draws :assigned :guesses. (alice :chooses [x X] [y Y] :wpp e) chooses jointly. For softmax write :wpp (exp (* beta u)).
  (alice :thinks stmt ...)         statements in alice's mental model of other agents.
  (alice :observes bob.x :is y)    alice observes bob's x to be the value of the choice y (an axis or another agent's choice, e.g. carol.y). Never a literal: for a fixed value use (alice :observes-that (= bob.x 2)).
  (alice :observes-that cond)      alice conditions on cond.
  (alice :observes-event :wpp e)   soft evidence: weights alice's beliefs by e.
  (alice :knows x bob.y)           alice learns these values (an axis, or another agent's choice).
  (alice :wants g e)               a goal; then (EU g) in alice's later choice is its expected value, once her later statements have happened.

EXPRESSIONS: prefix arithmetic (+ - * / ** % exp log abs), comparisons (= not= < <= > >=), (and or not), (if c a b), numbers, parameters, enum members. Inside an agent's own statements its choices are bare names (x); another agent's choice is agent.x. (as alice e) is e from alice's point of view. (E e), (Pr cond ...), (Var e) are expectations over the current frame's uncertainty. A memo call: (f [x y] args...) evaluates memo f at the choices x, y. Host functions: ordinary Clojure functions of numbers (domain values; enum members as integers), defined with defn before the defmemo, called like (utility a).

RULES memo checks (compile errors otherwise):
  - A choice's weight must be known to the chooser: wrap uncertain values in (E ...) or (Pr ...).
  - A return value must be known to the observer (the top level): wrap uncertain values in (E ...) or (Pr ...). Only agents can observe; the top level never conditions.
  - (as alice e) requires e to be known to alice: (as alice (E bob.x)), not (as alice bob.x).
  - An agent cannot observe a choice it has not modeled with :thinks.

EXAMPLE 1. alice bets on bob's points; she weighs each bet by its expected payoff.
```clojure
(def Points [0 1 2])
(defenum Bet UNDER OVER)
(defn payoff [bet points] (if (== bet 0) (- 2 points) points))
(defmemo bettor [[b Bet]] [beta]
  (alice :thinks (bob :chooses [points Points] :wpp (+ 1 points)))
  (alice :chooses [b Bet] :wpp (exp (* beta (E (payoff b bob.points)))))
  (return (Pr (= alice.b b))))
```

EXAMPLE 2. bob is told that alice's z is y, and infers alice's x.
```clojure
(def X [0 1 2])
(defmemo infer-x [[y X] [x X]] []
  (bob :thinks
    (alice :chooses [x X] :uniformly)
    (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :observes alice.z :is y)
  (bob :knows x)
  (return (as bob (Pr (= alice.x x)))))
```

EXAMPLE 3. carol learns the weather is not dry, and guesses the most likely weather.
```clojure
(defenum Weather DRY RAIN SNOW)
(defmemo guess [[g Weather]] []
  (carol :thinks (sky :chooses [w Weather] :wpp (if (= w Weather.SNOW) 1 2)))
  (carol :observes-that (not= sky.w Weather.DRY))
  (carol :chooses [g Weather] :to-maximize (Pr (= sky.w g)))
  (return (Pr (= carol.g g))))
```")

(def intro
  "You write probabilistic models in memo.clj, a Clojure port of the memo language for recursive reasoning about agents. A model is exact enumeration over finite domains: agents make choices, think about other agents, observe and know things, and a model returns a table over its axes.\n\n")

(def system-prompt
  "One shot: the whole model in one reply."
  (str intro
       "ANSWER FORMAT: one ```clojure code block containing only def, defn, defenum and defmemo forms, in dependency order. No ns or require forms, no printing, no calls. These are already available: defmemo, defenum, and gen.array as `array`. Use exactly the names and signatures the task asks for.\n\n"
       reference))

(def repl-system-prompt
  "The REPL agent: the model built in small steps with tools."
  (str intro
       "HOW YOU WORK: you build the model at a REPL, in small steps, with tools. Your namespace already has defmemo, defenum, defdomain, gen.array as `array`, memo.core as `memo` and memo.domains as `domains`; it keeps every definition between calls.\n"
       "- eval: evaluate Clojure forms (def, defn, defenum, defmemo, or a call to see its value). Define one thing at a time and test it: call a host function on a few values before using it in a defmemo. An error comes back with memo's message and a hint: read the hint, it usually names the fix.\n"
       "- table: call a memo, e.g. (forecast 0.8 1.0 0.5), and see its table with labeled axes. A memo is called with its parameters only: its axes are not arguments, the call returns the whole table over them.\n"
       "- To see what an agent knows at a point of a model, put (alice :inspect) there in the defmemo and eval it: what it prints comes back.\n"
       "- check: checks your memo's name, axes, parameters, that its table is finite and in range, and that its agents reason as the task asks (an agent's belief is read by a query in that agent's frame, not computed outside it), without telling you the answer.\n"
       "- submit: submits the model you have defined; the episode ends. Submit when the table is right.\n"
       "Use exactly the names and signatures the task asks for. Call one tool a turn.\n\n"
       reference))

(def tasks
  [{:id :coin :level 1
    :beliefs '{coin 1}
    :prompt "A coin's bias (its probability of heads) is one of 0.1, 0.3, 0.5, 0.7 and 0.9, each equally likely a priori. Bob flips the coin once and sees heads. Define `(def Bias [0.1 0.3 0.5 0.7 0.9])` and `(defmemo coin [] [] ...)` returning Bob's expected value of the bias after seeing heads."
    :calls [{:memo 'coin :args []}]
    :reference "(def Bias [0.1 0.3 0.5 0.7 0.9])
(defmemo coin [] []
  (bob :thinks (c :chooses [bias Bias] :uniformly))
  (bob :observes-event :wpp c.bias)
  (return (as bob (E c.bias))))"}

   {:id :act :level 1
    :prompt "An agent chooses one of three actions, `(defenum Action WAIT JUMP RUN)`, whose utilities are 0, 1 and 3, with probability proportional to exp(beta * utility). Define `(defmemo act [[a Action]] [beta] ...)` returning the probability that the agent chooses action a."
    :calls [{:memo 'act :args [1.5]} {:memo 'act :args [0.5]}]
    :reference "(defenum Action WAIT JUMP RUN)
(defn utility [a] (nth [0.0 1.0 3.0] a))
(defmemo act [[a Action]] [beta]
  (bob :chooses [a Action] :wpp (exp (* beta (utility a))))
  (return (Pr (= bob.a a))))"}

   {:id :monty :level 2
    :beliefs '{monty 1}
    :prompt "Monty Hall. `(def Door [0 1 2])`. A prize is behind one of the three doors, each equally likely. The contestant picks door 0. Monty knows where the prize is and opens a door that is neither the contestant's door nor the prize's, uniformly among the doors allowed. The contestant sees Monty open door 2. Define `(defmemo monty [[d Door]] [] ...)` returning the contestant's posterior probability that the prize is behind door d."
    :calls [{:memo 'monty :args []}]
    :reference "(def Door [0 1 2])
(defmemo monty [[d Door]] []
  (alice :thinks
    (host :chooses [prize Door] :uniformly)
    (host :chooses [opens Door] :wpp (and (not= opens prize) (not= opens 0))))
  (alice :observes-that (= host.opens 2))
  (alice :knows d)
  (return (as alice (Pr (= host.prize d)))))"}

   {:id :taste :level 2
    :beliefs '{taste 1}
    :prompt "Inverse planning. `(defenum Taste SWEET SOUR)` and `(defenum Fruit APPLE BANANA CHERRY)`. A child has a taste, SWEET or SOUR, equally likely a priori. The utility of each fruit: for SWEET, APPLE 1, BANANA 2, CHERRY 0; for SOUR, APPLE 2, BANANA 0, CHERRY 1. The child picks a fruit with probability proportional to exp(beta * utility). An observer, who knows all this, sees which fruit the child picks. Define `(defmemo taste [[f Fruit] [t Taste]] [beta] ...)` returning the observer's posterior probability that the child's taste is t, given that the child picked f."
    :calls [{:memo 'taste :args [1.0]} {:memo 'taste :args [2.0]}]
    :reference "(defenum Taste SWEET SOUR)
(defenum Fruit APPLE BANANA CHERRY)
(defn utility [t f] (nth (nth [[1.0 2.0 0.0] [2.0 0.0 1.0]] t) f))
(defmemo taste [[f Fruit] [t Taste]] [beta]
  (obs :thinks
    (kid :chooses [tt Taste] :uniformly)
    (kid :chooses [fruit Fruit] :wpp (exp (* beta (utility tt fruit)))))
  (obs :observes kid.fruit :is f)
  (obs :knows t)
  (return (as obs (Pr (= kid.tt t)))))"}

   {:id :sally :level 2
    :beliefs '{sally 1}
    :prompt "False belief. `(defenum Loc BASKET BOX)`. Sally puts a marble in the BASKET and leaves. While she is away, Anne moves it to the BOX with probability p (otherwise it stays). Sally does not see what Anne does, but knows that Anne moves it with probability p. When Sally returns she looks in the location she believes more likely to hold the marble (if both are equally likely, either one with probability 1/2). Define `(defmemo sally [[l Loc]] [p] ...)` returning the probability that Sally looks in l, and `(defmemo marble [[l Loc]] [p] ...)` returning the probability that the marble is in l."
    :calls [{:memo 'sally :args [0.3]} {:memo 'sally :args [0.8]} {:memo 'sally :args [0.5]}
            {:memo 'marble :args [0.3]} {:memo 'marble :args [0.8]}]
    :reference "(defenum Loc BASKET BOX)
(defmemo marble [[l Loc]] [p]
  (anne :chooses [moved Bool] :wpp (if (= moved Bool.YES) p (- 1 p)))
  (return (Pr (= (if anne.moved Loc.BOX Loc.BASKET) l))))
(defmemo sally [[l Loc]] [p]
  (sally :thinks (anne :chooses [moved Bool] :wpp (if (= moved Bool.YES) p (- 1 p))))
  (sally :chooses [look Loc] :to-maximize (Pr (= (if anne.moved Loc.BOX Loc.BASKET) look)))
  (return (Pr (= sally.look l))))"}

   {:id :compete :level 3
    :beliefs '{compete 1}
    :prompt "Planning with an uncertain outcome. `(def A [0 1 2])` and `(def N [0 1 2 3])`. Bob's score is uniform over N. Alice picks a number a in A; after she picks, her own score s in N is drawn with probability proportional to exp(-(s - a - 1)^2). Alice wins if her score is strictly greater than Bob's. She picks the a that maximizes her probability of winning (ties split evenly). Define `(defmemo compete [[a A]] [] ...)` returning the probability that Alice picks a."
    :calls [{:memo 'compete :args []}]
    :reference "(def A [0 1 2])
(def N [0 1 2 3])
(defmemo compete [[a A]] []
  (alice :thinks (bob :chooses [score N] :uniformly))
  (alice :wants win (> score bob.score))
  (alice :chooses [a A] :to-maximize (EU win))
  (alice :given [score N] :wpp (exp (- (** (- score a 1) 2))))
  (return (Pr (= alice.a a))))"}

   {:id :implicature :level 3
    :beliefs '{listener 1}
    :prompt "Scalar implicature (the Rational Speech Acts model). `(defenum Utt NONE SOME ALL)` and `(def Count [0 1 2 3])`, the number of apples eaten, uniform a priori. NONE is true of 0; SOME is true of 1, 2 and 3; ALL is true of 3. A literal listener L0(n | u) is proportional to prior(n) if u is true of n, else 0. A speaker who knows n says u with probability S1(u | n) proportional to L0(n | u)^alpha. A pragmatic listener L1(n | u) is proportional to prior(n) * S1(u | n). Define `(defmemo listener [[u Utt] [n Count]] [alpha] ...)` returning L1(n | u)."
    :calls [{:memo 'listener :args [1.0]} {:memo 'listener :args [3.0]}]
    :reference "(defenum Utt NONE SOME ALL)
(def Count [0 1 2 3])
(defn true-of [u n] (cond (== u 0) (== n 0) (== u 1) (>= n 1) :else (== n 3)))
(defmemo l0 [[u Utt] [n Count]] []
  (lit :knows u)
  (lit :chooses [n Count] :wpp (true-of u n))
  (return (Pr (= lit.n n))))
(defmemo listener [[u Utt] [n Count]] [alpha]
  (l1 :thinks
    (speaker :given [n Count] :uniformly)
    (speaker :chooses [u Utt] :wpp (** (l0 [u n]) alpha)))
  (l1 :observes speaker.u :is u)
  (l1 :chooses [n Count] :wpp (Pr (= speaker.n n)))
  (return (Pr (= l1.n n))))"}

   {:id :alarm :level 3
    :beliefs '{alarm 1}
    :prompt "A panic model. `(def Suds (vec (range 11)))` is fear from 0 to 10, `(defenum Sensation STILL RACING POUNDING)` and `(defenum Heart SOUND WEAK)`. Sam's prior probability that his heart is WEAK is p-weak. A WEAK heart feels STILL, RACING, POUNDING with probabilities 0.02, 0.13, 0.85 whatever his fear. A SOUND heart at fear level f feels a mixture: with x = insight * f / 10, the probabilities are (1 - x) * [0.80 0.15 0.05] + x * [0.05 0.25 0.70]. Sam knows his fear level and feels his heart. Define `(defmemo alarm [[felt Sensation] [fear Suds]] [p-weak insight] ...)` returning Sam's posterior probability that his heart is WEAK when it feels like felt at fear level fear."
    :calls [{:memo 'alarm :args [0.2 0.0]} {:memo 'alarm :args [0.01 1.0]} {:memo 'alarm :args [0.2 1.0]}]
    :reference "(def Suds (vec (range 11)))
(defenum Sensation STILL RACING POUNDING)
(defenum Heart SOUND WEAK)
(defn feels [s heart fear insight]
  (let [x (* insight (/ fear 10.0))]
    (if (== heart 1)
      (nth [0.02 0.13 0.85] s)
      (+ (* (- 1 x) (nth [0.80 0.15 0.05] s)) (* x (nth [0.05 0.25 0.70] s))))))
(defmemo alarm [[felt Sensation] [fear Suds]] [p-weak insight]
  (sam :knows fear)
  (sam :thinks
    (heart :given [state Heart] :wpp (if (= state Heart.WEAK) p-weak (- 1 p-weak)))
    (heart :knows fear)
    (heart :draws [s Sensation] :wpp (feels s state fear insight)))
  (sam :observes heart.s :is felt)
  (return (as sam (Pr (= heart.state Heart.WEAK)))))"}

   {:id :forecast :level 4
    :beliefs '{forecast 2}
    :prompt "A second-order belief. `(defenum Act QUIET SPEAK)`, `(defenum Skill LOW HIGH)`, `(defenum Sound SILENT STEADY SHAKY)`. At a meeting Alice either stays QUIET or SPEAKs, and Bob listens. If she is quiet she sounds SILENT; if she speaks she sounds SHAKY with probability leak and STEADY otherwise. Bob judges her skill from how she sounds: his prior is P(HIGH) = p0, and in his model a person's sound has weights SILENT 1, STEADY 1 - s, SHAKY s (normalized for each skill), where s = 0.25 - 0.2 * harsh if the skill is HIGH and s = 0.4 + 0.2 * harsh if it is LOW. Alice knows this whole story but not how she will sound. Define `(defmemo forecast [[a Act]] [leak harsh p0] ...)` returning, for each act a, Alice's expectation of Bob's posterior probability that her skill is HIGH after he hears her."
    :calls [{:memo 'forecast :args [0.8 1.0 0.5]} {:memo 'forecast :args [0.2 0.0 0.7]}]
    :reference "(defenum Act QUIET SPEAK)
(defenum Skill LOW HIGH)
(defenum Sound SILENT STEADY SHAKY)
(defn sounds [s a leak] (if (== a 0) (if (== s 0) 1.0 0.0) (nth [0.0 (- 1.0 leak) leak] s)))
(defn bob-hears [s skill harsh]
  (let [sh (if (== skill 1) (- 0.25 (* 0.2 harsh)) (+ 0.4 (* 0.2 harsh)))]
    (nth [1.0 (- 1.0 sh) sh] s)))
(defmemo forecast [[a Act]] [leak harsh p0]
  (alice :knows a)
  (alice :thinks
    (voice :knows a)
    (voice :chooses [s Sound] :wpp (sounds s a leak))
    (bob :thinks
      (alice :given [skill Skill] :wpp (if (= skill Skill.HIGH) p0 (- 1 p0)))
      (alice :given [s Sound] :wpp (bob-hears s skill harsh)))
    (bob :observes alice.s :is voice.s))
  (return (as alice (E (as bob (Pr (= alice.skill Skill.HIGH)))))))"}])

(def heldout
  "Tasks kept out of the nine (memoclj-pcw0): no prompt, hint or check was
  written from them. Run by naming them (synth.tasks)."
  [{:id :depression :level 4 :heldout true
    :beliefs '{engage 1 therapist 2}
    :prompt "A model of hopelessness in depression (learned helplessness). `(def Control [0.1 0.3 0.5 0.7 0.9])` is how likely engaging in an activity is to turn out well, `(def Hope [0 1 2 4])` are levels of hopelessness, and `(defenum Act REST ENGAGE)`. Sam does not know the control c: his prior weight on each c is (1 - c)^h, where h is his hopelessness (h = 0: no bias). If Sam engages, the activity turns out well with probability c; if he rests, the day turns out well with probability 0.2, whatever c is. A good outcome is worth 1, a bad one 0, and engaging costs `effort`. Sam picks an act with probability proportional to exp(beta * his expected value of the act). Define `(defmemo engage [[a Act]] [h effort beta] ...)` returning the probability that Sam picks a, for a given hopelessness h. A therapist does not know Sam's hopelessness: in her model it is one of Hope, each equally likely, and Sam reasons and chooses as above. She sees which act Sam picks. Define `(defmemo therapist [[a Act] [g Hope]] [effort beta] ...)` returning the therapist's posterior probability that Sam's hopelessness is g, given that Sam picked a."
    :calls [{:memo 'engage :args [0 0.2 3.0]} {:memo 'engage :args [2 0.2 3.0]} {:memo 'engage :args [4 0.5 2.0]}
            {:memo 'therapist :args [0.2 3.0]} {:memo 'therapist :args [0.4 5.0]}]
    :reference "(def Control [0.1 0.3 0.5 0.7 0.9])
(def Hope [0 1 2 4])
(defenum Act REST ENGAGE)
(defmemo engage [[a Act]] [h effort beta]
  (sam :thinks (world :chooses [c Control] :wpp (** (- 1 c) h)))
  (sam :chooses [a Act] :wpp (exp (* beta (E (if (= a Act.ENGAGE) (- world.c effort) 0.2)))))
  (return (Pr (= sam.a a))))
(defmemo therapist [[a Act] [g Hope]] [effort beta]
  (doc :thinks
    (sam :chooses [h Hope] :uniformly)
    (sam :thinks (world :knows h)
                 (world :chooses [c Control] :wpp (** (- 1 c) h)))
    (sam :chooses [act Act] :wpp (exp (* beta (E (if (= act Act.ENGAGE) (- world.c effort) 0.2))))))
  (doc :observes sam.act :is a)
  (doc :knows g)
  (return (as doc (Pr (= sam.h g)))))"}

   {:id :rumination :level 5 :heldout true
    :beliefs '{ruminate 1} :structure '{ruminate {:self-call true}}
    :prompt "Rumination in depression, with asymmetric updating. `(def Worth [0 1 2 3])` is Sam's self-worth, and `(defenum Reading NEUTRAL CRITICAL)` is how he reads a friend's ambiguous remark: if his worth is v, the remark reads as CRITICAL with probability (4 - v)/5, and as NEUTRAL otherwise. Sam ruminates: he replays the remark again and again. Before the first replay, his belief about his worth is uniform. In each replay he reads the remark afresh: the reading is CRITICAL with the probability his current belief gives it. He then updates his belief on that reading, but asymmetrically: after a CRITICAL reading by Bayes' rule, and after a NEUTRAL reading with the likelihood raised to the power a (0 <= a <= 1; a = 1 is Bayes' rule, a = 0 ignores good news). His belief after a replay is the average of the two updated beliefs, weighted by the probabilities of the two readings, and it is his prior for the next replay. Define `(defmemo ruminate [[w Worth]] [k a] ...)` returning Sam's belief that his worth is w after k replays (k >= 1)."
    :calls [{:memo 'ruminate :args [1 1.0]} {:memo 'ruminate :args [1 0.3]}
            {:memo 'ruminate :args [3 0.3]} {:memo 'ruminate :args [5 0.0]}]
    :reference "(def Worth [0 1 2 3])
(defenum Reading NEUTRAL CRITICAL)
(defn p-critical [v] (/ (- 4.0 v) 5.0))
(defn p-reading [x v] (if (== x 1) (p-critical v) (- 1.0 (p-critical v))))
(defn evidence [x v a] (if (== x 1) (p-critical v) (Math/pow (- 1.0 (p-critical v)) a)))
(defmemo ruminate [[w Worth]] [k a]
  (sam :thinks (me :chooses [v Worth] :wpp (if (= k 1) 1 (ruminate [v] (- k 1) a))))
  (sam :chooses [x Reading] :wpp (E (p-reading x me.v)))
  (sam :observes-event :wpp (evidence x me.v a))
  (sam :knows w)
  (return (E (as sam (Pr (= me.v w))))))"}
   {:id :meet :level 5 :heldout true
    :beliefs '{meet 1} :structure '{meet {:self-call true}}
    :prompt "Two friends want to meet at one of three cafes, `(def Place [0 1 2])`, without talking. The cafes' salience is 3, 2 and 1. A level-0 friend picks a cafe with probability proportional to its salience. A level-k friend (k >= 1) imagines the other friend as a level-(k - 1) friend, and picks cafe x with probability proportional to exp(beta * the probability that the other friend picks x). Define `(defmemo meet [[x Place]] [k beta] ...)` returning the probability that a level-k friend picks x (k >= 0)."
    :calls [{:memo 'meet :args [0 2.0]} {:memo 'meet :args [1 2.0]} {:memo 'meet :args [3 2.0]} {:memo 'meet :args [4 5.0]}]
    :reference "(def Place [0 1 2])
(defn salience [y] (nth [3.0 2.0 1.0] y))
(defmemo meet [[x Place]] [k beta]
  (me :thinks (other :chooses [y Place] :wpp (if (<= k 1) (salience y) (meet [y] (- k 1) beta))))
  (me :chooses [x Place] :wpp (if (= k 0) (salience x) (exp (* beta (Pr (= other.y x))))))
  (return (Pr (= me.x x))))"}

   {:id :panic-spiral :level 5 :heldout true
    :beliefs '{panic 1} :structure '{panic {:self-call true}}
    :prompt "Panic as a vicious circle (Clark 1986). `(def Fear [0 1 2 3])` is Sam's fear level, `(defenum Sensation CALM POUNDING)` is what he feels, and `(defenum Heart SOUND WEAK)`. In each cycle, Sam's heart pounds with probability (1 + g)/5, where g is his fear at the start of the cycle, and feels CALM otherwise. Sam does not see that his fear makes his heart pound. In his model his heart is WEAK with prior probability p-weak, a WEAK heart pounds with probability 0.8, and a SOUND one with probability 0.3. He feels his heart and infers from what he feels the probability w that his heart is WEAK; his fear for the next cycle is then f with probability proportional to exp(-2 (f - 3w)^2). Before the first cycle his fear is 0. Define `(defmemo panic [[f Fear]] [k p-weak] ...)` returning the probability that his fear is f after k cycles (k >= 1)."
    :calls [{:memo 'panic :args [1 0.3]} {:memo 'panic :args [2 0.3]} {:memo 'panic :args [4 0.3]} {:memo 'panic :args [4 0.05]}]
    :reference "(def Fear [0 1 2 3])
(defenum Sensation CALM POUNDING)
(defenum Heart SOUND WEAK)
(defn pound [g s] (let [p (/ (+ 1.0 g) 5.0)] (if (== s 1) p (- 1.0 p))))
(defn feel [h s] (let [p (if (== h 1) 0.8 0.3)] (if (== s 1) p (- 1.0 p))))
(defmemo panic [[f Fear]] [k p-weak]
  (body :chooses [g Fear] :wpp (if (<= k 1) (= g 0) (panic [g] (- k 1) p-weak)))
  (body :chooses [s Sensation] :wpp (pound g s))
  (sam :thinks (heart :chooses [h Heart] :wpp (if (= h Heart.WEAK) p-weak (- 1 p-weak)))
               (heart :chooses [s Sensation] :wpp (feel h s)))
  (sam :observes heart.s :is body.s)
  (sam :chooses [f Fear] :wpp (exp (* -2 (** (- f (* 3 (Pr (= heart.h Heart.WEAK)))) 2))))
  (return (Pr (= sam.f f))))"}

   {:id :guess :level 5 :heldout true
    :beliefs '{guess 3} :structure '{guess {:thinks-depth 3}}
    :prompt "A guessing game four deep. `(def N [0 1 2 3 4])`. Ann picks a number with probability proportional to 1, 2, 4, 2, 1 (she likes the middle). Bob guesses Ann's number: he picks y with probability proportional to exp(beta * the probability that Ann picked y). Cleo, who knows all this, guesses Bob's guess the same way, and Dan guesses Cleo's guess the same way. Define `(defmemo guess [[z N]] [beta] ...)` returning the probability that Dan's guess is z."
    :calls [{:memo 'guess :args [1.0]} {:memo 'guess :args [4.0]}]
    :reference "(def N [0 1 2 3 4])
(defn taste [x] (nth [1.0 2.0 4.0 2.0 1.0] x))
(defmemo guess [[z N]] [beta]
  (dan :thinks
    (cleo :thinks
      (bob :thinks (ann :chooses [x N] :wpp (taste x)))
      (bob :chooses [y N] :wpp (exp (* beta (Pr (= ann.x y))))))
    (cleo :chooses [w N] :wpp (exp (* beta (Pr (= bob.y w))))))
  (dan :chooses [z N] :wpp (exp (* beta (Pr (= cleo.w z)))))
  (return (Pr (= dan.z z))))"}

   {:id :interview :level 5 :heldout true
    :beliefs '{interview 1}
    :prompt "Planning over one's own later choice. `(defenum Act ATTEND SKIP)`. Today Sam decides whether to prepare for tomorrow's job interview (`Bool`: YES or NO); preparing costs c. Tomorrow he decides whether to ATTEND or SKIP. If he attends, he succeeds with probability 0.8 if he prepared and 0.4 if not; success is worth 1, failure 0. Skipping is worth 0.3 for sure. Tomorrow he picks an act with probability proportional to exp(beta * its expected value, given whether he prepared). Today he picks whether to prepare with probability proportional to exp(beta * (the expected value of tomorrow, over his own choice tomorrow and its outcome, minus c if he prepares)). Define `(defmemo interview [[p Bool]] [c beta] ...)` returning the probability that Sam's choice today is p."
    :calls [{:memo 'interview :args [0.1 3.0]} {:memo 'interview :args [0.2 10.0]} {:memo 'interview :args [0.05 1.0]}]
    :reference "(defenum Act ATTEND SKIP)
(defn p-ok [prep] (if (== prep 1) 0.8 0.4))
(defmemo interview [[p Bool]] [c beta]
  (sam :wants value (if (= act Act.ATTEND) (= ok Bool.YES) 0.3))
  (sam :chooses [prep Bool] :wpp (exp (* beta (- (EU value) (* c prep)))))
  (sam :chooses [act Act] :wpp (exp (* beta (if (= act Act.ATTEND) (p-ok prep) 0.3))))
  (sam :given [ok Bool] :wpp (if (= act Act.ATTEND) (if (= ok Bool.YES) (p-ok prep) (- 1 (p-ok prep))) (if (= ok Bool.YES) 0 1)))
  (return (Pr (= sam.prep p))))"}

   {:id :smile :level 4 :heldout true
    :beliefs '{smile 2}
    :prompt "Social anxiety and discounted praise (Clark and Wells 1995). `(defenum Skill LOW HIGH)`, `(defenum Sound STEADY SHAKY)` and `(defenum Face BLANK SMILE)`. Alice gives a talk and Bob listens. In Alice's mind: her voice sounds SHAKY with probability leak, and STEADY otherwise. Bob, as she imagines him, has a prior P(HIGH) = p0 about her skill, and in his model a speaker of HIGH skill sounds SHAKY with probability 0.25 - 0.2 * harsh, one of LOW skill with probability 0.4 + 0.2 * harsh. Bob hears her voice and forms his posterior P that her skill is HIGH. Then he smiles (SMILE) with probability (1 - d) * P + d * 0.9, where d is how much Alice thinks his smiles are mere politeness, and keeps a BLANK face otherwise. Alice sees Bob's face. Define `(defmemo smile [[face Face]] [leak harsh p0 d] ...)` returning Alice's expectation of Bob's posterior that her skill is HIGH, after she sees face."
    :calls [{:memo 'smile :args [0.8 1.0 0.5 0.0]} {:memo 'smile :args [0.8 1.0 0.5 0.9]} {:memo 'smile :args [0.2 0.0 0.7 0.5]}]
    :reference "(defenum Skill LOW HIGH)
(defenum Sound STEADY SHAKY)
(defenum Face BLANK SMILE)
(defn shaky [skill harsh] (if (== skill 1) (- 0.25 (* 0.2 harsh)) (+ 0.4 (* 0.2 harsh))))
(defmemo smile [[face Face]] [leak harsh p0 d]
  (alice :knows face)
  (alice :thinks
    (voice :chooses [s Sound] :wpp (if (= s Sound.SHAKY) leak (- 1 leak)))
    (bob :thinks
      (alice :given [skill Skill] :wpp (if (= skill Skill.HIGH) p0 (- 1 p0)))
      (alice :given [s Sound] :wpp (if (= s Sound.SHAKY) (shaky skill harsh) (- 1 (shaky skill harsh)))))
    (bob :observes alice.s :is voice.s)
    (bob :chooses [f Face] :wpp (if (= f Face.SMILE)
                                  (+ (* (- 1 d) (Pr (= alice.skill Skill.HIGH))) (* d 0.9))
                                  (- 1 (+ (* (- 1 d) (Pr (= alice.skill Skill.HIGH))) (* d 0.9))))))
  (alice :observes bob.f :is face)
  (return (as alice (E (as bob (Pr (= alice.skill Skill.HIGH)))))))"}

   {:id :food-truck :level 4 :heldout true
    :beliefs '{food-truck 2}
    :prompt "Inferring what someone wants and what they believe (Baker et al. 2017). `(defenum Truck KOREAN MEXICAN)`, `(defenum Way NEAR FAR)` and `(def Bel [0.2 0.5 0.8])`. Two food trucks are parked, one at a near spot and one at a far spot. A student prefers one of the trucks, and believes the KOREAN truck is at the near spot with probability b (the MEXICAN one is then at the far spot). She walks NEAR or FAR with probability proportional to exp(beta * (the probability, by her belief, that her preferred truck is at that spot, minus 0.2 if she walks FAR)). An observer knows all this but neither her preference nor her belief b: in his model each truck is equally likely to be her preference, and b is each of Bel with equal probability. He sees which way she walks. Define `(defmemo food-truck [[way Way] [pref Truck] [b Bel]] [beta] ...)` returning the observer's posterior probability that she prefers pref and believes b, given that she walked way."
    :calls [{:memo 'food-truck :args [2.0]} {:memo 'food-truck :args [5.0]}]
    :reference "(defenum Truck KOREAN MEXICAN)
(defenum Way NEAR FAR)
(def Bel [0.2 0.5 0.8])
(defmemo food-truck [[way Way] [pref Truck] [b Bel]] [beta]
  (obs :thinks
    (st :chooses [pref Truck] :uniformly)
    (st :chooses [b Bel] :uniformly)
    (st :thinks (lot :knows b)
                (lot :chooses [kn Bool] :wpp (if (= kn Bool.YES) b (- 1 b))))
    (st :chooses [w Way] :wpp (exp (* beta (- (E (if (= w Way.NEAR)
                                                   (= (if lot.kn Truck.KOREAN Truck.MEXICAN) pref)
                                                   (= (if lot.kn Truck.MEXICAN Truck.KOREAN) pref)))
                                              (if (= w Way.FAR) 0.2 0))))))
  (obs :observes st.w :is way)
  (obs :knows pref b)
  (return (as obs (Pr (and (= st.pref pref) (= st.b b))))))"}

   {:id :effort :level 3 :heldout true
    :beliefs '{effort 1}
    :prompt "Inferring how much a reward is worth from the effort spent. `(def Reward [1 2 3])` and `(defenum Move CLIMB STAY)`. A child wants a toy whose worth v to her is one of Reward, each equally likely to an observer. She faces two hills, of costs 1.5 and 2.5. For each hill, independently, she climbs with probability proportional to exp(beta * (v - cost)) and stays with probability proportional to 1. The observer sees her climb the first hill and stay at the second. Define `(defmemo effort [[r Reward]] [beta] ...)` returning the observer's posterior probability that the toy's worth to her is r."
    :calls [{:memo 'effort :args [1.0]} {:memo 'effort :args [3.0]}]
    :reference "(def Reward [1 2 3])
(defenum Move CLIMB STAY)
(defmemo effort [[r Reward]] [beta]
  (obs :thinks
    (kid :chooses [v Reward] :uniformly)
    (kid :chooses [a Move] :wpp (exp (* beta (if (= a Move.CLIMB) (- v 1.5) 0))))
    (kid :chooses [b Move] :wpp (exp (* beta (if (= b Move.CLIMB) (- v 2.5) 0)))))
  (obs :observes-that (and (= kid.a Move.CLIMB) (= kid.b Move.STAY)))
  (obs :knows r)
  (return (as obs (Pr (= kid.v r)))))"}

   {:id :persuasion :level 3 :heldout true
    :prompt "A seller who wants to be thought well of. `(def Q [0 1 2 3])` is a product's quality, and `(defenum Says NOTHING SOLID EXCELLENT)` is what the seller says about it. NOTHING is true of every quality, SOLID of quality 1 and above, and EXCELLENT only of quality 3. A literal buyer who hears an utterance believes each quality it is true of equally likely. The seller knows the quality, says only true things, and picks an utterance with probability proportional to exp(beta * the literal buyer's expected quality after hearing it). Define `(defmemo persuade [[q Q] [u Says]] [beta] ...)` returning the probability that the seller says u when the quality is q."
    :calls [{:memo 'persuade :args [1.0]} {:memo 'persuade :args [3.0]}]
    :reference "(def Q [0 1 2 3])
(defenum Says NOTHING SOLID EXCELLENT)
(defn holds? [u q] (cond (== u 0) true (== u 1) (>= q 1) :else (>= q 3)))
(defmemo heard [[u Says]] []
  (lis :knows u)
  (lis :chooses [q Q] :wpp (holds? u q))
  (return (E lis.q)))
(defmemo persuade [[q Q] [u Says]] [beta]
  (seller :knows q)
  (seller :chooses [u Says] :wpp (if (holds? u q) (exp (* beta (heard [u]))) 0))
  (return (Pr (= seller.u u))))"}])

(def all-tasks "The nine and the held-out ones." (into tasks heldout))

(defn with-expected
  "The tasks, each call with :expected from the reference solution, and
  :signatures, each memo the calls name as the reference defines it
  ({:params :axes :shape}) with the reasoning the task asks of it ({:beliefs
  n}, check/shape's spec: :beliefs, the agents that must read their beliefs,
  0 unless the task names it). An error if a reference fails, or departs
  from its own spec: the suite is wrong, not the model."
  [tasks]
  (mapv (fn [{:keys [id reference calls beliefs structure] :as t}]
          (let [n (check/fresh-ns)
                forms (check/read-forms reference n)
                ev (check/evaluate n forms)]
            (when-not (:ok ev) (throw (ex-info (str "reference of " id " fails") ev)))
            (assoc t
                   :calls (mapv (fn [c]
                                  (let [r (check/call-memo n (:memo c) (:args c) 120000)]
                                    (when (:kind r) (throw (ex-info (str "reference call of " id " fails") r)))
                                    (assoc c :expected (:value r))))
                                calls)
                   :signatures (into {} (for [s (distinct (map :memo calls))
                                              :let [m @(ns-resolve n s)
                                                    spec {:beliefs (get beliefs s 0)}
                                                    problems (check/shape s m spec)]]
                                          (do (when (seq problems)
                                                (throw (ex-info (str "reference of " id " fails its shape check")
                                                                {:memo s :problems problems})))
                                              (when-let [st (get structure s)]
                                                (when (not= :as-intended (check/structure reference s st))
                                                  (throw (ex-info (str "reference of " id " fails its structure check")
                                                                  {:memo s :structure st}))))
                                              [s (merge {:params (:memo/params (meta m)) :axes (:memo/axes (meta m))
                                                         :shape (:memo/shape (meta m))}
                                                        spec
                                                        (when-let [st (get structure s)] {:structure st}))]))))))
        tasks))
