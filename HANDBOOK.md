# The memo.clj Handbook

memo.clj is [memo](https://github.com/kach/memo) on the JVM, built on
gen.clj. memo is a language for recursive reasoning about agents: agents
make choices, think about each other's choices, observe and update. A
model compiles to array programs and gives exact answers.

This Handbook follows the memo Handbook (memo 1.3.3) part by part and
heading by heading, so the two can be read side by side. Its examples are
in memo.clj's syntax. A closing part lists the differences from Python
memo.

**This Handbook is executable.** `test/memo/handbook_doc_test.clj` runs
every `clojure` block below, in order, in one namespace, on every backend.
A line `;=> value` after a form is checked: the form's result, read back as
Clojure data, must equal the value (numbers within float32 tolerance). A
block whose first line is `;; error: message` must throw a memo error whose
message begins with that text.

*Adapted from "The memo Handbook" by Kartik Chandra and Tony Chen, MIT
License, Copyright (c) 2024-2025 Kartik Chandra and Tony Chen.*

## How This Guide Is Organized

1. **[Anatomy of a memo](#anatomy-of-a-memo)**: the structure of a model,
   its axes, parameters, agents and return values, and its domains.
2. **[Statements](#statements)**: what agents do. They choose
   (`:chooses`), model other agents (`:thinks`), observe (`:observes`),
   and more.
3. **[Expressions](#expressions)**: computing values. Literals, operators,
   host functions, probabilistic queries (`E`, `Pr`, `Var`),
   information-theoretic measures (`H`, `KL`) and hypotheticals
   (`imagine`).
4. **[Running and Configuring memos](#running-and-configuring-memos)**:
   calling a model, reading its results, options, gradients and errors.
5. **[Minds in gen.clj models](#minds-in-genclj-models)**: a memo as a
   gen.clj distribution or deterministic generative function
   (`memo.gen`).
6. **[Bounded minds](#bounded-minds)** (experimental): an agent that
   reasons from a few samples, with a budget given at the call
   (`memo.bounded`).
7. **[Stability](#stability)**: what 1.0 freezes, and what may change.
8. **[Differences from Python memo](#differences-from-python-memo)**.

## Setting Up

memo.clj is a Clojure library for the Clojure CLI and a JDK (gen.clj is
developed on Java 25; its MLX backend uses the foreign function interface,
final since Java 22). Depend on it from a `deps.edn`:

```text
{:deps {memo.clj/memo.clj {:local/root "/path/to/memo.clj"}}}
```

memo.clj pins its gen.clj. After the first download, run
`clojure -X:deps prep` once: gen.clj compiles its Java binding. Then load a
file of models with `clojure -M -i models.clj`, or work at a REPL (`clj`).
Models run on the JVM backend by default. MLX (Metal, or MLX's CPU) needs
gen.clj's native libraries, built by its `native/build-mlx.sh` (gen.clj's
README), and the system properties of memo.clj's `:mlx` alias:
`-Dgen.backend=mlx`, `-Dgen.mlx.lib=` the libraries' directory, and
`--enable-native-access=ALL-UNNAMED`. Inside memo.clj,
`clojure -M:mlx -i models.clj` runs a file on MLX, and `clojure -X:test`
runs the tests (CLAUDE.md lists the rest).

Every example below runs in this namespace:

```clojure
(ns handbook.models
  (:require [gen.array :as array]
            [memo.core :as memo :refer [defmemo]]
            [memo.domains :as domains :refer [defenum defdomain]]
            [memo.errors :as err]))
```

`memo.core` has `defmemo`, `print-table`, `program` and `memo?`.
`memo.domains` has `defenum`, `defdomain`, the predefined `Bool` and
functions on domains. `memo.errors` recognizes memo errors (see
[Errors](#errors)), and `memo.gen` puts minds in gen.clj models (see
[Minds in gen.clj models](#minds-in-genclj-models)). A model's results are
gen.array arrays, and `gen.array/->clj` reads them as Clojure numbers and
vectors. What 1.0 freezes is listed under [Stability](#stability).

---

## Anatomy of a memo

A memo model is a `defmemo` form:

```clojure
(def X [0 1 2])
(def Y [0 1 2])

(defmemo model [[x X] [y Y]] [a b (c 3.14)]
  "This is a model of two agents."
  (alice :chooses [x X] :wpp (exp (* a x)))
  (bob :chooses [y Y] :wpp (exp (* b y c)))
  (return (Pr (= alice.x x) (= bob.y y))))

(array/shape (model 0.5 -0.3))
;=> [3 3]
(array/->clj (array/sum (model 0.5 -0.3)))
;=> 1.0
```

| Component | Description |
|-----------|-------------|
| `model` | The name of the model |
| `[[x X] [y Y]]` | The axes of the array to compute: each axis variable ranges over a domain |
| `[a b (c 3.14)]` | Scalar parameters, with optional default values |
| `"..."` | An optional docstring |
| `(alice ...)` / `(bob ...)` | A sequence of **statements** that define what agents do |
| `(return ...)` | An **expression** computed for each cell of the returned array |

The result has one cell per combination of axis values, in the order the
axes are declared. The docstring is kept:

```clojure
(:memo/doc (meta model))
;=> "This is a model of two agents."
```

**Domains** are the finite sets that axes and choices range over. A
domain is a vector of numbers, an enum made by `defenum` (Python's
`IntEnum`), or a product made by `defdomain` (Python memo's
`domain(...)`). A choice's value is always a domain value, never an index.
An enum's members are numbered from 0, or given values:

```clojure
(defenum Coin HEADS TAILS)          ; HEADS is 0, TAILS is 1
(defenum Color [GREEN 1] [PINK 2])  ; explicit values

(Coin :TAILS)
;=> 1
(Color :PINK)
;=> 2
```

A number is not a domain: Python memo takes `S = 3` as the domain {3},
memo.clj asks for `[3]`. Nor can a parameter have a domain's name, which
Python memo would read as the domain. Both are memo errors.

Inside a `defmemo`, a member is written `Coin.TAILS`. Outside, an enum is a
function from a member's name to its value, as above. As in an `IntEnum`,
a member given the value of an earlier one is its alias:
`(defenum D [P 1] [Q 1] [R 2])` has the members `P` and `R`, and `D.Q` is
`P`. An axis or a choice
names its domain by a var: `[x X]`, never `[x [0 1 2]]`. A gen.array
array is not a domain: write `(vec (array/->clj a))`. A member written
without a value after one with a value is the previous value plus one, as
Python's `auto()` makes it: `(defenum C [RED 1] GREEN)` has GREEN 2.

A vector domain may repeat a value, as a Python list may, but a choice's
value then names more than one member, and memo compares them by value in
some queries and by position in others (`(Pr (= alice.x 1))` and
`(Pr (= alice.x y))` can disagree), in Python memo as here. memo.clj warns
when such a domain is used.

One enum is predefined: `Bool`, with members `NO` (0) and `YES` (1).
Every model can name it without defining it, and a namespace's own `Bool`
takes its place (in Python memo, memo's own always wins: see
[Differences](#differences-from-python-memo)). Outside a model it is
`memo.domains/Bool`:

```clojure
(defmemo says-yes [] []
  (alice :chooses [b Bool] :wpp (+ 1 b))
  (return (Pr (= alice.b Bool.YES))))

(array/->clj (says-yes))
;=> 0.6666666666666666
(domains/Bool :YES)
;=> 1
```

A product domain packs several fields into one number, the first field
most significant. `(defdomain Cell :x 3 :y 4)` has 12 members, 0 to 11.
The domain packs a map of every field, and reads a field of a member:

```clojure
(defdomain Cell :x 3 :y 4)

(Cell {:x 1 :y 2})
;=> 6
(Cell :x 6)
;=> 1
(domains/fields Cell 6)
;=> [1 2]
```

The sizes are expressions, such as `(count (domains/values Loc))`. Inside a
model, read a field through a host function (see
[Host functions](#host-functions)). `(Cell :x s)` works on a gen.array
array too, so the host function may be `^:memo/vectorized`:

```clojure
(defn cell-x [s] (Cell :x s))

(defmemo column [] []
  (alice :chooses [s Cell] :wpp (+ 1 (cell-x s)))
  (return (E (cell-x alice.s))))

;; each x has 4 cells, weighted 1, 2 and 3
(array/->clj (column))
;=> 1.3333333333333333
```

Functions on domains, outside a model:

```clojure
(domains/values Coin)                ; the members' values
;=> [0 1]
(domains/labels Coin)                ; display labels, as strings
;=> ["HEADS" "TAILS"]
(domains/labels [0.5 1.5])
;=> ["0.5" "1.5"]
(domains/members Coin)               ; an enum's members' names
;=> [HEADS TAILS]
(domains/field-keys Cell)            ; a product's fields
;=> [:x :y]
(domains/field Cell :y 6)            ; (Cell :y 6)
;=> 2
(domains/pack Cell {:x 1 :y 2})      ; (Cell {:x 1 :y 2})
;=> 6
(count (domains/values Cell))        ; a product's members are 0 .. n-1
;=> 12
[(domains/domain? [0 1 2]) (domains/enum? Coin) (domains/product? Cell)]
;=> [true true true]

;; a product with computed fields, as defdomain builds one
(def Grid (domains/product-domain 'Grid [[:row 2] [:col 3]]))
(count (domains/values Grid))
;=> 6
```

**Array parameters.** A parameter is a scalar unless it is marked
`^:array` (Python's `a: ...`). An array parameter takes a Clojure vector or
a gen.array array, and is passed whole to host functions and memo calls:
as given to a tabulated host function (so `nth` wants a vector), and as an
array to a vectorized one (a vector becomes one; see Host functions):

```clojure
(defn table-get [table i] (nth table i))

(defmemo lookup [[x X]] [^:array table]
  (return (table-get table x)))

(array/->clj (lookup [10 20 30]))
;=> [10 20 30]
```

```clojure
;; error: Parameter table was not an array, despite being annotated as ^:array table.
(lookup 1.0)
```

A gen.array array has no `nth`. To compute with an array parameter as an
array, and to differentiate with respect to it, write the host function
with gen.array's operators and mark it `^:memo/vectorized`:

```clojure
(defn ^:memo/vectorized table-at [table i]
  (array/take table (array/astype i :int32)))

(defmemo looked-up [] [^:array table]
  (alice :chooses [x X] :wpp (exp (table-at table x)))
  (return (E alice.x)))

(array/->clj (looked-up [0.1 0.5 -0.2]))
;=> 0.9198235667016574

;; the gradient with respect to the whole table
(array/->clj ((array/grad looked-up) (array/array [0.1 0.5 -0.2])))
;=> [-0.28454227390473374 0.037000431654636734 0.247541842250097]
```

**Cast and forall.** `(cast [alice bob])`, first in a model, names its
agents: a statement of any other agent at the top level, or a reference to
one in an expression, is then an error, which catches misspelled names (an
agent inside a mind or an `imagine` is not checked, as in Python memo).
`(forall [x X])`, a statement, declares an axis, as `[x X]` among the
axes does:

```clojure
(defmemo cast-and-forall [] []
  (cast [alice])
  (forall [x X])
  (alice :knows x)
  (alice :chooses [y X] :wpp (= y x))
  (return (E alice.y)))

(array/->clj (cast-and-forall))
;=> [0.0 1.0 2.0]
```

```clojure
;; error: agent `bob` is not in the cast
(defmemo miscast [] []
  (cast [alice])
  (bob :chooses [y X] :wpp 1)
  (return 1))
```

**Multiple return values** are written as several `return` forms. They are
stacked on a leading axis (see [Executing a memo](#executing-a-memo)).

---

# Statements

Statements define what agents do: the choices they make, what they think
about other agents, and what they observe. A statement is a list whose
head is the agent and whose second element is a keyword.

## `:chooses`

`:chooses` declares that an agent makes a probabilistic choice from a
domain.

```clojure
(defenum Action WAIT JUMP RUN)
(def Actions Action)
(defn utility [a] (nth [0.0 1.0 3.0] a))

(defmemo softmax-choice [[a Actions]] [β]
  (bob :chooses [a Actions] :wpp (exp (* β (utility a))))
  (return (Pr (= bob.a a))))

(array/->clj (softmax-choice 1.5))
;=> [0.010471333531834241 0.04692926104148601 0.9425994054266796]
```

- **Agent making the choice:** `bob`
- **Name of the choice, and its domain:** `[a Actions]`
- **Probability weighting:** `:wpp` is "with probability proportional
  to" the expression.
  - For softmax decision-making, use `:wpp (exp ...)`. It is computed as a
    stable softmax, exp(e − max e), when written in exactly this form.
  - For a uniform distribution, use `:wpp 1`.

Any Clojure function of scalars can serve as a distribution (see
[Host functions](#host-functions)). A Bernoulli with p = 0.9:

```clojure
(defn ber-pmf [k p] (if (== k 1) p (- 1 p)))

(defmemo bernoulli [[x Bool]] []
  (alice :chooses [x Bool] :wpp (ber-pmf x 0.9))
  (return (Pr (= alice.x x))))

(array/->clj (bernoulli))
;=> [0.1 0.9]
```

### Making multiple choices simultaneously

An agent can make several choices at once by listing them in the same
statement:

```clojure
(defn joint [x y] (+ 1.0 (* x y)))

(defmemo two-at-once [] []
  (bob :chooses [x X] [y Y] :wpp (joint x y))
  (return (Pr (= bob.x bob.y))))

;; weights 1, 2 and 5 on the diagonal, out of 18
(array/->clj (two-at-once))
;=> 0.4444444444444444
```

### Deterministic choice with `:to-maximize` and `:to-minimize`

For argmax behavior, choose `:to-maximize` an expression. Ties (by exact
float equality) split uniformly. `:to-minimize` is the argmin.

```clojure
(defmemo argmax-choice [[a Actions]] []
  (bob :chooses [a Actions] :to-maximize (utility a))
  (return (Pr (= bob.a a))))

(defmemo argmin-choice [[a Actions]] []
  (bob :chooses [a Actions] :to-minimize (utility a))
  (return (Pr (= bob.a a))))

(array/->clj (argmax-choice))
;=> [0.0 0.0 1.0]
(array/->clj (argmin-choice))
;=> [1.0 0.0 0.0]
```

### Deterministic value with `:to-be`

To choose a value deterministically, use `:to-be`:

```clojure
(defmemo jumps [[a Actions]] []
  (bob :chooses [a Actions] :to-be Action.JUMP)
  (return (Pr (= bob.a a))))

(array/->clj (jumps))
;=> [0.0 1.0 0.0]
```

Python memo writes the member in braces, `to_be={Action.JUMP}`. memo.clj
accepts that too, `:to-be #{Action.JUMP}`, but needs no braces: an enum
member is a value in memo expressions.

You can choose several values this way, as long as they are chosen to be
the same value:

```clojure
(defmemo jumps-twice [] []
  ;; Bob jumps twice
  (bob :chooses [a1 Actions] [a2 Actions] :to-be Action.JUMP)
  (return (Pr (= bob.a1 Action.JUMP) (= bob.a2 Action.JUMP))))

(array/->clj (jumps-twice))
;=> 1.0
```

The expression may use the agent's earlier choices: `:to-be x` copies
the choice `x` (see [`:forgets-about`](#forgets-about)).

### Aliases for `:chooses`

Several aliases behave exactly like `:chooses` but do not suggest agency
or goals. They can make a model easier to read: `:given`, `:draws`,
`:assigned` and `:guesses`.

```clojure
(defenum Roles HOST GUEST COOK)

(defmemo role [[r Roles]] []
  (bob :draws [r Roles] :wpp (+ r 1))
  (return (Pr (= bob.r r))))

(array/->clj (role))
;=> [0.16666666666666666 0.3333333333333333 0.5]
```

For a uniform choice over a subset of the domain, write `:such-that`
with a condition instead of `:wpp`:

```clojure
(defenum Stores BAKERY BANK CAFE)
(defn is-open [s] (not (== s 1)))

(defmemo errand [[s Stores]] []
  (bob :chooses [s Stores] :such-that (is-open s))
  (return (Pr (= bob.s s))))

(array/->clj (errand))
;=> [0.5 0.0 0.5]
```

For a uniform choice over the whole domain (`:wpp 1`), write
`:uniformly`:

```clojure
(defmemo any-store [[s Stores]] []
  (bob :chooses [s Stores] :uniformly)
  (return (Pr (= bob.s s))))

(array/->clj (any-store))
;=> [0.3333333333333333 0.3333333333333333 0.3333333333333333]
```

## `:thinks`

`:thinks` defines what an agent believes about other agents. The agent
reasons about the reasoning of others, recursively.

```clojure
(defmemo bobs-model [] []
  (bob :thinks
    (alice :chooses [x X] :wpp 1)
    (charlie :chooses [y Y] :wpp (+ y 1)))
  (return (as bob (E (+ alice.x charlie.y)))))

(array/->clj (bobs-model))
;=> 2.3333333333333335
```

- **Agent doing the thinking:** `bob`
- **Contents of the agent's mental model:** the statements inside, one
  form each (Python memo separates them with commas).

`(as bob ...)` evaluates an expression from bob's point of view (see
[Queries](#queries)).

## `:observes`

`:observes` updates an agent's beliefs by an observation:

```clojure
(defmemo hears-x [[y X]] []
  (bob :thinks
    (alice :chooses [x X] :wpp 1)
    (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :observes alice.x :is y)
  (return (as bob (E alice.z))))

;; z equals x with probability 3/5
(array/->clj (hears-x))
;=> [0.6 1.0 1.4]
```

- **Agent making the observation:** `bob`
- **Choice being observed:** `alice.x`, another agent's choice
- **Observed value:** `y`, the name of a choice

A bare name, like `y`, names a choice of the frame the statement is
written in: here, the axis `y`. Another agent's choice, and the observer's
own, are written with a dot, `charlie.y` or `bob.y`:

```clojure
(defmemo hears-charlie [] []
  (charlie :chooses [y X] :wpp (+ y 1))
  (bob :thinks
    (alice :chooses [x X] :wpp 1)
    (alice :chooses [z X] :wpp (+ 1 (* 2 (= x z)))))
  (bob :observes alice.x :is charlie.y)
  (return (E (as bob (E alice.z)))))

(array/->clj (hears-charlie))
;=> 1.1333333333333333
```

The observed value need not be alice's actual choice. This is how false
beliefs are modeled: only the observer's beliefs change. `:sees` and
`:hears` are aliases of `:observes`.

The observed value *cannot* be an arbitrary value. For that kind of
observation, use `:observes-that` or `:observes-event` (below).

```clojure
;; error: Unknown statement syntax
(defmemo hears-seven [] []
  (bob :thinks (alice :chooses [x X] :wpp 1))
  (bob :observes alice.x :is 7)
  (return 0))
```

### Advanced observation constructs

To condition on a boolean expression, use `:observes-that`. It is
`condition(...)` in WebPPL: the agent observes that the expression is true.

```clojure
(def Bias [0.1 0.3 0.5 0.7 0.9])

(defmemo fair-or-not [] []
  (bob :thinks (coin :chooses [bias Bias] :wpp 1))
  (bob :observes-that (> coin.bias 0.5))
  (return (as bob (E coin.bias))))

(array/->clj (fair-or-not))
;=> 0.8
```

For soft evidence, use `:observes-event`. It is `factor(log(...))` in
WebPPL: the observation has probability proportional to the expression.

```clojure
(defmemo saw-heads [] []
  (bob :thinks (coin :chooses [bias Bias] :wpp 1))
  (bob :observes-event :wpp coin.bias)
  (return (as bob (E coin.bias))))

;; (0.01 + 0.09 + 0.25 + 0.49 + 0.81) / 2.5
(array/->clj (saw-heads))
;=> 0.66
```

*Most models only need the basic `:observes`.*

## `:knows`

`:knows` is a shorthand for putting values into an agent's mental model,
known:

```clojure
(defmemo told [] []
  (alice :chooses [y Y] :wpp (+ y 1))
  (bob :knows alice.y)
  (return (E (as bob alice.y))))

(array/->clj (told))
;=> 1.3333333333333333
```

bob knows `alice.y`, so `(as bob alice.y)` is a value he can give; the
observer does not, and takes its expectation. `:knows` takes one or more
choices that the frame it is written in knows: a bare name is one of that
frame's own (at the top level, an axis, as `x` here), and `alice.y` is
another agent's: `(bob :knows x alice.y)`. bob's own choices are not among
them: he knows those already, and naming one is an error. It is roughly
this expanded form:

```clojure
(defmemo told-expanded [] []
  (alice :chooses [y Y] :wpp (+ y 1))
  (bob :thinks (alice :chooses [y Y] :wpp 1))
  (bob :observes alice.y :is alice.y)
  (return (E (as bob alice.y))))

(array/->clj (told-expanded))
;=> 1.3333333333333333
```

An axis is known to the observer only. An agent that needs it must be
told:

```clojure
(defmemo knows-axis [[x X]] []
  (bob :knows x)
  (bob :chooses [z X] :to-be x)
  (return (E bob.z)))

(array/->clj (knows-axis))
;=> [0.0 1.0 2.0]
```

## `:snapshots-self-as`

`:snapshots-self-as` lets an agent remember a snapshot of its past self.
It is for counterfactuals and hypotheticals, often with `imagine`.

```clojure
(defmemo two-snapshots [] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice younger_alice)
  (return (as alice (+ (as past_alice (E bob.x))
                       (as younger_alice (Pr (= bob.x 2)))))))

(array/->clj (two-snapshots))
;=> 1.3333333333333333
```

- **Agent taking the snapshot:** `alice`
- **Names of the snapshot:** `past_alice`, and any more

A snapshot is not affected by its owner's later observations:

```clojure
(defmemo before-and-after [[x X]] []
  (alice :thinks (bob :chooses [x X] :wpp 1))
  (alice :snapshots-self-as past_alice)
  (alice :observes bob.x :is x)
  (return (as alice (as past_alice (E bob.x)))))

;; the snapshot still has its uniform prior
(array/->clj (before-and-after))
;=> [1.0 1.0 1.0]
```

Python memo before 1.3.3 gives `[0 1 2]` here: its snapshot reads values
that its owner observes later.

## `:wants` and `EU`

`:wants` defines a goal: an expression that may depend on choices not yet
made. `(EU goal)` is the goal's expected value.

```clojure
(def A [0 1 2])
(def N [0 1 2 3])

(defmemo competitor [[a A]] []
  (alice :thinks (bob :chooses [score N] :wpp 1))
  (alice :wants win (> score bob.score))
  (alice :chooses [a A] :to-maximize (EU win))
  (alice :given [score N] :wpp (exp (- (** (- score a 1) 2))))
  (return (Pr (= alice.a a))))

(array/->clj (competitor))
;=> [0.0 0.0 1.0]
```

- **Agent with the goal:** `alice`
- **Goal name:** `win`
- **Goal expression:** `(> score bob.score)`, evaluated once `score` is
  defined
- **Expected utility:** `(EU win)`

`score` is chosen after the `:wants` statement, and the model is valid:
`EU` is evaluated in alice's imagined future, where `score` exists.

`(Predict e)` is the same idea without a name: what the agent expects `e`
to be, once its later statements have run. It also appears in Python
memo, though its Handbook does not describe it.

```clojure
(defmemo predictor [] []
  (bob :chooses [z X] :wpp 1)
  (bob :thinks
    (alice :chooses [x X] :to-maximize (Predict z))
    (alice :thinks (world :knows x) (world :chooses [z X] :wpp (+ z x)))
    (alice :observes world.z :is z)
    (alice :chooses [z X] :wpp (+ world.z z)))
  (return (E (as bob (E alice.x)))))

(array/->clj (predictor))
;=> 2.0
```

`EU` and `Predict` belong to an agent: at the top level they are compile
errors.

## `:forgets-about`

Sometimes an agent's earlier choice is never referenced again:

```clojure
(defmemo remembers [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :to-be x)
  ;; from here on, x is not referenced
  (return (+ (E alice.y) (Var alice.y))))

(array/->clj (remembers))
;=> 0.75
```

Every later computation still sums over the values of `x`.
`:forgets-about` sums them out once, at that point, and the model gives
the same answer with smaller arrays:

```clojure
(defmemo forgets [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :to-be x)
  (alice :forgets-about x)
  (return (+ (E alice.y) (Var alice.y))))

(array/->clj (forgets))
;=> 0.75
```

Referencing `alice.x` afterwards is an error:

```clojure
;; error: Unknown choice alice.x
(defmemo forgot [] []
  (alice :chooses [x Bool] :uniformly)
  (alice :chooses [y Bool] :to-be x)
  (alice :forgets-about x)
  (return (E alice.x)))
```

An agent can forget several choices at once,
`(alice :forgets-about x y z)`, but only its own choices, not values it
knows or observed. Think of it as Python's `del` for a choice.

*memo's `forgets_about` was inspired by the work of Li, Guo, Wang, Wang, &
Zhang (2026, under review).*

## `:has-theory`

An agent can have an alternate theory of mind, such as a child who cannot
represent false beliefs. `(child :has-theory :child)` gives the agent the
theory `:child`: when you write the agent's mental model (with `:thinks`),
the statements inside may not use `:thinks`.

```clojure
(defenum Loc BASKET BOX)

(defmemo childs-view [] []
  (child :has-theory :child)
  (child :thinks (sally :chooses [loc Loc] :wpp (+ 1 loc)))
  (return (as child (E sally.loc))))

(array/->clj (childs-view))
;=> 0.6666666666666666
```

It is fine to write `(child :thinks ...)` at the top level: the observer
may use `:thinks`. What is disallowed is the child modeling someone who
thinks:

```clojure
;; error: Using `thinks` with variant CHILD
(defmemo sally-anne [] []
  (child :has-theory :child)
  (child :thinks
    (sally :thinks (ball :chooses [loc Loc] :wpp 1)))
  (return 0))
```

A theory can be imposed on a whole model, the observer included, with
metadata on its name (Python's `@memo(variant=Variant.CHILD)`):

```clojure
(defmemo ^{:variant :child} a-childs-model [] []
  (child :chooses [loc Loc] :wpp (+ 1 loc))
  (return (E child.loc)))

(array/->clj (a-childs-model))
;=> 0.6666666666666666
```

Three more theories (memo's ablations) each remove one construct from an
agent's theory: `:no-thinks`, `:no-knows` and `:no-wants`.

```clojure
;; error: Unknown syntactic construct `knows`
(defmemo never-told [] []
  (bob :has-theory :no-knows)
  (bob :thinks
    (alice :chooses [x X] :wpp 1)
    (carol :knows alice.x))
  (return 0))
```

## `:inspect`

`(alice :inspect)` prints alice's state while the model compiles. It helps
when debugging a model.

```clojure
(defmemo inspected [] []
  (alice :chooses [x X] :wpp 1)
  (alice :inspect)
  (return (E alice.x)))
```

prints

```text
** Tracing alice in defmemo inspected, line 3
-> alice is tracking the following choices:
   - x: X (known)
-> alice is not yet tracking any agents.
-> alice is being modeled by observer.
```

---

# Expressions

Expressions compute values within a model. They can reference agents'
choices, apply operators and functions, and use probabilistic reasoning.

## Literals and Parameters

Numbers are literals; `true` and `false` are 1 and 0. Parameters are
referenced by name.

```clojure
(defmemo literal [] []
  (return 3.14))

(defmemo parameters [] [a b c]
  (return (+ a (* b c))))

(array/->clj (literal))
;=> 3.14
(array/->clj (parameters 1.0 2.0 3.0))
;=> 7.0
```

A var's value is not visible to a model by its name. Use the
[inline escape hatch](#referencing-clojure-values) for that.

## Operators

memo.clj's operators are Clojure's prefix forms.

### Binary operators

| Operator | Description |
|----------|-------------|
| `(+ a b ...)` | Addition |
| `(- a b ...)` | Subtraction |
| `(* a b ...)` | Multiplication |
| `(/ a b ...)` | Division |
| `(** a b)`, `(pow a b)` | Exponentiation |
| `(% a b)`, `(mod a b)` | Modulo, with the sign of the divisor (Python's) |

`+`, `-`, `*` and `/` take any number of arguments and fold left.

On literals and parameters, `**` is Python's: Python memo computes them
as Python numbers, so an integer to a negative power is a float, and
division or modulo by zero is Python's error, "division by zero". On
values that vary, an integer to a negative integer power is an error, as
in NumPy (JAX's integer power gives garbage there under `jit`, though 0
for the base 2); write `(** x -1.0)`.

```clojure
(defmemo inverse [] [a]
  (return (** a -1)))

(array/->clj (inverse 2))
;=> 0.5
```

### Comparison operators

| Operator | Description |
|----------|-------------|
| `(= a b)`, `(== a b)` | Equal to |
| `(not= a b)`, `(!= a b)` | Not equal to |
| `(< a b)` | Less than |
| `(<= a b)` | Less than or equal to |
| `(> a b)` | Greater than |
| `(>= a b)` | Greater than or equal to |

### Logical operators

| Operator | Description |
|----------|-------------|
| `(and a b ...)` | Logical AND |
| `(or a b ...)` | Logical OR |
| `(xor a b ...)` | Logical XOR (folding left, as `a ^ b ^ c`) |

### Unary operators

| Operator | Description |
|----------|-------------|
| `(- x)` | Negation |
| `(+ x)` | Unary plus |
| `(not x)` | Logical negation (Python's `~x` and `not x`) |

### Built-in functions

| Function | Description |
|----------|-------------|
| `(exp x)` | Exponential (e^x) |
| `(log x)` | Natural logarithm |
| `(abs x)` | Absolute value |

```clojure
(defmemo ops [[x X]] [a]
  (return (+ (* 2 x)                 ; 0 2 4
             (** x 2)                ; 0 1 4
             (% (- x 2) 2)           ; 0 1 0
             (- a)                   ; -1
             (if (> x 1) 10 0))))    ; 0 0 10

(array/->clj (ops 1.0))
;=> [-1.0 3.0 17.0]
```

**Booleans follow JAX, as in Python memo.** A comparison gives a boolean.
`+` of two booleans is logical or, `*` is logical and, and `-` of a
boolean is an error. On parameters and literals alone, which Python memo
computes as Python numbers, a boolean is the integer 0 or 1 instead:
`(+ (> a 0) (> a 0))` is 2, and `(- (> a 0))` is -1. A weight that adds
comparisons of choices counts each value once:

```clojure
(defmemo bool-weights [] []
  (alice :chooses [x X] :wpp (+ (= x 0) (< x 2)))
  (return (Pr (= alice.x 0))))

;; weights true, true, false: 1/2, not 2/3
(array/->clj (bool-weights))
;=> 0.5
```

gen.array's own `+` counts booleans as 1 and 0 (Julia's rule), so the same
expression gives other numbers outside a `defmemo`.

**There are no bitwise operators.** `and`, `or`, `xor` and `not` work on
booleans and on integers that are 0 or 1, such as a `Bool` choice, where
bitwise and logical agree. As in JAX, the result is a boolean when every
operand is one and an integer otherwise (so `+` then adds). `Pr` with
several conditions is their `and`:

```clojure
(defmemo both [] []
  (alice :chooses [p Bool] :uniformly)
  (alice :chooses [q Bool] :uniformly)
  (return (Pr alice.p alice.q)))

(array/->clj (both))
;=> 0.25
```

Python memo computes `&`, `|`, `^` and `~` bitwise on other integers;
memo.clj makes that an error:

```clojure
;; error: memo.clj computes xor on booleans and on 0 and 1 only
(defmemo bitwise [[x X] [y Y]] []
  (return (xor x y)))

(bitwise)
```

### Conditional expressions

`(if c a b)` is memo's conditional expression, Python's `a if c else b`:

```clojure
(defmemo conditional [[x X]] [a b]
  (return (if (> x 1) a b)))

(array/->clj (conditional 1.0 2.0))
;=> [2.0 2.0 1.0]
```

When the condition is static (parameters and literals only), the branches
become *path conditions*, which is how a recursive model stops (see
[Calling Other memos](#calling-other-memos)).

### Host functions

Python memo calls functions decorated with `@jax.jit`. memo.clj calls
Clojure functions: a **host function** is any Clojure function of scalars,
defined before the model that calls it. It must return a single number
or boolean. A ratio or a `BigDecimal` it returns is used as a double (as
Python's `/` gives a float), and an integer beyond 64 bits is an error.

```clojure
(defn f [x] (Math/cos x))

(defmemo cosine [[x X]] []
  (alice :chooses [y Y] :wpp (+ 1 (f y)))
  (return (+ (E alice.y) (f x))))

(array/->clj (cosine))
;=> [1.6566213744652316 1.1969236803333714 0.2404745379180891]
```

By default a host function is **tabulated**: it is called on host
numbers, once for each element of the arguments that vary, and the
arguments that are static (parameters, literals, array parameters) are
passed whole. Any Clojure function works this way, `if`, `case` and
`bit-and` included. Keyword arguments reach the function as a trailing
map:

```clojure
(defn norm-pdf [x {:keys [loc scale]}]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x loc) scale) 2)))
     (* scale (Math/sqrt (* 2 Math/PI)))))

(defmemo near-one [] []
  (alice :chooses [n N] :wpp (norm-pdf n :loc 1 :scale 0.5))
  (return (Pr (= alice.n 1))))

(array/->clj (near-one))
;=> 0.786778329216277
```

A tabulated function cannot be differentiated. A function written with
gen.array's operators, elementwise, can be marked `^:memo/vectorized`.
It is then called once on whole arrays, and gradients flow through it:

```clojure
(defn ^:memo/vectorized gauss [x mu]
  (array/exp (array/negate (array/square (array/- x mu)))))

(defmemo mean-near [] [mu]
  (alice :chooses [n X] :wpp (gauss n mu))
  (return (E alice.n)))

(array/->clj (mean-near 1.3))
;=> 1.2501972993084811
```

Inside a vectorized function, branch with `array/where`, never `if`: an
`if` on an array takes one branch for every element. memo.clj checks
each call against the function on host numbers, at up to 32 elements, and
a mismatch is an error:

```clojure
;; error: The vectorized function step-if gives other values on whole arrays than on single values
(defn ^:memo/vectorized step-if [x] (if (array/< x 2) 1.0 0.0))

(defmemo uses-step-if [] []
  (alice :chooses [x X] :wpp (step-if x))
  (return (E alice.x)))

(uses-step-if)
```

The check runs when the result is already computed, as it always is on
the JVM backend. MLX computes lazily, and checking would force the
computation, so its results are usually not checked: check a vectorized
function once on the JVM backend.

A tabulated function that throws becomes a memo error naming the
function and its arguments, with the original exception as its cause; a
vectorized function that throws on whole arrays (say, `Math/exp` on one)
becomes a memo error naming the function.

## Choice Expressions

Once an agent makes a choice, later expressions can reference it.

```clojure
(defn g2 [x y] (Math/exp (- (Math/pow (- x y) 2))))

(defmemo own-choices [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (+ alice.x alice.y))))

(array/->clj (own-choices))
;=> 2.0
```

Within an agent's statements, its own choices are plain names (`x` and
`y` above). Another agent's choice is written with a dot, `alice.x`.
`(as alice e)` evaluates `e` from alice's point of view (Python's
`alice[e]`), so these are the same:

```clojure
(defmemo dotted [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (+ (as alice x) (as alice y)))))

(defmemo as-alice [] []
  (alice :chooses [x X] :wpp 1)
  (alice :chooses [y Y] :wpp (g2 x y))
  (return (E (as alice (+ x y)))))

[(array/->clj (dotted)) (array/->clj (as-alice))]
;=> [2.0 2.0]
```

## Probabilistic Operators

memo has operators for statistics of distributions. As in Python memo, a
query of a number rather than a condition is its mean: `(Pr (+ alice.x 5))`
is `(E (+ alice.x 5))`; and `:such-that` with a number weights by it, as
`:wpp` does.

**Expectation**, `(E e)`:

```clojure
(defmemo expectation [] []
  (alice :chooses [x X] :wpp (+ x 1))
  (bob :chooses [z X] :wpp 1)
  (return (E (+ alice.x bob.z))))

(array/->clj (expectation))
;=> 2.3333333333333335
```

**Variance**, `(Var e)`:

```clojure
(def Signed [-2 -1 0 1 2])

(defmemo variance [] []
  (alice :chooses [y Signed] :wpp (+ 1 (* y y)))
  (return (Var (* alice.y 2))))

;; 4 · 44/15
(array/->clj (variance))
;=> 11.733333333333333
```

**Probability**, `(Pr c)`:

```clojure
(defmemo probability [] []
  (alice :chooses [y Signed] :wpp (+ 3 y))
  (return (Pr (>= alice.y 0))))

(array/->clj (probability))
;=> 0.8
```

For a joint probability, give several conditions:

```clojure
(defmemo joint-probability [] []
  (a :chooses [x Signed] :wpp 1)
  (b :chooses [y Signed] :wpp (+ 3 y))
  (return (Pr (> a.x 0) (< b.y 2))))

;; 2/5 · 10/15
(array/->clj (joint-probability))
;=> 0.26666666666666666
```

## Information-Theoretic Operators

**Entropy**, `(H a.x b.y ...)`, is the joint entropy of choices, in nats:

```clojure
(defmemo entropy [] []
  (alice :chooses [x X] :wpp 1)
  (bob :knows alice.x)
  (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y alice.x))))
  (return (H alice.x bob.y)))

(array/->clj (entropy))
;=> 1.9661755171495707
```

**KL divergence**, `(KL a.x b.y)` (Python's `KL[a.x | b.y]`), between the
distributions of two choices over the same domain:

```clojure
(defmemo divergence [] []
  (alice :chooses [x X] :wpp (+ x 1))
  (bob :chooses [y X] :wpp 1)
  (return (KL alice.x bob.y)))

;; 1/6 log(1/2) + 1/2 log(3/2)
(array/->clj (divergence))
;=> 0.08720802396075805
```

## Queries

`(as agent e)` queries an agent's mental model: it evaluates `e` from that
agent's point of view.

```clojure
(defmemo query-variance [] []
  (alice :chooses [x Signed] :wpp 1)
  (return (Var (as alice (* (abs x) 2)))))

(array/->clj (query-variance))
;=> 2.24
```

The answer must be something the agent knows. Asking an agent for a value
it is uncertain about is an error, and its hint suggests `E`:

```clojure
;; error: Asking an agent for an unknown value
(defmemo mind-reading [] []
  (alice :thinks (bob :chooses [y N] :wpp 1))
  (return (as alice (= bob.y 3))))
```

```clojure
(defmemo informed [] []
  (bob :chooses [y N] :wpp (+ y 1))
  (alice :knows bob.y)
  (return (E (as alice (= bob.y 3)))))

(array/->clj (informed))
;=> 0.4
```

## Hypotheticals with `imagine`

`(imagine stmt ... e)` sets up a hypothetical world and evaluates `e` in
it. The statements change the world; the last form is the expression.

```clojure
(defmemo what-if [] []
  (alice :thinks
    (bob :chooses [x N] :wpp 1)
    (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y (% x 3))))))
  (return (imagine
           (bob :chooses [y Y] :wpp 1)
           (alice :observes bob.y :is bob.y)
           (E (as alice (Pr (= bob.x 3)))))))

;; (2/5 + 1/7 + 1/7) / 3
(array/->clj (what-if))
;=> 0.22857142857142856
```

The value of `imagine` must be known to the frame that imagines. Here
alice's answer depends on `bob.y`, imagined but unknown to the top level,
so the expression is `(E (as alice ...))`. The memo Handbook's own
example, ending in `alice[Pr[bob.x == 7]]`, is an error at the top level
for this reason:

```clojure
;; error: trying to imagine an unknown value
(defmemo what-if-as-written [] []
  (alice :thinks
    (bob :chooses [x N] :wpp 1)
    (bob :chooses [y Y] :wpp (+ 1 (* 3 (= y (% x 3))))))
  (return (imagine
           (bob :chooses [y Y] :wpp 1)
           (alice :observes bob.y :is bob.y)
           (as alice (Pr (= bob.x 3))))))
```

## Calling Other memos

One memo can call another. The call names the callee, then a vector of
the axes it is evaluated at, then the callee's parameters. The vector is
what makes it a memo call, not a host function call.

```clojure
(defmemo f3 [[x X]] [a b c]
  (alice :chooses [x X] :wpp (exp (+ (* a x) b)))
  (return (* (Pr (= alice.x x)) c)))

(defmemo g [] []
  (alice :chooses [x X] :wpp (f3 [x] 1.0 0.0 3.1))
  (return (E alice.x)))

;; (e + 2e²) / (1 + e + e²)
(array/->clj (g))
;=> 1.5752103826044415
```

Pass every parameter the callee has no default for. A memo call's
parameters must be static: parameters and literals. Keyword arguments work,
`(f3 [x] 1.0 0.0 :c 2.0)`.

**Forwarding parameters.** `...` passes all of the caller's parameters:

```clojure
(defmemo g-forwards [] [a b c]
  (alice :chooses [x X] :wpp (f3 [x] ...))
  (return (E alice.x)))

(array/->clj (g-forwards 1.0 0.0 3.1))
;=> 1.5752103826044415
```

Here `...` is `a b c`.

**Choosing a return value.** To call a model with several returns, put the
index of the one you want first:

```clojure
(defmemo two-returns [[x X] [y Y]] [t]
  (return (+ x (* y t)))
  (return (* x y)))

(defmemo uses-first [] []
  (alice :chooses [x X] [y Y] :wpp (two-returns 0 [x y] 3.14))
  (return (E (+ alice.x alice.y))))

(array/->clj (uses-first))
;=> 2.6666666666666665
```

**Recursion** stops through a static `if`: the branch that recurses is a
path condition, and the call in it does not happen once the condition
fails. This is the Rational Speech Acts model of scalar implicature:

```clojure
(defenum Utt NONE SOME ALL)
(def Apples [0 1 2 3])
(defn denotes [u n]
  (cond (== u 0) (== n 0)
        (== u 1) (> n 0)
        :else (== n 3)))

(defmemo hears [[u Utt] [n Apples]] [alpha t]
  "A listener of depth t hears u and infers how many apples were eaten."
  (listener :thinks
    (speaker :given [n Apples] :wpp 1)
    (speaker :chooses [u Utt]
             :wpp (if (= t 0)
                    (denotes u n)
                    (* (denotes u n) (exp (* alpha (hears [u n] alpha (- t 1))))))))
  (listener :observes speaker.u :is u)
  (listener :chooses [n Apples] :wpp (Pr (= speaker.n n)))
  (return (Pr (= listener.n n))))

;; hearing SOME, how likely are all three apples?
(get-in (array/->clj (hears 3.0 0)) [1 3])
;=> 0.2
(get-in (array/->clj (hears 3.0 2)) [1 3])
;=> 0.02588228860302509
```

At depth 0 the speaker says any true utterance, so SOME is said of three
apples half as often as of one or two: 0.5 / 2.5. At depth 2, "some" has
come to mean "some but not all".

A model's calls of other models run before the rest of it, as in Python
memo, so a recursion holds no tables of the levels above it. Its calls
still nest: a model that recurses 1000 levels deep (memo's demo-mdp plans
1000 steps) needs a thread with a larger stack, as Python memo needs a
higher recursion limit:

```text
(let [result (promise)]
  (.start (Thread. nil #(deliver result (Q 1000)) "deep" (* 1024 1024 1024)))   ; 1 GB of stack
  @result)
```

## Cost Reflection

Python memo's `cost @ f(...)`, `time @`, `space @` and `size @` measure
the cost of computing a model. memo.clj has not built them yet: they are
milestone M4, written `(cost @(f ...))`, which the reader reads as
`(cost (clojure.core/deref (f ...)))`. Only that shape is reserved, so a
host function may be named `cost`, as in Python memo.

## Referencing Clojure Values

Python memo's curly braces reference Python values from inside a model.
memo.clj's are `#{...}`: the form inside is Clojure, a function of the
model's parameters, and its value is used as a number (a ratio as a
double).

```clojure
(def N5 5)

(defmemo five [] []
  (return #{N5}))

(defmemo shifted [] [x]
  (return #{(bit-shift-left x 4)}))

[(array/->clj (five)) (array/->clj (shifted 3))]
;=> [5 48]
```

An enum member inside the braces is its value, as everywhere in a model,
so `#{Action.WAIT}` is `Action.WAIT`:

```clojure
(defmemo wait [] []
  (return #{Action.WAIT}))

(array/->clj (wait))
;=> 0
```

Its value must be a number, a boolean or a gen.array array; anything
else, such as a vector, is a memo error. Under differentiation, its
parameters reach it as gen.array values, so a form to be differentiated
computes with gen.array's operators, `#{(array/* 2 a)}`, not clojure.core's
(that is a memo error naming the form).

This is an escape hatch for exceptional cases. Most models never need it.

**You cannot reference memo constructs (agents or their choices) inside the
braces:**

```clojure
;; error: Cannot reference alice.x inside the inline escape hatch #{...}
(defmemo peeks [] []
  (alice :chooses [x X] :wpp 1)
  (return #{alice.x}))
```

---

# Running and Configuring memos

## Executing a memo

Call a model like a function with its parameters. It returns a gen.array
array with the declared axes:

```clojure
(defmemo executed [[x X] [y Y]] [a b]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (+ (Pr (= alice.x x)) (* b y)))
  (return (E alice.x)))

(array/shape (executed 0.5 2.0))
;=> [2 3 3]
```

- **Axes** come in the order they are declared, each at its domain's full
  size. (Python memo leaves size 1 an axis a result does not depend on;
  memo.clj broadcasts.)
- **Several returns** are stacked on a leading axis: index 0 is the first
  return.
- **A model without axes** gives a 0-d value: a number on the JVM backend,
  a 0-d array on MLX. `array/->clj` reads both as a number.

```clojure
(get-in (array/->clj (executed 0.5 2.0)) [1 0 0])
;=> 1.3201566678298065
```

Parameters may be passed by name, in a trailing map, keyed by keywords or
symbols without a namespace (`{:a 0.5}` or `{'a 0.5}`). Keys under
`:memo/` are the call's options ([Bounded minds](#bounded-minds)); any
other key is an error. Inside a model, a memo call passes names inline,
`(f [x] 0.5 :c 2.0)`; outside, they always go in the map. Defaults fill in
what is not passed:

```clojure
(= (array/->clj (model 0.5 -0.3))
   (array/->clj (model {:a 0.5 :b -0.3}))
   (array/->clj (model 0.5 -0.3 {:c 3.14})))
;=> true
```

To print the results as a table, use `print-table` (with the memo or its
var); the layout is for reading, not parsing:

```clojure
(memo/print-table jumps (jumps))
```

prints

```text
+------------+-------+
| a: Actions | jumps |
+------------+-------+
| WAIT       | 0.0   |
| JUMP       | 1.0   |
| RUN        | 0.0   |
+------------+-------+
```

A model's metadata describes it:

```clojure
(select-keys (meta executed) [:memo/params :memo/axes :memo/shape :memo/num-results])
;=> {:memo/params [a b], :memo/axes [x y], :memo/shape [3 3], :memo/num-results 2}
(memo/memo? executed)
;=> true
```

The keys under `:memo/` are `:memo/name`, `:memo/doc` (the docstring),
`:memo/params`, `:memo/axes`, `:memo/domain-names` (the axes' domains'
names), `:memo/domains` (the domains), `:memo/shape` and
`:memo/num-results`. Keys under `:memo.impl/` are memo.clj's own, and may
change. The var holds the docstring as its `:doc` and the parameters as
its `:arglists`, so `(doc executed)` shows them. `memo?` tells a memo from
other functions.

`(memo/program m)` is the model's compiled program, as text (Python's
`debug_print_compiled=True`), for reading: its format may change.

Python memo's `return_pandas`, `return_xarray` and `save_comic` have no
counterpart: read results with `array/->clj`.

## Options

Options are metadata on the model's name (Python's decorator options).

**Caching.** `^:cache` (or `^{:cache true}`) remembers results by their
parameters, Python's `@memo(cache=True)`, for calls whose arguments are
all numbers or booleans: calls with values being differentiated, or with
an array parameter, are not cached. Every cache is dropped when any memo
is defined again, since a callee may have changed.

```clojure
(defmemo ^:cache cached [] [a]
  (alice :chooses [x X] :wpp (exp (* a x)))
  (return (E alice.x)))

(identical? (cached 0.5) (cached 0.5))
;=> true
```

**Theories.** `^{:variant :child}` imposes a theory on the whole model (see
[`:has-theory`](#has-theory)).

Python memo's `debug_trace=True` has no counterpart.

## Automatic Differentiation

memo.clj models are differentiable with gen.array, as Python memo's are
with JAX. This is useful for fitting models to data by gradient descent.
`gen.array/value-and-grad` takes a function with a scalar result:

```clojure
(defmemo differentiable [] [a b]
  (alice :chooses [x X] :wpp (exp (+ (* a x) (* b x x))))
  (return (E alice.x)))

((array/value-and-grad differentiable) 0.5 -0.2)
;=> [1.0619956713012506 0.6181783957583054]
```

This gives the value and the gradient with respect to the first
parameter. For several parameters, name them with `:argnums`:

```clojure
((array/value-and-grad differentiable {:argnums [0 1]}) 0.5 -0.2)
;=> [1.0619956713012506 [0.6181783957583054 1.259789800103964]]
```

A model with axes gives an array: reduce it to a scalar first, as a loss
does. The log-likelihood of three heads and one tail under `flip`:

```clojure
(defmemo flip [[c Coin]] [p]
  (alice :chooses [c Coin] :wpp (if (= c Coin.HEADS) p (- 1 p)))
  (return (Pr (= alice.c c))))

(defn log-likelihood [p]
  (array/sum (array/* (array/array [3.0 1.0]) (array/log (flip p)))))

;; 3/p - 1/(1 - p) at p = 0.3
((array/grad log-likelihood) 0.3)
;=> 8.571428571428571
```

Only `^:memo/vectorized` host functions are differentiated. A value being
differentiated that reaches a tabulated function is an error: memo.clj
calls that function on host numbers, so a gradient cannot pass through it.
(Python memo differentiates through its `@jax.jit` host functions, and
gives a silent zero gradient only where a call's arguments are all
static.)

```clojure
;; error: Cannot differentiate through the host function gauss-host
(defn gauss-host [x mu] (Math/exp (- (Math/pow (- x mu) 2))))

(defmemo mean-near-host [] [mu]
  (alice :chooses [n X] :wpp (gauss-host n mu))
  (return (E alice.n)))

((array/grad mean-near-host) 1.3)
```

## Errors

A memo error is an `ex-info`, and `err/memo-error?` recognizes one. Its
message is Python memo's where Python memo has that error; errors only
memo.clj has (the bridge, bounded minds, the loud errors where Python memo
gives a silent number) have their own. Its data has `:memo/error` (true),
a hint (`:memo/hint`), the location (`:memo/loc`, `{:name memo :line
n}`) and the frame it was found in (`:memo/frame-ctxt`). Messages and
`format-error`'s text are for reading: they may change in any release, so
a program should branch on `memo-error?`, not on the message. A model that fails to compile throws when
its `defmemo` is evaluated, wrapped by the Clojure compiler;
`err/root-memo-error` finds the memo error inside:

```clojure
(def failure
  (try (eval '(defmemo peeking [] []
                (alice :thinks (bob :chooses [y X] :wpp 1))
                (return (as alice bob.y))))
       (catch Exception e e)))

(ex-message (err/root-memo-error failure))
;=> "Asking an agent for an unknown value"
(:name (:memo/loc (ex-data (err/root-memo-error failure))))
;=> peeking
```

`(err/format-error e)` renders an error as Python memo prints it, the
memo error inside a wrapped one included:

```text
memo error: Asking an agent for an unknown value
  in: defmemo peeking, line 4
  hint: alice has uncertainty about the value of the expression that observer is imagining alice computing. Did you perhaps mean to take alice's *expected* value of that expression, using (E ...)?
  ctxt: This error was encountered in the frame of observer. ...
```

memo's compile-time warnings, such as a redundant expectation, print to
standard error. Bind `memo.errors/*warnings?*` to false to silence them.

A memo that recurses without end, or too deep for the JVM's stack, is the
memo error "Maximum recursion depth exceeded in memo ..." (Python memo's
`RecursionError`). Its hint gives the two remedies: a base case, or a
thread with a bigger stack.

---

# Minds in gen.clj models

A memo is a function from its parameters to a table. `memo.gen` makes it
part of a gen.clj model, where a scientist's model of a person can trace,
condition on and fit what the person's mind does:

- `mg/dist` turns a memo whose return is a probability over its axes
  into a gen.clj distribution;
- `mg/deterministic` turns any memo into a deterministic generative
  function.

```clojure
(require '[memo.gen :as mg]
         '[gen.choicemap :as cm]
         '[gen.dist.core :as dc]
         '[gen.dynamic :refer [gen]]
         '[gen.gfi :as gfi]
         '[gen.random :as rng])

(defenum Move STAY HOP)

(defmemo hop [[m Move]] [beta]
  (frog :chooses [m Move] :wpp (exp (* beta m)))
  (return (Pr (= frog.m m))))

;; a distribution constructor, as gen.dist/categorical is
(def hops (mg/dist hop))

;; the log-probability that the frog hops (HOP is 1)
(dc/log-prob (hops 1.0) 1)
;=> -0.3132616875182228

(def frog-model
  (gen [beta]
    (trace :m (hops beta))))

(:weight (gfi/generate frog-model [1.0] (cm/choicemap :m 1) (rng/key 0)))
;=> -0.3132616875182228
```

- **Calls.** The constructor takes the memo's parameters as the memo
  does: positional, with defaults, or by name in a trailing map.
- **Draws are domain values:** an enum member's value, or an element of a
  vector domain, never its index. With several axes, a draw is the vector
  of their values, in the order the axes are declared.
- **log-prob** is the log of the drawn cell. A value outside the domain is
  an error.
- **The trace holds only the draw.** The mind's own choices (here the
  frog's) are summed out exactly, as memo does.
- **The table is checked at every call.** A table with NaN, a negative
  cell, or a distribution not summing to 1 is an error naming the model.
  memo returns an all-zero table silently when an observation is
  impossible, and that table fails the check. On MLX the check evaluates
  the table at each call.

```clojure
;; error: The table of memo nowhere does not sum to 1: it sums to 0.0
(defmemo nowhere [[m Move]] []
  (pond :knows m)
  (pond :thinks (frog :chooses [n Move] :wpp 1))
  (pond :observes-that (= frog.n 2))
  (return (as pond (Pr (= frog.n m)))))

((mg/dist nowhere))
```

**Gradients.** log-prob is differentiable in the memo's parameters, so
gen.clj's gradient-based inference (`choice-gradients`, HMC, MAP, VI)
reaches the parameters inside a mind. As in [Automatic
Differentiation](#automatic-differentiation), a parameter that reaches a
host function needs that function marked `^:memo/vectorized`.

```clojure
;; d/d(beta) of log e^beta / (1 + e^beta), at beta = 1
((array/grad (fn [b] (dc/log-prob (hops b) 1))) 1.0)
;=> 0.2689414213699951
```

A NaN gradient that reaches a parameter is an error naming the parameter
and the model. A memo's values can be right while its gradient is NaN, as
JAX's is. The usual case is a speaker written `(exp (* alpha (log p)))`,
where p is 0 somewhere; `(** p alpha)` has the same values and a finite
gradient.

**Given axes.** `:given` names axes (by symbols or keywords) that are
given as values, not sampled. The constructor then takes their values
first, as a vector in `:given`'s order, as a memo call does:

```clojure
(def Gap [1 2 3])

(defmemo hop-across [[g Gap] [m Move]] [beta]
  (frog :knows g)
  (frog :chooses [m Move] :wpp (exp (* beta m (- 2 g))))
  (return (Pr (= frog.m m))))

;; the gap is 3 (a value of Gap, not an index)
(dc/log-prob ((mg/dist hop-across {:given '[g]}) [3] 1.0) 1)
;=> -1.3132616875182228
```

A memo with several returns names the distribution's with `:return`:
`(mg/dist m {:return 0})`. Any other option is an error. A distribution
(or a deterministic generative function) holds the memo it was made from:
after a memo is defined again at a REPL, make it again.

**Deterministic generative functions.** `mg/deterministic` returns the
memo's table (or one return of it, with `:return`) as a deterministic
generative function, for splicing a mind at an address where a trace
should show it. Its score is 0. Outside a model, calling the memo itself
is the same. Unlike a distribution's, its values are not checked for lost
mass: an impossible observation gives 0, silently, as in memo.

```clojure
(def frog-mind (mg/deterministic hop))

(:retval (gfi/simulate (gen [beta] (splice :mind frog-mind beta)) [1.0] (rng/key 0)))
;=> [0.2689414213699951 0.7310585786300049]
```

---

# Bounded minds

*Experimental (memo.clj's M3): outside what 1.0 freezes, and a memo.clj
extension, not part of Python memo.*

memo's agents reason exactly: every expectation is computed over
everything the agent is uncertain about. A **bounded** agent instead
imagines a few samples of what it is uncertain about, and judges from
those. Its budget is given when the memo is called, keyed by agent, so the
model text does not change:

```clojure
(require '[memo.bounded :as bounded]
         '[gen.dist :as d])

(def Points [0 1 2])
(defenum Bet UNDER OVER)

;; alice bets on how many points bob scored: UNDER pays 2 - points, OVER
;; pays points
(defn payoff [bet points] (if (= bet 0) (- 2 points) points))

(defmemo bettor [[b Bet]] [beta]
  (alice :thinks (bob :chooses [points Points] :wpp (+ 1 points)))
  (alice :chooses [b Bet] :wpp (exp (* beta (E (payoff b bob.points)))))
  (return (Pr (= alice.b b))))

(array/->clj (bettor 2.0))
;=> [0.20860852732604496 0.791391472673955]

;; alice imagines one score, and bets on it
(array/->clj (bettor 2.0 {:memo/budget {'alice {:samples 1}}}))
;=> [0.3393287366540305 0.6606712633459695]

;; eight
(array/->clj (bettor 2.0 {:memo/budget {'alice {:samples 8}}}))
;=> [0.24991496729434712 0.7500850327056535]
```

The table is still exact. The agent's samples are random, but the
observer does not see them: alice's choice probability is the average
over every way her samples can fall, P(b) = E_ω[π(b | estimates(ω))],
computed by enumerating them. As the budget grows, the table approaches
the exact one.

**The budget** is `{agent spec}` under `:memo/budget` in the call's
trailing map (namespaced, so a parameter named `budget` never collides).
An agent is a symbol or a keyword. A spec takes:

| Key | Meaning |
|---|---|
| `:samples k` | each query the agent evaluates is estimated from k samples |
| `:algorithm` | `:one-and-done` (the default): samples from its current beliefs; `:rejection`: samples from its prior, kept when consistent with what it observed |
| `:per-option true` | every option of a choice has samples of its own (below) |
| `:options k` | each of its `:chooses` considers only k options, drawn uniformly |
| `:monte-carlo true` | the samples are drawn, not enumerated (with `:draws R`, default 1); the call then needs `:memo/key` |
| `:limit n` | the enumerated states allowed at one of its choices (default 10^6) |

Each part of a budget must bound something, in the memo or the memos it
calls: `:samples` a query the agent evaluates about what it is uncertain
of, `:options` a `:chooses` it makes by weight, `:to-maximize`,
`:to-minimize` or `:uniformly`. Otherwise the call is an error, since its
table would be the exact one: fitting k to a mind whose choice reads no
query would find nothing to fit. `:options k` at or above a choice's
number of options considers them all (the limit of k), which is no error.
`bounded/report` shows what a budget bounds.

```clojure
;; error: The :samples of alice's budget bound nothing in memo hunch
;; alice's weights read no query: she judges by a fixed guess of 1 point
(defmemo hunch [[b Bet]] [beta]
  (alice :chooses [b Bet] :wpp (exp (* beta (payoff b 1))))
  (return (Pr (= alice.b b))))

(hunch 2.0 {:memo/budget {'alice {:samples 8}}})
```

The call's `:memo/limit` sets the limit for every agent that has no
`:limit` of its own; an agent's `:limit` wins, higher or lower. A budget
naming an agent the memo (and the memos it calls) does not have, an
unknown key, or a value of the wrong kind is an error. A `:memo/key`
without Monte Carlo is not used.

**A budget reaches the memos called,** and an agent is bounded wherever
its name appears: in the memo called, and in every memo it calls, so a
recursive planner is bounded at every level. Two agents of the same name
in different memos cannot be bounded apart.

## What is sampled

- **Every query the agent evaluates** is estimated: `E`, `Pr`, `Var`, `H`,
  `KL`, and the expectation inside `EU` and `Predict`, each computed on
  the samples in place of the agent's beliefs. That holds in every frame
  the agent owns: its own, wherever another agent models it, its imagined
  worlds, and its future selves and snapshots. A query whose value the
  agent already knows is not an expectation, and stays exact.
- **Only what a query depends on** is sampled, so a query over one choice
  of a large model samples that choice alone.
- **Samples are fresh for each query,** and the agent's choice averages
  over them.
- **`EU` and `Predict` are two queries.** The goal is the agent's future
  self's expectation, over what the future self will still not know, and
  `EU` is the agent's expectation of that over what will happen. Each is
  estimated from samples of its own: k samples of what will happen, and,
  for a goal over something the agent never observes, k samples of that
  too. The goal's estimate is then averaged over every pair, not over k
  joint samples. `bounded/report` lists the two (their `:what` is `"EU"`
  and `"EU's goal"`). Where the future self's belief about the unobserved
  variable depends on what happened, it has a set of samples for each
  value of what happened. A `Predict`'s expression must be known to the
  future self, as in memo, so the second query is `EU`'s (or an `E` inside
  a `Predict`).
- **What the agent will observe** in its imagined future (an `:observes`
  after the choice, which memo replays as a guess of the future self's) is
  part of what happens, and sampled with it: not a query of its own.
- **An agent with no samples kept** (under `:rejection`) estimates 0, as memo's
  queries give 0 for beliefs of no mass: a future self that keeps none
  estimates its goal at 0, even the part it knows.
- **Options share samples** when the agent's belief about what it samples
  does not depend on the option. Above, alice imagines bob's score once
  and judges both bets against it (Vul et al.'s "one and done"). When
  what follows depends on the option, each option has samples of its
  own:

```clojure
(defenum Patch NEAR FAR)
(def Yield [0 1 2 3])

;; NEAR yields 1 or 2; FAR yields 0 or 3
(defn yield-odds [patch y] (if (= patch 0) ([0.0 0.5 0.5 0.0] y) ([0.4 0.0 0.0 0.6] y)))

(defmemo forager [[p Patch]] [beta]
  (fox :wants food y)
  (fox :chooses [p Patch] :wpp (exp (* beta (EU food))))
  (fox :given [y Yield] :wpp (yield-odds p y))
  (return (Pr (= fox.p p))))

;; the fox imagines two yields of each patch
(array/->clj (forager 1.0 {:memo/budget {'fox {:samples 2}}}))
;=> [0.43764976955024126 0.5623502304497587]
```

  Ties among estimates (`:to-maximize`) are split by float equality, as
  memo's ties are, so sample means equal in exact arithmetic but not in
  floats (k not a power of 2) need not tie.

  `:per-option true` gives every option samples of its own even when the
  agent's belief does not depend on it. The two are different claims
  about how a mind works, and predict different choices:

```clojure
(array/->clj (bettor 2.0 {:memo/budget {'alice {:samples 1 :per-option true}}}))
;=> [0.30826425155204645 0.6917357484479536]
```

## Observations

With `:one-and-done`, the agent samples its beliefs after what it
observed. With `:rejection`, it samples its prior and keeps the samples
consistent with its observations: an `:observes-event` weight is the
probability of keeping one, between 0 and 1 (outside, an error). When none
is kept, its beliefs have no mass, as after an impossible observation in
memo. What the agent `:knows` is not an observation: it knows the value
from the start, so no sample is rejected for it. Its observations reject
every query's samples, also those of a query about another variable.
Sampling needs probabilities, so a belief with a negative cell (from
weights of both signs, in a `:wpp` or an `:observes-event`) is an error
under a budget; weights all negative normalize to a probability, as in
memo.

```clojure
(defmemo hopeful [[b Bet]] [beta]
  (alice :thinks (bob :chooses [points Points] :wpp (+ 1 points)))
  (alice :observes-that (> bob.points 0))
  (alice :chooses [b Bet] :wpp (exp (* beta (E (payoff b bob.points)))))
  (return (Pr (= alice.b b))))

(array/->clj (hopeful 2.0 {:memo/budget {'alice {:samples 2}}}))
;=> [0.1436924381569694 0.8563075618430307]
(array/->clj (hopeful 2.0 {:memo/budget {'alice {:samples 2 :algorithm :rejection}}}))
;=> [0.17222856149157736 0.8277714385084228]
```

## Options considered

`:options k` narrows each of the agent's `:chooses` (by weight, by
`:to-maximize` or `:to-minimize`, or `:uniformly`) to k of its options,
drawn uniformly without replacement, and the agent chooses among those.
A `:given`, `:draws`, `:assigned` or `:guesses`, and a `:to-be` or
`:such-that`, are not deliberate choices, and keep all their options.
`:options` combines with `:samples`. A choice of several variables at once
has joint options (a 3 × 3 choice has 9). With k at least the number of
options, the agent considers them all: the exact choice.

```clojure
(defenum Snack APPLE BAR CAKE)
(defn appeal [s] ([1.0 2.0 4.0] s))

(defmemo snacker [[s Snack]] []
  (kid :chooses [s Snack] :wpp (appeal s))
  (return (Pr (= kid.s s))))

(array/->clj (snacker {:memo/budget {'kid {:options 2}}}))
;=> [0.17777777777777776 0.3333333333333333 0.4888888888888889]
```

## Where a bounded estimate can go

The samples are averaged at the agent's own choice, so an estimate can
drive that choice (`:wpp`, `:to-maximize`, `:to-minimize`, `:to-be`,
`:such-that`), and nothing else: returned, read by another agent, or
weighting an observation, it is an error. To model a reported estimate,
make the report a choice of the agent's, as `(alice :chooses [r R] :to-be
...)`.

A choice that is impossible for some draws loses their mass, as an
impossible choice does in memo, and the table then sums to less than 1:
options considered whose weights are all 0, a `:to-be` whose estimate is
none of the options (a sample mean not in the domain), a `:such-that`
that no option meets for some samples, and, under `:rejection`, weights
that a belief with no mass makes all 0 (a `:wpp` of `(Pr ...)` when every
sample was rejected). `mg/dist` refuses such a table.

```clojure
;; error: alice's bounded estimate is used outside alice's own choice
(defmemo guess [] []
  (alice :thinks (bob :chooses [points Points] :wpp (+ 1 points)))
  (return (as alice (E bob.points))))

(guess {:memo/budget {'alice {:samples 2}}})
```

## Sizes and the limit

Exact over samples enumerates the samples' count vectors: C(k + m - 1,
m - 1) for k samples of m values (m + 1 under `:rejection`: a sample can
be rejected), its power for options with samples of their own, and the
product of a choice's queries (two for `EU` over something never
observed). `bounded/report` gives the sizes of each choice that averages
over samples, in the memo called and in the memos it calls (`:memo`); a
choice above its limit is reported with `:over-limit true` rather than
thrown, so it sizes a budget before the limit is raised:

```clojure
(def sizes (first (bounded/report forager {'fox {:samples 2}})))

(dissoc sizes :line :sources)
;=> {:memo forager :agent fox :states 100 :limit 1000000 :monte-carlo false}

;; the fox's EU: 10 count vectors for 2 samples of 4 yields, for each of
;; the 2 patches
(dissoc (first (:sources sizes)) :line)
;=> {:kind :samples :what "EU" :k 2 :categories 4 :sets 2 :states 100}
```

Above the limit, the call is an error naming the size. The ways out: fewer
samples, a smaller sampled variable (sample what the goal depends on, at
its natural size), a higher `:limit`, or Monte Carlo.

```clojure
;; error: fox's EU from 30 samples enumerates 29767936 states, above the limit of 1000000
(forager 1.0 {:memo/budget {'fox {:samples 30}}})
```

```clojure
(:over-limit (first (bounded/report forager {'fox {:samples 30}})))
;=> true
```

memo.clj never switches to Monte Carlo by itself: a fit's density would
turn from exact into noisy partway through.

## Monte Carlo

`:monte-carlo true` draws the samples (R times, with `:draws R`) instead of
enumerating them, from the call's `:memo/key`; its cost grows with k times
R, the limit does not apply, and `bounded/report` gives R as its
`:states`. Monte Carlo draws only at the choices the observer models
directly, outside any choice's weights. A bounded agent inside another
agent's mind, or a choice inside another choice's weights (the future
self's choice that `EU` replays), stays exact over its samples, Monte Carlo
declared or not: the reasoning above samples from its probabilities, and an
estimate there would bias it. The table is then an unbiased estimate of
the exact-over-samples one, the same for the same key:

```clojure
(def mc {:memo/budget {'alice {:samples 8 :monte-carlo true :draws 4}}})

(= (array/->clj (bettor 2.0 (assoc mc :memo/key (rng/key 0))))
   (array/->clj (bettor 2.0 (assoc mc :memo/key (rng/key 0)))))
;=> true
```

The estimate is unbiased for the agent's choice probabilities, and for
anything linear in them, as a memo returning `(Pr (= alice.b b))`. What a
table computes from them nonlinearly is not: an observer conditioning on
the choice afterwards divides one estimate by another.

`mg/dist` refuses such a budget: a distribution needs exact probabilities.
`mg/estimated` gives instead a gen.clj generative function whose density is
estimated (`gen.custom/estimated`): its one choice, at `:value`, is drawn
as `mg/dist`'s draws, and its density is the estimated cell, computed with
the operation's key. Pseudo-marginal MH and PMMH on a model that splices it
then target the bounded model's exact posterior:

```clojure
(def bets (mg/estimated bettor))

;; define the budget once: gen.clj sees a splice's argument unchanged
;; only when it is the same object, and then keeps its estimate
(def bettor-model
  (gen [] (let [beta (trace :beta (d/uniform 0.5 4.0))]
            (splice :bet bets beta mc))))
```

How noisy the estimate is matters for pseudo-marginal MCMC, which mixes
well while the standard deviation of a participant's log-likelihood
estimate is about 1 to 1.7. `mg/log-estimate-variance` measures it over
a participant's trials, each `[args observed-value]`; more draws per trial
(`:draws`) lower it:

```clojure
(:draws (mg/log-estimate-variance bets (repeat 50 [[2.0 mc] 1]) {:draws 20 :key (rng/key 1)}))
;=> 20
```

## Gradients

Exact over samples is differentiable, in the agent's beliefs and in what it
values, so a bounded mind is fitted as an exact one is (the budget itself
is an integer, and is not differentiated). Under Monte Carlo the samples
carry no gradient through the beliefs they are drawn from, and
differentiating there is an error; with respect to what the agent values
(a temperature, a utility), the gradient is that of the key's table.

An exact budget keeps its constant arrays (the count vectors, their
probabilities) for later calls, up to a few hundred megabytes near the
limit; Monte Carlo keeps nothing.

```clojure
((array/grad (fn [beta] (array/take (bettor beta {:memo/budget {'alice {:samples 2}}}) 1 0))) 2.0)
;=> 0.03118199951779749
```

---

# Stability

memo.clj 1.0 freezes what is below: after it, a release may add to it,
and only a new major version removes or changes it.

- **Vars.** `memo.core`: `defmemo`, `memo?`, `print-table`, `program`.
  `memo.domains`: `defenum`, `defdomain`, `Bool`, `domain?`, `enum?`,
  `product?`, `values`, `labels`, `members`, `field-keys`, `field`,
  `fields`, `pack`, `product-domain`. A domain's record fields are not
  frozen: read them through these. `memo.errors`: `memo-error?`, `root-memo-error`,
  `format-error`, `*warnings?*`. `memo.gen`: `dist`, `deterministic`. A
  public var marked `^:no-doc` is internal.
- **The language:** the statements and expressions of this Handbook,
  Python memo's at the pinned release. New Python memo releases are
  followed deliberately: new constructs in minor releases, fixes in
  patch releases, each in the release notes.
- **Calls:** positional parameters, then a trailing map of names
  (keywords or symbols without a namespace) and options (keys under
  `:memo/`, reserved); the result's shape (axes in order, each at full
  size, several returns on a leading axis, a 0-d value without axes, read
  portably with `array/->clj` or `array/item`).
- **Host functions:** `^:memo/vectorized`, `^:array` parameters, the
  model options `^:cache` and `^{:variant ...}`.
- **Metadata and error data:** the keys under `:memo/` listed in
  [Executing a memo](#executing-a-memo) and [Errors](#errors).

Not frozen: error messages, `format-error`'s, `print-table`'s and
`program`'s text, how often a host function is called (tabulation,
spot-checks), keys under `:memo.impl/`, and **bounded minds**
(`memo.bounded`, the call options `:memo/budget`, `:memo/limit` and
`:memo/key`, `memo.gen/estimated` and `memo.gen/log-estimate-variance`),
which are experimental. `test/memo/api_test.clj` checks the frozen vars
and keys.

# Differences from Python memo

memo.clj implements every statement and expression of the memo Handbook
(1.3.3) except cost reflection. It adds two things Python memo does not
have: minds in gen.clj models (`memo.gen`), and bounded minds
(`memo.bounded`, experimental). The syntax:

| Python memo | memo.clj |
|---|---|
| `@memo def f[x: X](a, b=1.0):` | `(defmemo f [[x X]] [a (b 1.0)] ...)` |
| `class C(IntEnum)`, `Bool` | `(defenum C ...)`, `Bool` |
| `D = domain(x=3, y=4)`, `D.x(z)`, `D(x=1, y=2)` | `(defdomain D :x 3 :y 4)`, `(D :x z)`, `(D {:x 1 :y 2})` |
| `a: ...` (array parameter) | `^:array a` |
| `alice: chooses(x in X, wpp=e)` | `(alice :chooses [x X] :wpp e)` |
| `to_be=`, `such_that=`, `to_maximize=`, `uniformly` | `:to-be`, `:such-that`, `:to-maximize`, `:uniformly` |
| `alice: thinks[ s1, s2 ]` | `(alice :thinks s1 s2)` |
| `alice: observes [bob.x] is y` | `(alice :observes bob.x :is y)` |
| `alice: observes_that [e]` | `(alice :observes-that e)` |
| `alice: observes_event(wpp=e)` | `(alice :observes-event :wpp e)` |
| `alice: knows(x, bob.y)` | `(alice :knows x bob.y)` |
| `alice: snapshots_self_as(p)` | `(alice :snapshots-self-as p)` |
| `alice: wants(g=e)`, `EU[g]`, `Predict[e]` | `(alice :wants g e)`, `(EU g)`, `(Predict e)` |
| `alice: forgets_about(x, y)` | `(alice :forgets-about x y)` |
| `alice: has_theory(CHILD)` | `(alice :has-theory :child)` |
| `alice: INSPECT()` | `(alice :inspect)` |
| `cast: [alice, bob]`, `forall: x in X` | `(cast [alice bob])`, `(forall [x X])` |
| `alice[e]` | `(as alice e)` |
| `E[e]`, `Pr[a, b]`, `Var[e]` | `(E e)`, `(Pr a b)`, `(Var e)` |
| `H[a.x, b.y]`, `KL[a.x \| b.y]` | `(H a.x b.y)`, `(KL a.x b.y)` |
| `imagine[s1, s2, e]` | `(imagine s1 s2 e)` |
| `f[x](a, b)`, `f[x](...)`, `f[0][x](a)` | `(f [x] a b)`, `(f [x] ...)`, `(f 0 [x] a)` |
| `a if c else b` | `(if c a b)` |
| `x ** y`, `x % y`, `x ^ y`, `~x` | `(** x y)`, `(% x y)`, `(xor x y)`, `(not x)` |
| `{Action.WAIT}`, `{expr}` | `Action.WAIT` or `#{Action.WAIT}`, `#{expr}` |
| `@jax.jit def g(x)` | any Clojure function; `^:memo/vectorized` for gen.array ones |
| `@memo(cache=True)`, `variant=Variant.CHILD` | `^:cache`, `^{:variant :child}` |
| `f(a, print_table=True)` | `(memo/print-table f (f a))` |
| `debug_print_compiled=True` | `(memo/program f)` |
| `jax.value_and_grad(f)(a, b)` | `((gen.array/value-and-grad f) a b)` |

What memo.clj computes differently, on purpose:

- **Results are full tables.** Axes come at their domains' full sizes;
  Python memo leaves size 1 an axis a result does not depend on.
- **Precision.** The JVM backend computes in float64, MLX in float32 by
  default, Python memo (JAX) in float32. A few answers differ by more than
  rounding under float64: `KL` against a zero probability, `exp` overflow
  outside `:wpp (exp ...)` (past about 709 instead of 88), weights so small
  that float32 rounds them to zero (`:wpp 1e-50` gives zeros in Python
  memo and on MLX, uniform weights on the JVM), and `:to-maximize` ties
  (0.1 + 0.2 ties with 0.3 in float32 only). Values equal in exact
  arithmetic can also differ in the last bit between the JVM, MLX's CPU
  and Metal, so `:to-maximize` breaks such a tie differently on each.
- **Integers** are 64-bit on the JVM backend and 32-bit on MLX and in
  Python memo, which wrap silently past 2^31: `(* x 1000000000)` at x = 3
  is 3000000000 on the JVM, -1294967296 on MLX and in Python memo. A
  parameter beyond 64 bits is an error, as in Python memo.
- **Your own `Bool`.** A namespace's own `Bool` replaces the predefined
  one. Python memo's generated code imports memo's, which silently wins
  over a model's own: with `Bool` of three members, a uniform choice gives
  1/3 each here and 1/2 over memo's two there.
- **Domains are vars.** A number is not a domain (Python memo takes
  `S = 3` as the domain {3}; write `[3]`), and a parameter cannot have a
  domain's name (Python memo would take its value as the domain).
- **A memo call checks its axes' domains by value.** Python memo compares
  their names, so a caller's own `X = [2 1 0]`, passed where the callee's
  `X` is `[0 1 2]`, is silently misaligned there and an error here; the
  same values under another name pass.
- **Domain values beyond 32 bits** are an error on MLX, whose domain
  integers are 32 bits, as Python memo's OverflowError (the JVM backend's
  are 64 bits).
- **No bitwise operators.** `and`, `or`, `xor` and `not` are logical, on
  booleans and on integers that are 0 or 1. Python memo computes `&`,
  `|`, `^` and `~` bitwise on other integers (its `~x` is `True ^ x`, so
  `~2` is 3); memo.clj makes those an error.
- **Loud errors where Python memo is silent or stuck.**
  - Differentiating through a tabulated host function is an error, since
    it runs on host numbers; Python memo differentiates through its
    `@jax.jit` functions, with a silent zero gradient where a call's
    arguments are all static.
  - A memo without axes is called directly, so it is differentiated and its
    recursion stops at its path conditions; in Python memo, called from
    another memo, its gradient is zero (`jax.grad` of it directly is right)
    and a recursive one never terminates.
  - `Predict` at the top level, and a goal named twice in `:wants`, are
    compile errors with hints; Python memo raises bare assertions.
  - An observation that would put two of an agent's beliefs on one axis,
    and one whose observed value a mind inside the observer had already
    observed, make reading the beliefs they leave behind a compile error
    ("Observation against an already-used value"), with the repair in its
    hint; Python memo fails there with a shape error.
  - Reading another agent's copy of a choice its owner has forgotten
    (`:forgets-about`) is a compile error; in Python memo, a `KeyError`.
  - `self` cannot be an agent's name; Python memo accepts it, and its
    choices collide with the observer's.
  - A vectorized host function is checked against its values on host
    numbers.
  - A host function that throws is a memo error naming the function and its
    arguments.
  - An integer array to a negative integer power is an error; under `jit`,
    Python memo returns garbage (`3 ** -1` is 2111105451). So is a
    negative number to a fractional power, which Python computes as a
    complex number.
- **Caching.** A `^:cache` model can be differentiated (calls with values
  being differentiated are not cached); Python memo's `cache=True` raises a
  `TypeError` under `jax.grad`. Its cache is dropped when any memo is
  redefined; Python memo's keeps tables computed with the old definition.
- **The escape hatch is Clojure.** `#{alice.x}` is a compile error; in
  Python memo, `{alice.x}` is a `NameError` when the model is called.
- **Var** is Python memo's E[X²] − E[X]², computed as the centered
  E[(X − E[X])²] plus E[X]²(1 − M), M the total mass of the frame's
  weights: the same number (M is below 1 only when a branch has zero
  mass), without float32's cancellation when M is 1.

Where the memo Handbook and Python memo 1.3.3 disagree, memo.clj follows
Python memo:

- `~x` is listed as bitwise inversion; memo computes `True ^ x`.
- `observes [alice.x] is y`, with `y` "the observer's own choice": a bare
  name is a choice of the enclosing frame, and the observer's own choice is
  written `bob.y` (see [`:observes`](#observes)).
- The `imagine` example fails at the top level; it needs `E[...]` (see
  [`imagine`](#hypotheticals-with-imagine)).
- `jax.value_and_grad(f)(a, b)` differentiates with respect to `a` only,
  not `a` and `b`.

Not built:

- cost reflection (`cost @`, `time @`, `space @`, `size @`): milestone M4;
- `return_pandas`, `return_xarray`, `save_comic` and `debug_trace`;
- a cache of tabulated host functions per value of their static
  arguments.

The design, and the reasons for each decision, are in `DESIGN.md`.
