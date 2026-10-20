# memo.clj — Design

*Draft, 2026-10-20. Written in GenGDT (`~/code/GenGDT/notes/80-gen-memo-integration.md`
and the discussion behind it) before any code, as gen.clj's DESIGN.md was.
**Every decision marked (proposed) awaits the owner's confirmation**; the
two marked (decided) were confirmed on 2026-10-20. Corrected the same day
after probing Python memo 1.3.2 through memo-mcp (§2.4).*

## 1. Goal

memo (Chandra, Chen, Tenenbaum & Ragan-Kelley, "A Domain-Specific
Probabilistic Programming Language for Reasoning about Reasoning", PACMPL 9,
OOPSLA2, 2025, doi:10.1145/3763078; github.com/kach/memo) is a
probabilistic programming language for recursive reasoning about minds:
agents who choose, think about other agents, observe, know and want, with
exact inference over finite domains. memo.clj is **memo on the JVM, built on
gen.clj**. Its models are gen.clj generative functions, and its reasoning can
be exact (memo's semantics) or bounded (an agent that thinks with a budget).

In order:

1. **M1 — the exact core.** Every construct of Python memo at a pinned
   release (1.3.2 today, the pin of `memo-mcp/memo`), compiled to gen.array
   programs, giving Python memo's numbers on every test and demo of its own.
   - It runs on every gen.clj backend: the JVM, MLX's CPU, Metal and CUDA.
   - Gradients with respect to parameters come from gen.array's reverse-mode
     autodiff.
2. **M2 — the GFI bridge.** A memo model is a gen.clj object:
   - a distribution, where a return is a normalized distribution over
     declared axes;
   - a deterministic generative function, for expectations and utilities;

   with gen.clj's laws as tests. Minds can then be traced, conditioned on and
   fitted inside gen.clj models.
3. **M3 — bounded minds.** An agent's reasoning can be evaluated by a
   declared algorithm with a budget, instead of by exact inference over its
   beliefs. There are two lowerings, tried in this order:
   - **exact over the agent's own samples.** Where the agent samples from
     finite sets, its choice probabilities are computed by enumerating the
     possible samples, or their counts. The result is again a deterministic
     table, with no noise; M2's bridge applies, and it does not wait on
     encapsulated randomness.
   - **Monte Carlo** only where that space is too large. The query then
     compiles to a generative function with encapsulated randomness
     (gen.clj's genclj-1ogo, thesis §4.5). Its score is an unbiased estimate
     of the bounded agent's choice probability.

   The exact core is the limit of both.
4. **M4 — the rest, as needed.** The Python transpiler as user tooling (it is
   test tooling from M1), cost reflection, certification, model synthesis,
   and the static graph that gen.clj's static modeling language provides.

Its first client is GenGDT, which writes minds in memo.clj and cycles, data
and fitting in gen.clj. memo.clj is general: theory-of-mind, pragmatics,
planning and clinical models are equally its users.

**(decided, 2026-10-20)**
- Level 3 (bounded minds) is part of the ambition.
- memo.clj is JVM-only; it does not keep memo-clj's multi-host identity
  (bb, nbb, Scittle, the browser).

## 2. What we are building on — findings

### 2.1 memo, the reference

The reference is Python memo at a pinned release, MIT, © Kartik Chandra and
Tony Chen. Its parts:
- `parse.py`, `core.py`, `codegen.py`, `lib.py`, `utils.py`;
- the Handbook;
- the test suite, `demo/test.py`: 96 cases, each expecting a pass, a compile
  error (`expect='ce'`, 27 of them) or a runtime error (`expect='re'`, 3);
- about 35 demos.

The big idea (a close read is in memo-clj's `notes/memo-internals.md`;
re-checked against the 1.3.2 source on 2026-10-20, with the details that
matter for parity in §2.4):
- There is no sampling. A model is exact enumeration over the product of
  finite domains, expressed as broadcast tensor arithmetic. One global axis
  per choice.
- Every agent's perspective is a **frame**. A frame holds:
  - a normalized likelihood array (`ll`) over the choices it is uncertain
    about;
  - children, which are its models of other agents;
  - alias links ("conditions") to its parent's choices.
- Nested belief is just which axes a frame treats as known, plus the aliases.
- Every expression carries `known` and `deps`. The known/deps dataflow makes
  illegal models compile errors with hints rather than wrong numbers. For
  example, a choice's weight must be known to the chooser, and a return must
  be known to the observer. **This is memo's semantic layer, the principles
  of agency: no mind-reading, beliefs only through observation.**
- `Predict`/`EU`/`wants` evaluate a goal in the agent's imagined future,
  using the program's own continuation, retargeted to a snapshot of the
  agent. Only the agent's own later statements are replayed. A later
  `observes` becomes a guess drawn from her current belief, followed by an
  observation of that guess.

### 2.2 memo-clj, the earlier port (`~/code/memo-clj`)

A complete port of memo to Clojure on SCI (bb, nbb, Scittle), checked against
Python goldens.
- **Its design choices worth keeping:**
  - named-dimension tensors at compile time, which make broadcasting by name
    replace positional bookkeeping;
  - compilation to an SSA instruction program (data) rather than code strings;
  - immutable frames, so snapshots and imagination need no deep copy;
  - agent-first s-expression syntax;
  - frame-aware error hints.
- **Size:**
  - language core: about 3.3k lines (ast, parse, frames, compile 1.3k,
    interp, tensor, results, core, domains, errors, dialect);
  - AD, cost and IR checks: 0.9k;
  - certification and the Python transpiler: 1.6k;
  - extras: about 3.9k, including a GDT layer that is **not to be trusted**
    (GenGDT's rule, 2026-10-19).
- **On the JVM** (tested 2026-10-20): the tensor, parse, grad, IR-check and
  cost suites pass, 290 assertions. The language and parity suites do not
  compile: `defenum` and `defdomain` define vars named like `Loc.LEFT`, which
  JVM Clojure reads as a Java class name (`ClassNotFoundException`).
- **Speed:** about 88 µs per call of a small kernel on the JVM. Tensors are
  persistent vectors of doubles, interpreted.
- **Trust:** memo-clj predates GenGDT and was built fast. memo.clj lifts its
  ideas and code only after review, and verifies everything against Python
  memo itself, never against memo-clj's numbers.

### 2.3 gen.clj, the substrate (`~/code/gen.clj`)

- The GFI (simulate, generate, update, regenerate, assess, propose,
  project), with the thesis's laws as test.check properties.
- `gen.array`: one backend-neutral array API. It has:
  - broadcasting, reductions with axes, `where`, `take-along-axis`,
    `transpose`, `expand-dims` and `stack`;
  - reverse-mode autodiff (`grad`, `value-and-grad`, `custom-vjp`);
  - `compile` (one MLX graph);
  - backends: the JVM, MLX's CPU, Metal and CUDA.
- `gen.dist/defdist`: distributions as values with sample, log-prob, support
  and value-shape.
- `gen.custom/deterministic`: generative functions without choices, with
  their own gradients if needed.
- Exact enumeration (`gen.inference.enumerative`): Gen.jl's grids, and exact
  internal proposals.
- **In progress:** encapsulated randomness (genclj-1ogo), the owner's rules of
  2026-10-20. A generative function may own randomness ω outside its choice
  map, with its score an unbiased estimate of its density. ω is kept when
  nothing it depends on changes, and redrawn otherwise. Such functions get
  their own law set.

### 2.4 Python memo 1.3.2 under probe (2026-10-20)

Checked against the source, and by running small programs through memo-mcp
(JAX 0.12, float32).

**Semantics to copy exactly.** These are what memo computes, and parity
depends on them.
- **`nan_to_num`, not NaN → 0.** Every normalization, and the logs in `H`
  and `KL`, go through `jnp.nan_to_num`: NaN → 0, and ±∞ → ± the dtype's
  largest finite number. Entropy relies on the second half: 0 · log 0 comes
  out as 0 · (−max) = 0.
- **Zero mass is silent.** A choice whose weights are all zero is an
  all-zero distribution, not an error. An impossible observation zeroes the
  observer's `ll` the same way.
- **Redundant expectations skip the `ll`.** When everything in `E[...]` or
  `Pr[...]` is known to the frame, memo warns and returns the value itself
  (`Var` returns 0). After an impossible observation, `Pr` can then be 1
  while the frame's `ll` is 0.
- **`wpp=exp(e)` is matched by its syntax** and computed as a stable
  softmax, exp(e − max e). The same weight written any other way, such as
  `1.0 * exp(1000 * x)`, overflows and silently gives Pr = 0.
- **`to_maximize` breaks ties by exact float equality**, then is uniform
  over the tied values.
- **The posterior shortcut is required.** `E[x == agent.y]`, with x known
  and agent.y uncertain over the same domain, moves an axis of the marginal
  (or takes its diagonal) instead of computing the comparison. memo's own
  `post_optim` test does not fit in memory without it.
- **Results keep size-1 axes.** An axis the return does not depend on comes
  back with length 1: shape (3, 1) for `return x` over `[x: X, y: Y]`.
  Several returns are broadcast only against each other.
- **Only agents condition.** `observer: observes_that[...]` makes a child
  frame named `observer`; the root frame never conditions.
- **Memo calls need brackets.** `f[x](…)` is a memo call. `f(…)` is parsed
  as a host-function call, and it is the only way to call a memo without
  axes.
- **An observation against an already-used value** (slice 2). When the
  observer's beliefs already vary along the source's dim (it knows the
  source, and a choice it models depends on it), the observation swaps two
  live axes. A query that later reads those beliefs fails in Python memo
  with a shape error; a query that does not (a redundant expectation)
  still answers correctly. memo.clj makes the same swap, marks the frame,
  and makes a later read a compile error that names the cause and the
  repair, `observes-that`. (memo-clj rejected the observation itself,
  which loses Python memo's valid answers.)
- **Var** is computed centered, E[(X − E[X])²], as memo-clj does: Python
  memo's E[X²] − E[X]² is the same number but cancels in float32. A
  redundant Var (everything known) is zeros, as in Python memo.
- **NaN gradients where the values are right** (slice 3). A speaker written
  `exp(alpha * log(L0))`, the usual RSA form, gets NaN from jax.grad
  wherever L0 has a zero: log 0 is -inf, and the chain rule multiplies 0 by
  -inf. Its values are right, and `L0 ** alpha` has the same values with a
  finite gradient, but only for moderate alpha: it does not get the stable
  softmax that `wpp=exp(…)` does, and its weights underflow, so a referent's
  row becomes all zeros (silently; alpha = 200 in float32, against 100 still
  agreeing). memo.clj differentiates the same computation and matches JAX's
  NaN (§13 asks whether it should).
- **Booleans follow JAX, not gen.array** (found in slice 1). In JAX, as in
  NumPy, `True + True` is `True` (bool `+` is logical or, `*` logical and),
  and `bool - bool` and `-bool` are TypeErrors. gen.array converts booleans
  to integers in arithmetic instead, as Julia does: `(array/+ true true)` is
  2 and `(array/negate true)` is -1. That is gen.clj's rule by design (the
  owner's decision of 2026-10-20, genclj-6pnx; gen.clj DESIGN.md §9). memo models add comparisons
  (`wpp=(x == 0) + (x < 2)` gives P(x=0) = 1/2 in memo, 2/3 under
  gen.array's rule), so the lowering applies JAX's rules.
  - **Across the boundary,** the same expression gives different numbers
    inside a defmemo and in plain gen.clj code, such as GenGDT's cycles,
    where `+` is gen.array's (gen.clj's house style refers it). Code moved
    from one to the other should say `logical-or`, or convert booleans to
    numbers, explicitly.

**Upstream bugs.** memo.clj does not reproduce these. Each is a documented
divergence (principle 1), with its correct value derived independently and
kept in the upstream ledger (§9).
- **A snapshot's queries read values its owner observes later.** `observes`
  rebinds the generated variable holding the observed choice's values
  (`core.py`, lines 1396–1398). Frames copied by `snapshots_self_as`, or by an imagined
  future, share that variable. In the Handbook's own example, with a
  uniform prior over three values, `alice[past_alice[E[bob.x]]]` after
  `alice: observes [bob.x] is x` returns x instead of 1, and
  `Pr[bob.x == 2]` returns 0, 0, 1 instead of 1/3 each. The snapshot's
  beliefs are untouched; its queries pair them with the new values, so
  with a second variable the answer is neither the snapshot's nor its
  owner's: with bob.y = bob.x, `E[bob.x + bob.y]` gives x + 1, not 2 or
  2x. Queries that read only the `ll` (`H`, `KL`, the posterior shortcut) are unaffected, so
  demo-eig's answer stands. The same leak out of an imagined future (an
  `EU` whose continuation observes) corrupts the real frame until its own
  observation; in the probe it ended in a shape error. memo.clj must keep
  the half of this behaviour that is right: an observation reaches every
  live alias of the observed choice, such as a `knows` copy made before
  it, but never a copy (a snapshot or an imagined world). After
  `alice: thinks[ charlie: knows(bob.x) ]` and then
  `alice: observes [bob.x] is x`, `alice[charlie[bob.x]]` must give x, as
  it does in 1.3.2. Immutable frames give the second half for free, not
  the first: the observation has to reach the live aliases explicitly.
  Both cases are tests. (Done in slice 2: an observation emits a fresh
  alias of the source's value and, in the frame tree at compile time,
  repoints every choice that shared the target's value; the IR stays
  single-assignment.) memo-clj had the same bug until 2026-10-20; its
  fix (`fresh-tags` in `compile.cljc`, the same one tested against Python
  memo) and its test `copies-keep-their-own-values` are what memo.clj
  lifts.
- **Silent zero gradients.** A host call whose arguments are all static is
  evaluated outside `jit` and converted with `.item()`, so its gradient
  with respect to a parameter is 0 (0 against 0.246 in the probe). A memo
  without axes can only be called that way, so a gradient through such a
  call is also 0 (0 against 1.275).
- **A recursive memo without axes never terminates.** Path conditions guard
  only bracketed memo calls.
- **Assertions instead of hints.** `Predict`/`EU` at the top level, and two
  `wants` with the same goal name, raise a bare `AssertionError`.
- **`self` is not reserved** (§8).

**Where float64 changes the answer**, not just its last digits:
- `KL` against a zero probability: a term p · log(p/0) becomes p times the
  dtype's largest number, 1.7e38 for p = 1/2 under float32 and about 9e307
  under float64.
- `exp` overflow outside the `wpp=exp(e)` form: past about 88 under
  float32, 709 under float64.
- `to_maximize` ties: 0.1 + 0.2 ties with 0.3 under float32, not under
  float64.

## 3. Principles (proposed)

1. **The exact core is memo.** Same semantics as Python memo at the pinned
   release, checked against it. Extensions are marked as such and never
   change an exact result. Where memo.clj must diverge (the JVM's names, for
   one), the divergence is documented with its reason.
2. **A memo model is a gen.clj generative function or distribution.** The GFI
   is the contract between the two projects. There is no second inference
   interface.
3. **One numerical substrate: gen.array.** memo.clj has no tensor library,
   kernels or autodiff of its own. Gradients, MLX, compilation and batching
   come from gen.clj.
4. **Knowing is separate from computing.** The front end (parse, frames,
   known/deps, static flags, path conditions) decides what an agent can know,
   whatever the algorithm. The exact and the bounded lowering share it.
5. **A bounded mind is generative, not an approximation.**
   - A bounded agent's behaviour is *defined* as the marginal over its own
     samples, ω.
   - From the scientist's side nothing is approximated.
   - **Where ω is finite and small, its marginal is computed exactly**
     (§7.2). Only where it cannot be is ω encapsulated: one draw then gives
     an unbiased estimate of the bounded model's density, and gen.clj's
     inference targets that model's exact posterior.
   - The exact agent is the limit.
6. **Loud errors with memo's hints.** Illegal models are compile errors that
   name the frame, the choice and the fix. Silent wrong numbers are the
   failure this project exists to prevent.
7. **Verification as in gen.clj.**
   - Every emitter has an oracle (§9): Python memo for the exact core,
     gen.clj's laws for the bridge, convergence to the exact core for bounded
     minds.
   - Properties are checked over random programs.
   - Independent acceptance passes come before a milestone closes.
8. **A clean surface for modelers.** memo-clj's agent-first syntax, made safe
   for the JVM (§8).

## 4. Architecture

```
L4  tooling           inspect, explain, print-table; transpile (test oracle from M1)
L3  GFI bridge        memo/dist, memo/fn (M2) · bounded generative functions (M3)
L2  runtime           calls, sub-memo calls under path-condition guards, caching,
                      several returns, results as tables over named axes
L1  compiler          read → parse → AST → epistemic analysis (frames, known/deps,
                      static flags, path conditions, continuations) → IR (SSA over
                      named dims) → lowering:  exact → a gen.array program
                                               bounded → a generative function (M3)
L0  gen.clj           gen.array (JVM | MLX CPU | Metal | CUDA), GFI, inference
```

Namespaces (proposed; root `memo`):
- `memo.core`: `defmemo`, calling, results.
- `memo.domains`: `defenum`, `defdomain`.
- The front end: `memo.parse`, `memo.ast`, `memo.frames`, `memo.analyze` (the
  epistemic analysis, split out of compile).
- The IR: `memo.ir`, with its verifier.
- `memo.lower` (exact) and `memo.runtime`.
- `memo.gen`, the bridge (M2).
- `memo.bounded` (M3).
- `memo.pygen`, the transpiler: under `test` until M4.

memo-clj's single `compile` namespace (1.3k lines) is split along the
analysis/IR seam. M3 needs the analysis without the exact lowering.

## 5. Tensors on gen.array (M1)

- **Compile time keeps named dims.** This is memo-clj's insight: broadcasting
  by name removes the positional pad/permute/swap bookkeeping, the most
  error-prone part of Python memo's implementation.
- **Lowering assigns each dim one fixed position** in a full-rank layout with
  size-1 axes, as Python memo does (choice i lives at position −1 − i).
  - Every shape is known at compile time.
  - Runtime ops are plain gen.array broadcasting, with nothing named left.
  - Every choice in every frame, imagined and nested ones included, gets
    its own axis, so the rank is the model's total number of choices: up to
    14 in memo's demos. Every backend must broadcast at that rank.
- **Ops needed:**
  - elementwise arithmetic, comparison and logic;
  - `where`;
  - `sum` and `max` over axes with `:keep-dims?`;
  - `exp`, `log` and `abs`;
  - memo's `nan_to_num`: NaN → 0 (its conditioning convention, 0/0 → 0)
    and ±∞ → ± the dtype's largest finite number. Entropy needs the second
    half: 0 · log 0 must come out 0, not NaN. It is `where` over `isnan`
    and `isinf`. KL against a zero probability therefore depends on the
    dtype (§2.4);
  - axis swaps, which are `transpose`;
  - the diagonal collapse of `posterior-align`, which is `take-along-axis`;
  - `stack`, for several returns.

  All are in gen.array's v1 surface.
- **The FFI.** Host functions of scalars, such as `(denotes u n)`, are
  tabulated over the finite domains once per value of their static
  parameters, and cached, as memo-clj does.
  - A host function of a differentiable parameter needs a gen.array
    expression, which is traced, or a `custom-vjp`. Otherwise its gradient is
    zero, and that is a loud error, not a silent zero (proposed). Python
    memo gives exactly that silent zero, for a host call whose arguments
    are all parameters and for a call of a memo without axes (§2.4).
- **Parameters:**
  - *Path-condition* parameters (recursion depth, `(if (= t 0) …)`) must be
    host values at call time.
  - Every other parameter may be an array: differentiable, and batchable
    along a leading axis for vmap over parameter grids or particles.
- **Dtypes:** float64 on the JVM, float32 on MLX. Python memo runs float32
  under JAX, so parity tolerances are set per test. A few constructs give a
  different answer under float64, not a nearby one (§2.4, §13).
- **Results are full tables** (proposed, slice 1). A call returns the table
  over the memo's axes in declaration order and at their full sizes; several
  returns are stacked on a leading axis. Python memo leaves size 1 the axes
  a result does not depend on.
- **No bitwise operators** (slice 1). `&`, `|`, `^` and `not` are logical on
  booleans and on the literals 0 and 1, which covers memo's own uses
  (`to_be` builds `1 & (x == v)`). Python memo computes them bitwise on other
  integers; gen.array cannot, so those are a loud error.
- **Speed (to measure, not promise):** benchmarks against Python memo on JAX
  and against memo-clj, on small recursive models (RSA depth 5, Sally-Anne)
  and dense ones (Penelope). `array/compile` for repeated calls on MLX.

## 6. The GFI bridge (M2)

The contract GenGDT code depends on, so it is settled first (proposed):

- **`(memo/dist model opts & args)` → a gen.clj distribution.** Use it where a
  return is a normalized distribution over one or more declared axes, such as
  `(return (Pr (= alice.a a)))` over `[a Act]`.
  - **Values** are domain values (memo's), not indices.
  - **Sampling** is categorical on the table.
  - **log-prob** reads the table.
  - **Normalization** is checked at each call, loud beyond a tolerance.
    memo produces all-zero rows silently (zero weights, impossible
    observations; §2.4), and they fail this check.
  - **Several axes.** For a return over several axes, `opts` names the axes
    sampled. The others are given as values (conditioning by indexing).
  - **Gradients.** log-prob is differentiable in the array parameters, so
    HMC, MAP and VI reach parameters inside the mind.
- **`(memo/fn model opts & args)` → the table**, as a gen.array value: for
  expectations, utilities and anything else that is not a distribution.
  Called in a model it is plain computation. Wrapped by
  `gen.custom/deterministic`, it can be spliced at an address where needed.
- **Laws:**
  - gen.clj's distribution tests: normalization, sample frequencies, and
    log-prob against the table;
  - its GFI laws (`gen.laws`) over models that trace memo distributions;
  - gradients against finite differences;
  - batched against per-particle evaluation.
- **First GenGDT client:** an observer inverting GenGDT's synthetic agent,
  inferring its goal value and temperature from its choices. That is
  third-person inference with the agent's policy as a memo model.

## 7. Bounded minds (M3)

### 7.1 Semantics

- **What becomes bounded:** an agent's reasoning inside `thinks` (its models
  of the world and of others), and the queries it evaluates there (`Pr`, `E`,
  `EU`/`Predict`, `H`).
- **The bounded agent** draws ω, its samples of what it is uncertain about,
  by a declared algorithm with a budget, and forms its estimates from ω. Its
  choice rule (wpp, softmax, argmax) then acts on those estimates. Its
  behaviour is the marginal over ω:

      P(a | x) = E_ω[ π(a | estimates(ω), x) ]

- **The scientist's view.** In the compiled generative function ω is
  encapsulated randomness. The function's score for an observed choice is
  π(a | estimates(ω), x), an unbiased estimate of P(a | x) for one draw of ω.
  - The scientist's inference therefore targets the exact posterior of the
    bounded model: pseudo-marginal MH, SMC with unbiased weights, PMMH.
  - The agent's estimator **need not be unbiased**. Only the outer density
    estimate must be, and it is by construction. *Do not read bounded minds
    as approximations of the exact agent's probabilities*: that reading is
    biased, and muddles what the model claims.
- **Nesting.** Alice's ω includes her simulations of Bob's bounded reasoning.
  The same rule applies recursively.
- **The exact limit.** As the budget grows, the estimates converge to the
  exact core's tables. This is a statistical law checked against the exact
  lowering of the same program.
- **The front end is unchanged** (principle 4). The same programs are legal
  and illegal under both lowerings.

### 7.2 Exact over the agent's own samples first (proposed; from a side note, 2026-10-20)

**Why.** Monte Carlo estimates are unbiased, but their noise accumulates.
- A participant's log-likelihood estimate sums T per-trial log-estimates, so
  its variance grows about linearly in T.
- Pseudo-marginal MH mixes well only while the standard deviation of the
  log-likelihood estimate is around 1 to 1.7. That is Doucet, Pitt,
  Deligiannidis & Kohn (2015, *Biometrika*); verify before citing.
- With 100–200 trials per person, that needs many inner draws per trial, so
  fitting long experiments becomes slow, or the chains stick.

**The way out.** Most bounded agents that matter for GDT and memo models
sample from **finite** sets: outcomes, options, other agents' types.
- When an agent draws k i.i.d. samples from m outcomes and uses them only
  through their counts (a sample-mean EU, a vote, "one and done"), its choice
  probability is an exact sum over the C(k+m−1, m−1) count vectors, weighted
  by their multinomial probabilities. For k = 8 and m = 4 that is 165 vectors.
- Over several options with independent samples, the space is the product of
  the per-option spaces, so the size has to be watched: 81 for 2 options with
  binary outcomes and k = 8, but about 4.5 million for 3 options with m = 4.
- This is itself a finite enumeration, memo's own strength. The bounded agent
  lowers to an exact tensor program over its sufficient statistics: no noise,
  differentiable, batchable, and usable through M2's bridge.

**The rule.**
- The bounded lowering computes the agent's sampling exactly when the
  enumerated space (sufficient statistics where the analysis can find them,
  raw samples otherwise) is below a size limit. It reports the size it chose.
- It falls back to Monte Carlo with encapsulated randomness only above the
  limit, or for samplers it cannot enumerate (continuous proposals, adaptive
  budgets).
- When it falls back, it says so. Variance control (several inner draws per
  trial, correlated pseudo-marginal moves, SMC over trials) is then part of
  the fitting recipe, with the estimator's log-variance reported per
  participant.
- Both lowerings must agree. The Monte Carlo lowering's average converges to
  the exact-over-samples table (a law, §9).

### 7.3 Proposals (to discuss before M3)

- **The budget is a call-time argument, keyed by agent**, so the model text
  does not change, e.g. `{:budget {alice {:samples 8}}}`.
  - Moors' operating conditions belong to the agent and the moment.
  - The budget can itself be a function of the outer model's state, or a
    random variable. Urgency narrowing computation is then expressible in
    GenGDT as `budget = f(goal value)`.
- **First algorithms:**
  - sample the agent's belief and average (forward sampling where nothing is
    observed);
  - likelihood weighting or self-normalized importance sampling where the
    agent observes;
  - a fixed number of sampled futures for `Predict`/`EU` ("one and done",
    Vul et al.);
  - a few options considered (Moors' accessibility).

  Each is a cognitive hypothesis, not only a technique.
- **Open research questions:**
  - which algorithms are psychologically meaningful;
  - how bounded observation (rejection vs weighting) interacts with
    snapshots and continuations;
  - what cost means (op counts vs samples), and how it relates to memo's
    `cost @`.
- **Dependencies:** gen.clj's encapsulated randomness (genclj-1ogo) and its
  law set.
- **First GenGDT client:** the agent that imagines k outcomes before it
  chooses, with the exact agent (k → ∞) as its limit.

## 8. Surface syntax (proposed)

memo-clj's agent-first syntax, unchanged where the JVM allows:

```clojure
(defenum Utt SILENT SOME ALL)
(defenum Count NONE ONE TWO ALL)

(defmemo L1 [[u Utt] [n Count]] [alpha]
  (listener :thinks
    (speaker :given [n Count] :uniformly)
    (speaker :chooses [u Utt] :wpp (S1 [n u] alpha)))
  (listener :observes speaker.u :is u)
  (listener :chooses [n Count] :wpp (Pr (= speaker.n n)))
  (return (Pr (= listener.n n))))
```

- **Dotted references inside `defmemo`** (`speaker.u`, `Utt.SOME`) stay. The
  macro reads them as data and never emits them as Clojure symbols. An enum
  member is resolved to its value at macroexpansion, which fixes memo-clj's
  JVM failure where it arose.
- **Outside `defmemo`**, an enum is a function from keyword to value,
  `(Utt :SOME)`, and a domain's fields are `(G :x z)`. No vars with dots in
  their names.
- **The agent name `self`** is reserved: using it as an agent is a compile
  error. This diverges from Python memo 1.3.2 (principle 1), which accepts
  `self: chooses(x …)`; its choice takes the observer's own key
  `(self, x)`, and the answer comes out right by accident. memo-clj
  accepted it and returned Pr = 1.
- **memo's bare assertions become compile errors with hints:** `Predict` or
  `EU` at the top level, and a goal named twice in `wants` (§2.4).
- **Statements and expressions** are those of Python memo 1.3.2. memo-clj's
  README has the table as of 1.2.26; 1.3.2 adds `time @`, `space @` and
  `size @`, and rejects `is` in expressions. It is to be re-checked against
  the Handbook.

## 9. Verification

- **The external oracle: Python memo at the pinned release**, in a fresh clone
  of `kach/memo`, not the copy inside memo-clj. A script dumps goldens from:
  - its Handbook;
  - its test suite, including the 30 compile-error cases, whose messages must
    be matched in content;
  - its demos.

  Parity tolerances follow the dtypes.
- **The upstream ledger.** The oracle is not always right (§2.4). A ledger
  lists each known upstream bug: a minimal program, Python memo's answer,
  the correct answer derived by hand or by brute-force enumeration, and the
  upstream issue once filed. A disagreement found by the parity suite or by
  fuzzing goes into the ledger or is fixed in memo.clj; it is never
  accepted silently.
- **Comparing tables.** memo returns size-1 axes where a return does not
  depend on an axis (§2.4), so results are broadcast to their declared
  shape before comparison.
- **Differential fuzzing.**
  - Random well-formed programs from a grammar-based generator run in
    memo.clj and, through the transpiler, in Python memo, and must agree.
  - Mutated programs must be rejected by both.
  - This is the strongest check available, and the reason the transpiler is
    test tooling from M1.
- **Backend conformance:** the same programs give the same tables on the JVM
  and on MLX, within dtype tolerance.
- **The bridge (M2):** gen.clj's distribution and GFI laws, finite-difference
  gradients, batched against per-particle.
- **Bounded minds (M3):**
  - the exact-over-samples lowering against brute-force enumeration of the
    raw samples (where both fit);
  - the Monte Carlo lowering's average against the exact-over-samples table;
  - gen.clj's laws for encapsulated randomness;
  - convergence to the exact core as the budget grows;
  - the outer estimate's unbiasedness in exp space.
- **Acceptance passes** before each milestone closes: independent agents
  working as memo users from the Handbook and the docstrings alone, against
  Python memo's answers.

## 10. What is lifted from memo-clj, and what is not

| Lifted, after review | Not lifted (or deferred) |
|---|---|
| the parser and AST (agent-first syntax), frames, the compiler's analysis and IR design, error hints | the portable interpreter and named-dim runtime tensors (replaced by gen.array), the JS and wasm kernels, `memo.backend`, the nbb runners, the browser build |
| the ported Python test suite and the demo corpus, re-run against fresh goldens | memo.grad (forward-mode AD; gen.array's reverse mode replaces it) |
| the transpiler (`pygen`) and the round trip, as test tooling | the certification stack and Malli grammar (M4), the JSON AST mirror, the studio, the LLM proposer, synthesis, the Task IR |
| the IR verifier (`ircheck`) | **the GDT layer** (`memo.gdt`, `loop`, `belief`, `link`, `recover`, `confirm`): untrusted, and outside memo's scope; GDT belongs in GenGDT |
| op-count cost reflection (M4) | the two-agents page, voice, the archived training arc |
## 11. Dependencies (proposed)

- **The core:** Clojure and gen.clj, pinned to a commit, the same policy as
  GenGDT's. Nothing else.
- **Tests:** test.check and the Cognitect test runner.
- **Oracle tooling (optional, documented):** Python 3, and a virtual
  environment with the pinned memo release and JAX, for goldens and
  differential fuzzing. Tests that need it skip loudly without it.

## 12. Conventions (proposed)

- **Namespace root `memo`.**
- **Domains are values; indices are 0-based internally.**
- **Errors are `ex-info`** with `:memo/hint`, `:memo/loc` and
  `:memo/frame-ctxt`. The message content follows Python memo's.
- **Explicit keys.** No randomness in M1–M2. In M3, keys are explicit and
  threaded, as in gen.clj.
- **gen.array's operators are referred in place of clojure.core's,** gen.clj's
  house style.
- **License and attribution:** memo.clj is a port of an MIT-licensed design.
  Keep a compatible license, and credit Chandra et al. and memo's authors in
  the README.

## 13. Open questions for the owner

1. **The bridge's API** (§6): `memo/dist` and `memo/fn` as proposed, or
   something else? This is what GenGDT models will be written against.
2. **The budget's place** (§7.3): call-time and keyed by agent, as proposed?
   Or also declarable in the model text?
3. **Choosing the bounded lowering** (§7.2): automatic below a size limit and
   reported, as proposed, or always declared by the modeler? And what limit?
4. **Upstream:** does the exact core track new memo releases, with the pin
   moved deliberately? Or freeze at 1.3.2 until M3?
5. **Continuous variables in minds:** memo is finite. Is discretizing enough,
   or should bounded minds (M3) admit gen.clj distributions inside an agent's
   model? That would go beyond memo's semantics, so it would be an extension
   by principle 1.
6. **The name:** check that "memo.clj" does not collide with an existing
   project ("gen.clj" collided with probcomp's Gen.clj).
7. **Git and tracking:** a repository and a beans prefix (`memoclj-`?) from
   the start, as gen.clj has.
8. **Float64 against float32 parity** for the constructs where the dtype
   changes the answer (§2.4): run those parity tests in float32, keep them
   out of the fuzzer's grammar, or accept documented differences?
9. **Zero-mass distributions:** memo returns them silently (§2.4). Keep
   that for parity, warn, or make it an error outside a parity mode? This
   is where principle 1 (same semantics) meets principle 6 (no silent wrong
   numbers).
10. **NaN gradients where the values are right** (§2.4): match JAX, as now;
    warn when a gradient is NaN; or differentiate with the convention that
    a zero-probability term contributes 0 (which makes fitting the usual RSA
    form work, and departs from JAX)?
