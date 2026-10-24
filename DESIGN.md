# memo.clj — Design

*Draft, 2026-10-20. Written in GenGDT (`~/code/GenGDT/notes/80-gen-memo-integration.md`
and the discussion behind it) before any code, as gen.clj's DESIGN.md was.
Corrected the same day after probing Python memo 1.3.2 through memo-mcp
(§2.4). **What is marked (decided) was confirmed by the owner on
2026-10-20:** §1's two decisions; the principles, namespaces, surface
syntax, dependencies and conventions (§3, §4, §8, §11, §12); the choices
made while building M1 (§2.4, §5, §8); and §13.7 and §13.10. Everything
marked (proposed) still awaits confirmation.*

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
   release (1.3.3 since 2026-10-21, the pin of `memo-mcp/memo`), compiled
   to gen.array programs, giving Python memo's numbers on every test and
   demo of its own.
   - It runs on every gen.clj backend: the JVM, MLX's CPU, Metal and CUDA.
     Its whole suite passes on each (2026-10-22, gen.clj 3022944): the JVM
     and Metal, MLX's CPU in float32 and in float64 on the Mac and on x86,
     and CUDA on an RTX PRO 6000 (fenrir).
   - Gradients with respect to parameters come from gen.array's reverse-mode
     autodiff.
2. **M2 — the GFI bridge.** A memo model is a gen.clj object:
   - a distribution, where a return is a normalized distribution over
     declared axes;
   - a deterministic generative function, for expectations and utilities;

   with gen.clj's laws as tests. Minds can then be traced, conditioned on and
   fitted inside gen.clj models. *Closed 2026-10-24.*
3. **1.0 — the release** (added by the owner, 2026-10-24). The exact core
   and the bridge, frozen (M3, built first, stays outside the freeze as
   experimental):
   - the API of `memo.core`, `memo.domains`, `memo.errors` and `memo.gen`
     stable, with breaking changes only in a new major version;
   - the Handbook complete for both;
   - §13's open questions decided (4, 6, 8, 9);
   - a tag, timed with gen.clj's 1.0 if it has one by then. GenGDT's Phase A
     targets the 1.0 APIs, and moves its pins there. *2026-10-24
     (memoclj-p6zs):* memo.clj's 1.0 waits for gen.clj's 1.0 (gen.clj's
     owner, via its session gen-clj-65; genclj-04h2, a draft without scope
     or date), and pins it by gen.clj's GitHub URL (memoclj-kxj1). Until
     then memo.clj runs against gen.clj's HEAD now and then, and settles
     which gen.clj surface it needs frozen (memoclj-x05s).
4. **M3 — bounded minds.** An agent's reasoning can be evaluated by a
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

   *Built 2026-10-24, before 1.0 (the owner's decision that day), and
   experimental: the `:memo/budget` call option, `memo.bounded` and
   `memo.gen/estimated` are outside what 1.0 freezes (§7). Closed the same
   day, after three acceptance passes; every backend passes.*
5. **M4 — the rest, as needed.** The Python transpiler as user tooling (it is
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
(JAX 0.12, float32). 1.3.3 (2026-10-21, the pin since that day) changes
only `Frame.copy`, fixing the snapshot bug below; the rest of this section
holds for it.

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
  `post_optim` test does not fit in memory without it (10^12 cells; the
  parity corpus runs it at full size). Where the shortcut does not apply,
  both build the whole table: `post_optim_distinctness`'s 10^9 cells take
  4 GB at float32 (JAX, MLX) and more than 14 GB at float64 on gen.array's
  JVM backend, which fuses nothing, so the corpus runs it over 30 values.
- **Results keep size-1 axes.** An axis the return does not depend on comes
  back with length 1: shape (3, 1) for `return x` over `[x: X, y: Y]`.
  Several returns are broadcast only against each other.
- **Only agents condition.** `observer: observes_that[...]` makes a child
  frame named `observer`; the root frame never conditions.
- **Memo calls need brackets.** `f[x](…)` is a memo call. `f(…)` is parsed
  as a host-function call, and it is the only way to call a memo without
  axes.
- **Literals and parameters are Python numbers** (the first acceptance
  pass, 2026-10-21). memo hoists them out of JAX and emits `l ** r`, so
  `**` on them is Python's: `2 ** -1` is 0.5 (gen.array's integer power
  gives 0), `0 ** -1` raises "zero to a negative power", and a negative
  number to a fractional power is complex (memo.clj: a memo error). The
  parity fixtures record an integer argument as an integer, so the twins
  are called with the numbers Python memo was given; until then every
  argument was recorded as a float, and this difference could not show.
- **An observation against an already-used value** (slice 2; decided, 2026-10-20). When the
  observer's beliefs already vary along the source's dim (it knows the
  source, and a choice it models depends on it), the observation swaps two
  live axes. A query that later reads those beliefs fails in Python memo
  with a shape error; a query that does not (a redundant expectation)
  still answers correctly. memo.clj makes the same swap, marks the frame,
  and makes a later read a compile error that names the cause and the
  repair, `observes-that`. (memo-clj rejected the observation itself,
  which loses Python memo's valid answers.) *The first acceptance pass
  (2026-10-21) found the nested case:* a mind inside the observer that
  observed the target first holds its own alias of it, which the
  observation's repoint does not reach, and beliefs along its old dim.
  Python memo fails with a shape error when they are read; memo.clj marks
  them, and reading them is the same compile error, with the repair
  `:knows` (a knows-copy shares the value and follows it).
- **Var** is computed centered (decided, 2026-10-20), E[(X − E[X])²], as memo-clj does: Python
  memo's E[X²] − E[X]² is the same number but cancels in float32. A
  redundant Var (everything known) is zeros, as in Python memo.
  *Corrected 2026-10-20 (the differential fuzzer; for the owner's
  review):* the two are the same number only when the frame's weights sum
  to 1. Where a branch has zero mass (a `:to-be` or `:such-that` no value
  satisfies), Python memo's E is over weights of total mass M < 1, and
  E[X²] − E[X]² differs from the centered form by E[X]²(1 − M). memo.clj
  now adds that term: Python memo's number always, the centered precision
  when M is 1. `test/fixtures/fuzz.edn` keeps the finding.
- **NaN gradients where the values are right** (slice 3). A speaker written
  `exp(alpha * log(L0))`, the usual RSA form, gets NaN from jax.grad
  wherever L0 has a zero: log 0 is -inf, and the chain rule multiplies 0 by
  -inf. Its values are right, and `L0 ** alpha` has the same values with a
  finite gradient, but only for moderate alpha: it does not get the stable
  softmax that `wpp=exp(…)` does, and its weights underflow, so a referent's
  row becomes all zeros (silently; alpha = 200 in float32, against 100 still
  agreeing). memo.clj differentiates the same computation and matches JAX's
  NaN; the M2 bridge makes a NaN gradient a loud error (§13.10, decided).
- **Booleans follow JAX, not gen.array** (found in slice 1; decided, 2026-10-20). In JAX, as in
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
- **The gradient of `abs` at 0 is JAX's, 1** (the first acceptance pass,
  2026-10-21). gen.array's is 0, its sign, so in `exp(-|x - mu|)` a
  parameter equal to a domain value lost half its gradient (0.2826
  against 0.5652). The lowering computes `abs` on floats as
  `where(x >= 0, x + 0.0, -x)`.

**Upstream bugs.** memo.clj does not reproduce these (decided, 2026-10-20). Each is a documented
divergence (principle 1), with its correct value derived independently and
kept in the upstream ledger (§9).
- **A snapshot's queries read values its owner observes later** (fixed
  upstream in 1.3.3). `observes`
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
  it does in Python memo. Immutable frames give the second half for free,
  not the first: the observation has to reach the live aliases explicitly.
  Both cases are tests. (Done in slice 2: an observation emits a fresh
  alias of the source's value and, in the frame tree at compile time,
  repoints every choice that shared the target's value; the IR stays
  single-assignment.) memo-clj had the same bug until 2026-10-20; its
  fix (`fresh-tags` in `compile.cljc`, the same one tested against Python
  memo) and its test `copies-keep-their-own-values` are what memo.clj
  lifts. (Done in slice 5: snapshots and imagined worlds take fresh value
  names; the ledger's snapshot and imagine entries require the correct
  values, and a planted bug that drops the renaming fails them.) Fixed
  upstream in 1.3.3 (kach/memo `a354de8`, issue 126, 2026-10-21):
  `Frame.copy` gives a copy its own lls and choice tags, the same fix. With
  the pin moved to 1.3.3 (bean memoclj-0s5l), the four ledger entries are
  ordinary slice5 cases, the Handbook's example is ported directly, Python
  memo's six tests of the fix are in testpy.edn, and the fuzzer generates
  observations after snapshots.
- **Silent zero gradients.** A host call whose arguments are all static is
  evaluated outside `jit` and converted with `.item()`, so its gradient
  with respect to a parameter is 0 (0 against 0.246 in the probe). A memo
  without axes can only be called that way, so a gradient through such a
  call is also 0 (0 against 1.275).
- **A recursive memo without axes never terminates.** Path conditions guard
  only bracketed memo calls.
- **Assertions instead of hints.** `Predict` at the top level, and two
  `wants` with the same goal name, raise a bare `AssertionError`. (`EU` at
  the top level is a MemoError, "Unknown goal", corrected by the first
  acceptance pass.)
- **`self` is not reserved** (§8).
- **Met by the acceptance checks** (2026-10-21), memo.clj sound in
  each:
  - A memo call compares its axes' domains by name: another module's `X`
    of other values is silently misaligned (memo.clj compares values).
  - Duplicate values in a list domain: equality compares values in some
    queries and positions in the posterior shortcut (memo.clj the same,
    and warns).
  - `g[x, x]` on a memo of two axes crashes in JAX (memo.clj: a compile
    error, the same choice passed twice); `h[()](a)` crashes.
  - `KL[3 | 4]` is 7: a `KL[...]` other than `KL[p.x | q.y]` is read as a
    query of an agent named KL, and `|` is bitwise or (so a malformed KL
    reports "Unknown choice a.x"). memo.clj: "Unexpected variable in (KL
    ...)".
  - int32 arithmetic wraps silently past 2^31 (as on MLX; the JVM
    backend computes in 64 bits, a documented difference).
  - The generated code imports memo's own `Bool`, which silently wins over
    a model's own `Bool` (memo.clj lets the namespace's win, decided
    2026-10-21).
- **Forgetting a choice another mind refers to** (the first acceptance
  pass). After `alice: forgets_about(x)`, reading a copy of alice.x that
  bob holds (through `knows`, or an observation), or bob's beliefs that
  vary with it, is a `KeyError` when the memo is defined. memo.clj marks
  them, and reading one is a compile error; unread, they are harmless in
  both.
- **Integer arrays to negative powers are garbage** (the first acceptance
  pass). Under `jit`, `x ** -1` over int32 `[1 2 3]` is
  `[1 0 2111105451]`; eager JAX rejects it ("Integers cannot be raised to
  negative powers"). memo.clj raises that error (principle 4).

**Where float64 changes the answer**, not just its last digits:
- `KL` against a zero probability: a term p · log(p/0) becomes p times the
  dtype's largest number, 1.7e38 for p = 1/2 under float32 and about 9e307
  under float64.
- `exp` overflow outside the `wpp=exp(e)` form: past about 88 under
  float32, 709 under float64.
- `to_maximize` ties: 0.1 + 0.2 ties with 0.3 under float32, not under
  float64.

**The Handbook, checked entry by entry** (the parity corpus, 2026-10-20).
Every code block and code-led table row has a twin run in both
(`handbook.py`, `handbook_test.clj`). Where the Handbook and Python memo
disagree, memo.clj follows Python memo:
- `~x` is listed as bitwise inversion; memo computes `True ^ x`, so `~2`
  is 3, not -3 (memo.clj: `not` on booleans and on 0 and 1 only, §5).
- `observes [alice.x] is y` with `y` "the observer's own choice": a bare
  `y` names the enclosing frame's choices, and an agent's own choice is
  rejected ("Observation of unknown choice"); it is written `bob.y`.
- The `imagine` example, `imagine[…, alice[Pr[bob.x == 7]]]`, fails at
  the top level ("trying to imagine an unknown value"): alice's answer
  depends on the imagined `bob.y`. It needs `E[alice[…]]`.
- `jax.value_and_grad(f)(a, b)` differentiates with respect to `a` only
  (JAX's `argnums=0`), not "the parameters `a` and `b`".

## 3. Principles (decided, 2026-10-20)

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
L3  GFI bridge        memo.gen: dist, deterministic (M2, proposed) · bounded generative functions (M3)
L2  runtime           calls, sub-memo calls under path-condition guards, caching,
                      several returns, results as tables over named axes
L1  compiler          read → parse → AST → epistemic analysis (frames, known/deps,
                      static flags, path conditions, continuations) → IR (SSA over
                      named dims) → lowering:  exact → a gen.array program
                                               bounded → a generative function (M3)
L0  gen.clj           gen.array (JVM | MLX CPU | Metal | CUDA), GFI, inference
```

Namespaces (decided, 2026-10-20; root `memo`):
- `memo.core`: `defmemo`, calling, results.
- `memo.domains`: `defenum`, `defdomain`.
- The front end: `memo.parse`, `memo.ast`, `memo.frames`, `memo.analyze` (the
  epistemic analysis, split out of compile).
- The IR: `memo.ir`, with its verifier.
- `memo.lower` (exact) and `memo.runtime`.
- `memo.gen`, the bridge (M2).
- `memo.bounded` (M3; built 2026-10-24, experimental): the budget, the
  combinatorics, and the lowering of the bounded IR instructions. The
  analysis (`memo.analyze`) emits them when it is given a budget; the exact
  program is compiled without one.
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
- **Host functions** (built in slice 4; decided, 2026-10-20). A host function, such as
  `(denotes u r)`, is a Clojure function of scalars, as Python memo's are
  jitted functions of scalars.
  - **By default it is tabulated:** called on host numbers at every element
    of its dynamic arguments (its static ones passed whole), which is
    Python memo's vmap. Any Clojure function works, `bit-and` and `if`
    included. Caching the table per value of the static arguments, as
    memo-clj did, is not built yet.
  - **A function marked `^:memo/vectorized`** (written with gen.array's
    operators, elementwise, branching with `where`; or a `custom-vjp`) is
    called once on whole arrays, and so differentiated. An `if` on a value
    inside it would take one branch for every element, silently, so each
    call is spot-checked against the function on host numbers at up to 32
    evenly spaced elements (all of them, for most memo domains), and a
    mismatch is a loud error. The check runs on eager backends; on MLX it
    would force evaluation, so vectorized functions are best validated
    once on the JVM backend. Under gen.array's batch tracer (`gen.batched`)
    it checks up to 8 evenly spaced particles, each as its own run would be
    (2026-10-24, memoclj-m2jw). Before, it read a batched value on the
    host, which sent `gen.batched` one particle at a time.
  - **A value being differentiated that reaches an unmarked function is a
    loud error.** Python memo gives a silent zero gradient for a host call
    whose arguments are all parameters (§2.4).
  - **A memo without axes,** which can only be called this way, is called
    directly, so differentiated, and only when the path conditions around
    the call hold, as a memo call is. Python memo gives its call a zero
    gradient, and a recursive one never terminates.
  - **Keyword arguments** reach the function as a trailing map.
  - gen.array's batch tracer (vmap) runs a vectorized function on every
    particle at once. At gen.clj 3022944 it does not compose with
    gradients: a differentiated parameter in a batched run fails ("arrays
    from different backends (:batch and :tape)"). §6.4 has gen.clj's plans.
- **Parameters:**
  - *Path-condition* parameters (recursion depth, `(if (= t 0) …)`) must be
    host values at call time.
  - Every other parameter may be an array: differentiable, and batchable
    along a leading axis for vmap over parameter grids or particles.
- **Dtypes:** float64 on the JVM, float32 on MLX. Python memo runs float32
  under JAX, so parity tolerances are set per test. A few constructs give a
  different answer under float64, not a nearby one (§2.4, §13).
  - **MLX's CPU in float32** passes memo.clj's whole suite since gen.clj
    73f22a0 (2026-10-22, on the Mac and on x86; memoclj-rkvc).
  - Before, MLX's CPU summed float32 sequentially, 1.4e-5 off at 10,001
    terms, which put demo-scalar and demo-empowerment outside float32's
    tolerance.
  - gen.clj now sums a float32 sum of more than 256 terms there in blocks
    of 32 (genclj-y0mp, genclj-v3uw). demo-empowerment, entropies over
    528,125 unequal cells, is 5.1e-6 from float64, against 3.6e-6 under
    JAX and 2.2e-6 on Metal.
  - Blocks of 256 had left it at 1.9e-5, though each sum was within a few
    ulps: the error compounds through the model's five levels and the
    cancellation in H(x) + H(y) − H(x,y).
- **Results are full tables** (slice 1; decided, 2026-10-20). A call returns the table
  over the memo's axes in declaration order and at their full sizes; several
  returns are stacked on a leading axis. Python memo leaves size 1 the axes
  a result does not depend on.
- **No bitwise operators** (slice 1; decided, 2026-10-20). `&`, `|`, `^` and `not` are logical on
  booleans and on the literals 0 and 1, which covers memo's own uses
  (`to_be` builds `1 & (x == v)`). Python memo computes them bitwise on other
  integers; gen.array cannot, so those are a loud error. *Widened by the
  first acceptance pass (2026-10-21):* on integers that are all 0 or 1 too
  (a `Bool` choice, `Pr[alice.p, alice.q]`), read on the host, and with
  JAX's result type: a boolean when every operand is one, an integer
  otherwise, so `(+ (and b 1) (and b 1))` is 2, as in Python memo.
- **Memo calls run first** (2026-10-20, from demo-mdp). The lowering runs
  every `:call-memo` instruction, with what it reads (statics), before the
  rest of the program, as Python memo hoists its calls out of the kernel.
  A recursion then holds no tables of the levels above it: before, the JVM
  backend ran out of memory on demo-mdp's Q(1000), whose levels each hold
  tables of 1.5M cells. The calls still nest, so a 1000-level recursion
  needs a large stack (the demo tests run calls on a 1 GB-stack thread, as
  Python memo needs a higher recursion limit).
- **Memory on the JVM backend** (2026-10-20, from demo-empowerment).
  gen.array's JVM backend is eager and stores every dtype as doubles, so
  it holds in memory every value the lowering keeps. demo-empowerment's
  tables have 3125 x 169 x 169 cells, 714 MB each. Its model q's values,
  kept to the end of the model, peak at 10.7 GB; and the spot-check of a
  vectorized function built its arguments and result whole on the host
  (four times 89M numbers for Tr). The demo ran out of a 16 GB heap. Both
  were memo.lower's: it now drops each value after its last use (a peak
  of 2.9 GB) and spot-checks the sampled elements only. The demo then
  runs on the JVM backend in 89 s with an 8 GB heap, which the :test alias
  sets (-Xmx8g; the JVM's default is a quarter of the machine's memory);
  it fails at 6 GB, a gap to the 2.9 GB not yet explained. MLX takes 19 s on its CPU in float64 and 6 s on
  Metal; Python memo about 28 s.
- **Constants** (2026-10-24, memoclj-c0xj). What does not depend on a call
  is computed at a memo's first call on each backend, and kept: literals,
  domains, ones, and what memo's operators compute from them alone.
  - They are kept by backend, since an array belongs to one, and the JVM's
    and MLX's meet in one process.
  - Never a parameter, an inline form, a memo call or a host function. A
    host function's values can change with nothing in the memo changed (an
    atom it reads, a helper it calls redefined), and a stale table would be
    a silent wrong number.
  - Constants made under a tracer (the batch tracer, a gradient, MLX's
    compile) serve later calls outside it (`memo.lower-test`).
  - §6.4 has the measurements.
- **Speed against Python memo (to measure, not promise):** benchmarks
  against Python memo on JAX and against memo-clj, on small recursive models
  (RSA depth 5, Sally-Anne) and dense ones (Penelope).

## 6. The GFI bridge (M2)

*Proposed 2026-10-22 and approved by the owner the same day (§13.1).* It
replaces the sketch of 2026-10-20 (`memo/dist` and `memo/fn`). GenGDT code
is written against it.

*Built 2026-10-22* (`src/memo/gen.clj`, `test/memo/gen_test.clj`, the
Handbook's "Minds in gen.clj models"), as specified below. The whole suite
passes on the JVM, Metal and MLX's CPU in float64 on the Mac, and on MLX's
CPU in float64 and CUDA on fenrir; since gen.clj 73f22a0, MLX's CPU in
float32 too, on both. Each of eleven planted bugs is caught. gen.clj's GFI laws run over two models that trace memo
distributions: the test puts the `test/` of the gen.clj in use (the pin's
checkout, or the working copy under `:dev`) on the classpath, so the laws
are that gen.clj's own. The first client is done (§6.5). Still to do: the
speed research (§6.4, question 3).

### 6.1 What already works, and what is missing (measured 2026-10-22, gen.clj 34e9a44)

- **A memo is already a function from parameters to a gen.array table.**
  gen.clj's gradients flow through it. GenGDT's stand-in,
  `(d/categorical (apply choose args))`, gives a β gradient from
  `choice-gradients` that agrees with finite differences (GenGDT
  notes/90).
- **Batched runs work.** A memo called under gen.array's batch backend with
  a batch of parameter values gives the batch of tables, tabulated host
  functions included. Vectorized ones too since 2026-10-24: their
  spot-check had sent `gen.batched` one particle at a time (memoclj-m2jw,
  §5).
- **What the bridge must add:**
  - **The stand-in's gradient is NaN whenever any cell of the table is 0,**
    whichever cell is observed. `d/categorical` takes the log of every cell,
    then picks one, and the log's gradient at 0 is 0/0. memo tables often
    have exact zeros, such as an option that never comes to mind. GenGDT's
    `choose` at q = 0 is the table [1 0 0]. Observing PASS, the stand-in's
    β gradient is NaN; logging only the observed cell gives the right
    gradient, 0. (gen.clj fixed this in bd6b8b4, genclj-rnbd: categorical
    and the other families now pick before the log.)
  - **Draws are indices,** not domain values.
  - **Nothing checks the table.** memo's silent all-zero rows (§2.4) are
    sampled from and scored.
  - **A NaN gradient inside the mind passes silently.** On slice 3's RSA
    speaker S1, `array/grad` gives NaN.
- **Not the bridge's to fix, but its first client meets it:** only
  `^:memo/vectorized` host functions are differentiated (§5). GenGDT's `eu`
  is tabulated, so `choose` differentiates in β and q but not in the goal
  value v ("Cannot differentiate through the host function eu"). To fit v,
  `eu` must be written with gen.array's operators and marked.

### 6.2 The API: `memo.gen`

```clojure
(require '[memo.gen :as mg])

(def choice (mg/dist choose))     ; a distribution constructor, as d/categorical is

(gen [...]
  (trace :o (choice beta q v s m-fix m-flee)))   ; a draw is a value of Opt
```

- **`(mg/dist model)` or `(mg/dist model opts)` gives a gen.clj
  distribution constructor.** Called with the memo's parameters, as in a
  memo call (positional, defaults, a trailing map by name), it gives a
  distribution, as `(d/categorical p)` does.
  - **Construction checks:** the model is a memo, `:given` names its axes,
    and `:return` is in range. It is checked once, when the constructor is
    made.
  - **Draws are domain values:**
    - an enum member's value;
    - an element of a vector domain (20 of `[10 20 30]`);
    - over several sampled axes, the vector of their values, in the order
      the axes are declared.
  - **Sampling** is categorical on the table, over its flattened cells when
    several axes are sampled.
  - **log-prob** is the log of the drawn cell: picked first, then logged,
    so another cell's zero gives no NaN.
    - A domain value of probability 0 scores −Inf.
    - A value outside the domain is a loud error. A domain value is not its
      index unless the domain is 0…n−1.
  - **The table is checked at every call.** Each distribution in it (the
    whole table, or one row per value of the given axes) must have no NaN
    and no negative cell, and must sum to 1 within §6.4's tolerance.
    Otherwise it is a loud error naming the model and the row. The hint
    covers memo's silent zeros: an impossible observation, or all-zero
    weights.
  - **Gradients.** log-prob is differentiable in every array parameter,
    under §5's rule that only vectorized host functions are differentiated.
  - **The NaN guard** (§13.10). Each parameter enters the memo through an
    identity whose gradient rule checks the gradient arriving for that
    parameter. A NaN is a loud error naming the model and the parameter.
    - Measured on S1: the guard throws "NaN gradient with respect to
      alpha" where `array/grad` gives NaN.
    - It passes S1-pow's finite gradient unchanged (−0.512 both ways).
    - It costs nothing in the forward pass: it is one `array/custom-vjp`
      per parameter.
- **Options:**
  - `:given [axis …]` lists the axes given as values, not sampled. The
    constructor then takes their values first, as a vector, as a memo call
    does. `((mg/dist L1 {:given '[u]}) [u] alpha)` is the listener's
    distribution over r for utterance u.
  - `:return i` picks the return that is the distribution, for a memo with
    several. They stack on a leading axis.
- **`(mg/deterministic model opts)` gives a `gen.custom/deterministic`
  generative function** returning the memo's table, or one return of it,
  through the same NaN guard. Use it to splice a mind at an address where a
  trace should show it. Otherwise call the model directly: it already is
  the plain function the old sketch's `memo/fn` described. A memo's
  parameters as trainable `:params` (gen.clj's learning) can come later, if
  a client needs them.
- **`mg/dist` traces nothing inside the mind.** A mind's choices are summed
  out exactly; that is memo. The scientist's trace holds only what the
  scientist observes or infers.

### 6.3 Laws (the tests)

- **Distribution checks over a set of models:**
  - The models:
    - GenGDT's `choose`, including a table with zero cells;
    - RSA's L1 and S1-pow;
    - a joint over two axes;
    - a conditional with a given axis;
    - a domain whose values are not indices.
  - The checks:
    - normalization over the support;
    - sample frequencies against the table, within five standard errors;
    - log-prob against the log of the table;
    - draws in the domain.
- **gen.clj's GFI laws over models that trace memo distributions.**
  `gen.laws` is under gen.clj's `test/`, not `src/`, so another project
  cannot depend on it. memo.clj's test adds the `test/` of the gen.clj in
  use to the classpath at load time (its `:deps/root` in the basis), so it
  needs nothing from gen.clj.
- **Gradients** against finite differences, and against the gradient GenGDT
  checked by hand.
- **The zero-cell gradient and the NaN guard,** each checked against a
  planted bug (CLAUDE.md rule 7).
- **Batched against one particle at a time:** `simulate` and `generate`
  under `gen.batched`, with a vectorized function of a batched parameter
  among them.
- **On every backend.**

### 6.4 The five questions (answers proposed 2026-10-22)

1. **Names.** A namespace, `memo.gen`, with `dist` and `deterministic`.
   - `memo/fn` would shadow `clojure.core/fn` wherever it was referred.
     Also, a memo already is the function.
   - `deterministic` is named after `gen.custom/deterministic`, which it
     builds on.
2. **Several returns.** `:return i` is required when a memo has more than
   one return. A memo with a single return needs nothing.
3. **Speed.** The owner, 2026-10-22: research caching and compiling when M2
   gets there (bean memoclj-c0xj). The leads:
   - memo's `^:cache`, which remembers results by parameter values, but not
     under differentiation;
   - `array/compile`, one MLX graph per input shape (on the JVM it is the
     function itself);
   - `gen.custom/deterministic`, which keeps its trace when an update
     leaves its arguments unchanged;
   - HMC's cache of compiled steps.

   *Measured 2026-10-23, decided by the owner 2026-10-24* (memoclj-c0xj;
   `bench/memo/bench/agent.clj`, run by `clojure -M:bench -m
   memo.bench.agent`). The model is GenGDT's `fit-beta-v`: its mind `choose`
   is called once per trial, 150 trials.
   - **Where a call's time went.** A call took 0.077 ms on the JVM, and
     0.109 ms with its gradient. The same table written by hand in gen.array
     takes 0.0087 and 0.030 ms. A third of a call went to tabulating the
     host function `available`, most of it flattening its arguments'
     elements. A ninth went to `eu`'s spot-check.
   - **What is built.** Constants are computed once per backend (§5), and
     host functions are tabulated faster. A call now takes 0.051 ms, and
     0.081 ms with its gradient.
   - **Host functions are not folded.** Folding them too would give 0.039
     ms, but one could read an atom or call a redefined helper (§5). It
     would need a purity mark, if a model ever needs it.
   - **No help from `^:cache`:** each trial's arguments differ, and β and v
     are arrays.
   - **No help from `gen.custom/deterministic`'s unchanged update:** HMC
     moves β and v at every step.
   - **`array/compile` on MLX** helps a memo whose host functions are
     vectorized: 150 trials go from 35 to 3.5 ms. A memo that tabulates
     reads host numbers, which a compiled function cannot.
   - **The JVM is about 10× faster than MLX Metal for this model.** An HMC
     step takes 248 ms on the JVM against 2679 ms on Metal: the arrays are
     tiny, and the model reads values on the host.
   - **The lever is batching the trials.** One hand-written call on the 150
     trials' inputs takes 0.08 ms with its gradient. 150 calls of the memo
     take 12.4 ms, against 16.4 ms before the constants.
     - It needs batched parameters for a memo, by the batch tracer or by a
       leading axis in the lowering.
     - It needs a batched fitting model in GenGDT.
     - GenGDT's research (gengdt-nvoo) decides whether its fits need it.
4. **Batched gradients.** gen.array's batch tracer and its tape still do
   not combine. Measured: "arrays from different backends (:batch and
   :jvm)". gen.clj's M5 did not change that. M2 does not need them:
   - gen.clj's gradient-based inference (HMC, MALA, MAP, VI) runs one chain
     or sample at a time, and the bridge's gradients work there;
   - batched inference (importance sampling, SMC) takes no gradients, and
     batched memo calls work there.

   Batched gradients, for many chains at once, would be gen.clj's work, if
   wanted.

   *2026-10-23:* gen.clj's wave 2 (genclj-7dti, not yet at memo.clj's pin)
   takes gradients inside a batched run, one per particle. The gradient of
   a function with a batched run inside stays an error. That case is
   genclj-hwk1, in gen.clj's backlog at low priority. Batched parameters
   made by the batch tracer would need it: the gradient is taken outside
   the batched run, by HMC on the fitting model's trace.
5. **Normalization tolerance.** |Σ − 1| ≤ 1e-6 + n·ε, where n is the number
   of cells in the row and ε is the dtype's machine epsilon.
   - The 1e-6 was `d/categorical`'s tolerance when this was decided. n·ε
     widens it for long rows, as the bound on rounding in a sequential sum
     does.
   - Since gen.clj bd6b8b4, categorical's tolerance is
     max(1e-6, 4·log2(n)·ε) (genclj-ggdl). It is never wider than this
     one.
   - It accepts float32's rounding in a 10,001-cell row (a bound of 1.2e-3),
     whatever order a backend sums in.
   - It rejects what the check is for: zero rows (Σ = 0), NaN, and tables
     off by more than rounding.

### 6.5 The first client

GenGDT's `fit-beta-v` infers the synthetic agent's goal value and
temperature. That is the scientist's inference (kind 2 of GenGDT's DESIGN
§2.3), with the agent's policy, `choose`, as an `mg/dist`. In memo's own
sense an observer is an agent frame that conditions, which is kind 3. It
runs by HMC, or on a grid with gen.clj's enumerative inference, once `eu`
is vectorized (§6.1).

*Done 2026-10-22* (GenGDT 1e99526, where the var was named `observer`;
renamed `gengdt.memo-agent/fit-beta-v` in GenGDT 0173e7e, the owner's
decision of 2026-10-23):
- **Wiring.** GenGDT pinned memo.clj b7ef474 (4045276 since GenGDT
  9d4c39a). The stand-in is replaced by `mg/dist`. `eu` and the affect
  report's mean are in gen.array's operators, so v differentiates as β
  does.
- **Identifiability.** The agent's choices alone identify only β·v,
  because EU is v times an expected satisfaction. `fit-beta-v` therefore
  uses every observation of a life: choices, outcomes and affect reports.
  The reports' mean is v times a difference of satisfactions, and that
  separates the two, but only in the toy, whose reports are in value
  units. A real rating's scale factor trades off against v, as β does
  (GenGDT's DESIGN §15.1).
- **Results at 150 cycles.**
  - β's and v's gradients agree with finite differences.
  - The grid posterior equals the hand posterior to 1e-9.
  - v is identified: 2.00 ± 0.24, the true value 2.0.
  - β is covered, but with a long tail: 6.2 ± 2.1, the true value 4.0.
  - 200 HMC steps agree with the grid on v.
- **Tests.** Four planted bugs in GenGDT's code are each caught.

## 7. Bounded minds (M3)

*Built 2026-10-24, before 1.0 (the owner's decision of that day), and
**experimental**: the `:memo/budget` call option, `memo.bounded` and
`memo.gen/estimated` stay outside what 1.0 freezes. §7.1–§7.3 keep the
owner's decisions of 2026-10-24 (§13.2, §13.3, §13.5); what M3 decided
while it was built was marked (proposed, built) for the owner's review, and
the owner kept every choice on 2026-10-24 (§13.11): those marks now read
(built; decided 2026-10-24).
The code: `src/memo/bounded.clj`, the analysis's `sample-ll` and
`eval-choose` (`memo.analyze`), three IR instructions (§7.5). The tests:
`test/memo/bounded_test.clj` (§9). The Handbook's "Bounded minds" part is
its reference.*

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
  and illegal under both lowerings, with one M3 restriction (§7.4): a
  bounded estimate drives only its agent's own choice.
- **Finite in M3** (decided 2026-10-24, §13.5). Bounded agents sample from
  finite domains, as memo's agents reason over them. Continuous variables
  are discretized.
  - gen.clj distributions inside an agent's model would be a later
    extension (principle 1), once a client needs one.
  - They would always lower by Monte Carlo, since their samples cannot be
    enumerated.

**What M3 built (built; decided 2026-10-24):**

- **Plug-in on sampled beliefs.** Every query a bounded agent evaluates that
  is an expectation (`E`, `Pr`, `Var`, `H`, `KL`, the posterior shortcut,
  and the expectation inside `EU` and `Predict`) is computed with the same
  formula on an empirical belief, k samples, in place of the agent's
  belief. A query whose value the agent already knows (memo's redundant
  expectation) is not one, and stays exact.
- **Only what the query depends on is sampled:** the frame's uncertain dims
  in the value's deps (the queried choices, for the posterior, `H` and
  `KL`), the others summed out first. In distribution that is the same as
  sampling everything and reading the marginal, and much smaller.
- **The agent is every frame it owns.** A frame's owner is the agent it was
  made for; a snapshot, an imagined world and a future self (EU's and
  Predict's) keep their agent's. So a budget for alice bounds alice's
  frame wherever it is modeled (`[alice]`, `[bob alice]`), her imagined
  worlds, her future selves, and her snapshots; a snapshot's alias cannot
  be given a budget of its own.
- **Fresh samples for each query,** marginalized at the innermost choice of
  the same agent that encloses the query (the "consumer"). A query inside a
  nested choice of that agent (its future self's choice, replayed by EU) is
  marginalized there.
- **`EU` and `Predict` are two queries,** as memo computes them: the goal is
  the future self's expectation over what it will still not know, and `EU`
  the present self's expectation of that over what will happen. Each is
  estimated from samples of its own, so a goal over an outcome and a
  variable the agent never observes (memo's competitor: `score >
  bob.score`) averages over every pair of k outcome samples (per option)
  and k samples of the unobserved variable (shared), not over k joint
  samples. That follows from bounding every query the agent evaluates; the
  acceptance pass found it undocumented, and the Handbook now says so. A
  joint-futures algorithm (sample k complete futures) would need EU's two
  expectations fused, and is not built.
- **Sharing (built; decided 2026-10-24; a claim about minds, flagged by a side agent
  and filed for GenGDT as gengdt-lu5i).** Within one query the samples are
  shared across the values the agent knows, except along the dims its
  reasoning *mixes* before the choice, where its belief varies: there each
  value has samples of its own.
  - The mixed dims are the options of the consuming choice, and the dims an
    enclosing query averages over.
  - "Its belief varies" is structural: the frame's `:ll-dims`. The lowering
    checks it (a belief varying along a mixed dim without samples of its
    own is an internal error, `check-mixing`). For it to be precise, a
    query's value depends, under a budget, on the known dims its beliefs
    vary along and the known choices its expression reads (`query-deps`),
    not on every choice the frame knows, which made a later observation's
    guess give options samples of their own (acceptance pass 3).
  - So options share samples when the agent's model of what follows does not
    depend on the option (alice judges both bets against one imagined score:
    Vul, Goodman, Griffiths & Tenenbaum's "one and done", 2014), and each
    option has samples of its own when it does (outcomes imagined per
    option). A future self whose belief depends on its outcome samples
    afresh for each outcome its present self averages over.
  - `:per-option true` gives every option samples of its own even when the
    belief does not depend on it: a different hypothesis, with different
    predictions. Which one GDT's imagined outcomes follow is GenGDT's to
    decide (gengdt-lu5i).

### 7.2 Exact over the agent's own samples first (decided 2026-10-24; from a side note, 2026-10-20)

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

**The rule** (the owner, 2026-10-24, §13.3).
- **Exact by default.** The bounded lowering computes the agent's sampling
  exactly when the enumerated space is below a size limit. The space is
  over sufficient statistics where the analysis can find them, and raw
  samples otherwise. It reports the size it chose.
- **Monte Carlo only when declared.** It never switches by itself. A space
  above the limit, or a sampler it cannot enumerate (an adaptive budget),
  is a loud error naming the size. The way out is to declare Monte Carlo
  for that agent in its budget (`:monte-carlo`, §7.3).
  - GenGDT fits budgets. An automatic switch would turn an exact density
    into a noisy estimate partway through a fit, and HMC on that is
    silently wrong (rule 4).
- **The limit:** 10^6 enumerated states per agent query, overridable at
  the call. Examples:
  - k = 8 samples over 4 outcomes: 165 states;
  - 2 options with binary outcomes: 81;
  - 3 options over 4 outcomes: about 4.5 million, so it needs an override
    or Monte Carlo.

  The limit is a computational convenience, never a psychological claim
  (GenGDT DESIGN §2).
- **With Monte Carlo declared,** variance control is part of the fitting
  recipe: several inner draws per trial, correlated pseudo-marginal moves,
  SMC over trials. The estimator's log-variance is reported per
  participant.
- **Both lowerings must agree.** The Monte Carlo lowering's average
  converges to the exact-over-samples table (a law, §9). Declaring Monte
  Carlo below the limit gives the comparison.

**As built (built; decided 2026-10-24):**
- **The sufficient statistic is the count vector** over the joint cells of
  the sampled dims (m = the product of their sizes): an estimate is a
  function of the counts whatever the query, since every query is a plug-in
  on the empirical belief. n sets of samples (one per cell of the dims with
  samples of their own) enumerate S^n states, S = C(k+m−1, m−1).
- **The limit is checked twice,** both when the analysis compiles the budget:
  on each query's states, and on the product at each choice (all the
  queries and options it marginalizes). Per agent (`:limit`), or for the
  call (`:memo/limit`); the error names the size, the agent and the query,
  and its hint lists the ways out. "Per agent query" is read as one choice
  of the agent: what one decision enumerates.
- **The report:** `(memo.bounded/report model budget)` lists each choice
  that marginalizes samples, in the memo and the memos it calls (`:memo`),
  with its states, limit, and sources (the query, k, the categories m, one
  more under rejection, the sets n). A choice above its limit is reported
  (`:over-limit true`) instead of thrown: the analysis has a report-only
  mode. Under Monte Carlo, `:states` is the number of draws.
  `memo.bounded/program` prints the bounded IR.
- **Which limit:** an agent's own `:limit`, else the call's `:memo/limit`,
  else 10^6 (the agent's wins, higher or lower). It does not apply under
  Monte Carlo.
- **Caches.** The count vectors and an exact budget's constant arrays are
  kept for later calls, bounded by size (about 4·10^6 numbers on the host,
  2·10^7 in arrays per backend).
- **The multinomial probabilities are computed in log space,** with their
  gradient written out (`array/custom-vjp`): an outcome of probability 0
  keeps its exact first derivative, which log space would make NaN, and
  which pruning it would lose. A large k (m = 2 allows k near 10^6) then
  neither overflows the coefficient nor underflows the powers.
- **Zero mass.** A belief of no mass (after an impossible observation) gives
  estimates of 0, as the exact query's, whatever the samples. A choice that
  is impossible for some draws (options considered whose weights are all 0,
  a `:to-be` whose estimate is no option, a `:such-that` no option meets)
  loses their mass, as an impossible choice does in memo (§13.9), and the
  table sums to less than 1; `mg/dist` refuses it.
- **A belief must be a probability.** memo computes exactly with negative
  weights (a negative `:wpp` or `:observes-event`), but a sample needs
  probabilities: a bounded agent's belief with a negative cell, or a
  rejection's acceptance probability below 0 or above 1, is a loud error
  (acceptance pass 1 found tables of 2, 8, 32 under rejection, and above 1
  under one-and-done, before).
- **Not built:** pruning the count vectors of probability 0, which would
  shrink GenGDT's case (3 options, 4 outcome levels, 2 of them possible)
  from 4.5 million to thousands, exactly in values and first derivatives.
  It waits for a client that needs it; meanwhile, a sampled variable at its
  natural size (whether a level was gained, not the level) keeps m small.

### 7.3 The budget (decided 2026-10-24) and the first algorithms (built)

- **The budget is a call-time argument, keyed by agent**, so the model text
  does not change, e.g. `{:budget {alice {:samples 8 :algorithm
  :one-and-done}}}` (the owner, 2026-10-24, §13.2).
  - The budget is a host value: it fixes the tables' shapes (the number of
    samples), so it is not differentiated.
  - A budget that varies is a call with other arguments: `budget = f(goal
    value)`, or a fitted depth.
  - Moors' operating conditions belong to the agent and the moment.
  - The budget can itself be a function of the outer model's state, or a
    random variable. Urgency narrowing computation is then expressible in
    GenGDT as `budget = f(goal value)`.
- **As built (built; decided 2026-10-24):**
  - **The key is namespaced,** `:memo/budget`, in the call's trailing map,
    so that a parameter named `budget` never collides: `(choose 2.0
    {:memo/budget {'alice {:samples 8}}})`. The call's other options are
    `:memo/limit` and `:memo/key`. `mg/dist` and `mg/deterministic` pass
    them through as they pass any trailing map.
  - **An agent** is a symbol or a keyword; `self`, the observer and a
    snapshot's alias cannot be given one; an agent that neither the memo
    nor a memo it calls has is an error.
  - **Each part of a budget bounds something** (memoclj-wxuc, 2026-10-24),
    in the memo or a memo it calls: `:samples` a query the agent evaluates
    about what it is uncertain of, `:options` a choice it can narrow
    (`:options k` at or above the choice's options is its limit, and
    passes). Otherwise a loud error, checked once per budget with the
    agents (and by `bounded/report`). First built as a silent no-op, the
    exact table: GenGDT's `memo-agent/choose`, whose weights come from a
    host function, would have fitted k to a flat likelihood.
  - **A spec:** `:samples k`, `:algorithm` (`:one-and-done`, the default,
    or `:rejection`), `:per-option`, `:options k`, `:monte-carlo` with
    `:draws R`, `:limit n`. Unknown keys and values of the wrong kind are
    errors.
  - **A budget reaches the memos called** (as memos, or as host functions):
    an agent is bounded wherever its name appears in the call, so a
    recursive planner is bounded at every level. Each memo call gets a key
    of its own (the call's folded with the instruction's position).
  - **Programs are compiled per budget**, on the first call with it, and
    cached; a `^:cache` memo caches by budget too (never under Monte Carlo).
  - **The exact program is compiled without a budget,** so its IR is the
    same as before M3, instruction for instruction.
  - **A range of budgets** (gengdt-3xne: a budget summed out, or trials whose
    budgets differ) needs no construct of its own. Every budget gives a
    table of the memo's own shape, so the tables for k in a range are a
    stack of calls, `(array/stack (mapv #(f x {:memo/budget {'a {:samples
    %}}}) ks))`, each budget compiled once. A budget computed from a fitted
    value is read on the host, which `array/compile` and gen.batched's
    batching cannot do (GenGDT's to weigh, gengdt-3xne).
- **First algorithms (proposed 2026-10-20; built and decided 2026-10-24):**
  - **Sample the agent's belief and average** (`:one-and-done`): samples
    from its current belief, after what it observed (forward sampling where
    nothing is observed; Vul et al. assume posterior samples). This also
    covers "a fixed number of sampled futures for `Predict`/`EU`": their
    expectation is a query in the agent's imagined future, and k = 1 is
    "one and done" literally.
  - **Where the agent observes:** `:rejection`, the prior sampled and the
    samples consistent with the observations kept. An `:observes-event`
    weight is the probability of keeping a sample (it must not exceed 1, a
    loud error otherwise); with none kept, the agent's beliefs have no mass.
    Exactly: count vectors over m + 1 categories, the last "rejected" with
    probability 1 − Z, Z the probability of the observations (the product
    of their normalizers, tracked per frame as `:evidence` when a budget is
    given, and moved with the beliefs by an observed choice). Where the
    frame observed nothing, nothing is rejected, and the m categories of
    one-and-done are used. Each `:observes-event` weight is checked to be in
    [0, 1] (the frame keeps them, `:event-weights`), not only Z. What the agent
    `:knows` is not an observation (it knows the value from the start), so
    no sample is rejected for it, unlike `:knows`'s expanded form with an
    `:observes`.
    - Self-normalized importance sampling coincides with it for hard
      evidence (`:observes`, `:observes-that`). For soft evidence it
      differs (weights, not acceptance), and is **not built**: it needs the
      prior and the weights over the sampled cells separately.
  - **A few options considered** (`:options k`, Moors' accessibility): each
    of the agent's `:chooses` (by weight, `:to-maximize`, `:to-minimize` or
    `:uniformly`; not a `:given`/`:draws`/`:assigned`/`:guesses`, nor a
    `:to-be` or `:such-that`) considers k of its options, drawn uniformly
    without replacement, and chooses among those: C(n, k) subsets, each its
    own draw. Accessibility weights (GenGDT's options coming to mind with
    probability q) stay in the model text, as GenGDT writes them now.

  Each is a cognitive hypothesis, not only a technique.
- **Open research questions:**
  - which algorithms are psychologically meaningful;
  - how bounded observation (rejection vs weighting) interacts with
    snapshots and continuations. *As built:* a snapshot copies its agent's
    evidence with its beliefs, and observations after it do not reach it; a
    future self replays later observations as memo does (a guess, then an
    observation of it), so with `:rejection` it keeps the samples consistent
    with what it will observe. The guess is exact: the imagined observation
    is part of the imagined future, which the agent's query over that future
    samples. (A sampled guess, as first built, sampled it twice; with
    `:one-and-done` that left the table unchanged, but it cost states, and by
    Monte Carlo it biased the outer query: acceptance pass 2);
  - what cost means (op counts vs samples), and how it relates to memo's
    `cost @`.
- **Dependencies:** gen.clj's encapsulated randomness (genclj-1ogo) and its
  law set. *At the pin (3022944):* `gen.custom/estimated`, which
  `memo.gen/estimated` builds on, and `gen.laws/encapsulated`, which the
  tests run.
- **First GenGDT client:** the agent that imagines k outcomes before it
  chooses, with the exact agent (k → ∞) as its limit. GenGDT's to build:
  gengdt-lu5i, starting when GenGDT pins the commit that closes M3.

### 7.4 Where a bounded estimate can go (built; decided 2026-10-24)

- **The agent's own choice.** The samples are marginalized there, so an
  estimate can drive that choice's `:wpp`, `:to-maximize` or `:to-be`.
  The marginalization multiplies the choice's normalized weights (per draw)
  by the draws' probabilities and sums over the draws' dims, before the
  parent frame's beliefs are multiplied: so an observer conditioning later
  on the choice normalizes over the samples too (the posterior over the
  agent's type from its choice is P(type) P_bounded(choice | type), as the
  tests check).
- **Anywhere else is a loud error in M3:** an estimate returned, read by
  another agent (another agent cannot know alice's samples; a frame that
  models her would be uncertain of them), or weighting an observation. The
  error's hint gives the way to model a reported estimate: a choice of the
  agent's, `(alice :chooses [r R] :to-be ...)`.
- **Later, if a client needs it:** an estimate read by an enclosing query of
  a frame that models the agent, which would average over the samples with
  its own uncertainty (linear, so well defined), and soft-evidence
  weighting.

### 7.5 The lowering (built; decided 2026-10-24)

- **The analysis takes the budget** (`:bounded`, normalized, and the call's
  `:limit`). At a bounded agent's query, `sample-ll` emits, instead of
  reading the frame's beliefs:
  - the belief summed to the sampled dims;
  - `{:i :sample-belief}`: the empirical belief of each draw, on a new dim of
    draws ω (allocated as any choice's dim, so the IR stays single
    assignment over dims);
  - `{:i :sample-weight}`: each draw's probability (exact over samples).

  The query's formula then reads the empirical belief. The draw is pending
  until `eval-choose` marginalizes it. With `:options`, `{:i :subsets}`
  masks the options not considered, before the softmax, the argmax or the
  normalization.
- **`memo.bounded` lowers the three instructions:** layout (the dims moved
  to the front, the cells flattened, and back), the count vectors and their
  probabilities (§7.2), the empirical beliefs, the subsets.
- **Monte Carlo** (`:monte-carlo true`, `:draws R`) uses the same IR: the
  draws' dim holds R random draws, shared by all the sources one choice
  marginalizes (each drawn independently, from its own key), averaged with
  weight 1/R. Only a choice the observer models directly, outside any
  choice's weights, is drawn: one inside another agent's mind, or inside a
  choice's weights (the future self's choice EU replays), stays exact over
  its samples, since the reasoning above samples from (or weighs by) its
  probabilities, and an estimate there enters nonlinearly, a bias
  (acceptance passes 2 and 3). Its size error then says why Monte Carlo was
  not used. Draws come from the call's `:memo/key` folded with the
  instruction's position, so a key reproduces its table; they are never
  folded into constants (memo.lower's `constant`). The samples carry no
  gradient through the belief they are drawn from: a belief being
  differentiated is a loud error, while values the agent judges may be.
- **`memo.gen/estimated`** is the generative function of §7.1's scientist's
  view: `gen.custom/estimated`, one choice at `:value` (as `mg/dist`'s
  draws), its density the table's cell computed with the operation's key
  (log ξ), its joint sampler a draw from that key's table (the thesis's
  importance-sampler construction: the value from the averaged table, the
  draws given it). `mg/dist` refuses a Monte Carlo budget. A model that
  splices it should pass the same budget object at each run: gen.clj sees a
  splice's argument unchanged only when it is identical, and only then keeps
  ω on an identity update.
- **What is unbiased.** A Monte Carlo table is an unbiased estimate of the
  agent's choice probabilities, and of anything linear in them (a memo
  returning `(Pr (= alice.a a))`, marginals over other variables). What the
  table computes from them nonlinearly is not: an observer that conditions
  after the choice (its posterior is a ratio of estimates), or another
  agent's softmax over them. `mg/estimated`'s density is valid for
  pseudo-marginal MCMC only in the first case; nothing checks it (a
  conservative check would refuse every model with a later observation).
- **The log-estimate variance** per participant:
  `memo.gen/log-estimate-variance` over a participant's trials.

## 8. Surface syntax (decided, 2026-10-20)

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
  error. This diverges from Python memo 1.3.3 (principle 1), which accepts
  `self: chooses(x …)`; its choice takes the observer's own key
  `(self, x)`, and the answer comes out right by accident. memo-clj
  accepted it and returned Pr = 1.
- **memo's bare assertions become compile errors with hints:** `Predict` or
  `EU` at the top level, and a goal named twice in `wants` (§2.4).
- **Statements and expressions** are those of Python memo 1.3.3, the same
  as 1.3.2's. memo-clj's README has the table as of 1.2.26; 1.3.2 adds
  `time @`, `space @` and `size @`, and rejects `is` in expressions. It is
  to be re-checked against the Handbook. All are built (M1, slices 1-6)
  except cost reflection (`time @`, `space @`, `size @`), which is M4.
- **Options and parameters** (slice 6): metadata on the memo's name,
  `^{:cache true}` (Python's `cache=True`: calls memoized by their host
  number arguments, never for values being differentiated, and every cache
  dropped when any memo is (re)defined, since a callee may have changed) and
  `^{:variant :child}`; a parameter marked `^:array` (Python's `p: ...`)
  takes a Clojure vector or a gen.array array, passed whole to host
  functions and memo calls.
- **The inline escape hatch** (2026-10-20): `#{form}` is Clojure, a
  function of the memo's parameters (Python memo's `{...}`). An enum member
  in it, `#{Action.WAIT}`, is its value, as in memo expressions. An
  `agent.choice` there is a compile error, the Handbook's rule that braces
  cannot reference memo constructs (Python memo raises a `NameError` when
  the memo is called).
- **A malformed `observes`** is Python memo's "Unknown statement syntax",
  with a hint saying what is wrong.
- **Product domains** (built for the demos, 2026-10-20): `(defdomain S :x 3
  :y 4)` is Python memo's `S = domain(x=3, y=4)`, members 0 .. 11 packed as
  Python packs them (the first field most significant). `(S :x s)` reads a
  field, on a number or an array (so in a `^:memo/vectorized` function);
  `(S {:x 1 :y 2})` packs; `(memo.domains/fields S s)` is Python's
  `S._tuple(s)`. Five demos use them.
- **`Bool` is predefined** (2026-10-20; for the owner's review): every
  defmemo can name `Bool` (`NO`, `YES`) without defining it, as every
  Python memo model can (its generated code imports `memo.lib`'s `Bool`;
  demo-lemons relies on it). A namespace's own `Bool` comes first, by
  Clojure's scoping, where Python memo ignores the user's `Bool` inside a
  model: only labels can differ.
- **Cost reflection's syntax** (2026-10-20; built in M4): `(cost @(f ...))`,
  which the reader reads as `(cost (clojure.core/deref (f ...)))`, and
  likewise `time`, `space` and `size`, as Python memo's `cost @ f(...)`.
  Only that shape is reserved: any other `(cost ...)` is a host function
  call, as `cost(...)` is in Python memo (demo-trucks defines one).

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
  accepted silently. (Since slice 4, a fixture's `:ledger` entries record
  Python memo's value and the correct one, from a twin that Python memo
  computes correctly, and the tests require the correct one.)
- **Comparing tables.** memo returns size-1 axes where a return does not
  depend on an axis (§2.4), so results are broadcast to their declared
  shape before comparison.
- **Differential fuzzing.**
  - Random well-formed programs from a grammar-based generator run in
    memo.clj and, through the transpiler, in Python memo, and must agree.
  - Mutated programs must be rejected by both.
  - This is the strongest check available, and the reason the transpiler is
    test tooling from M1.
  - *Built 2026-10-20* (`memo.fuzz`, `memo.pygen`, `memo.python`, all under
    `test/`). The transpiler is checked first by round trip: the corpus's
    twins, transpiled, reproduce the fixtures in Python memo (327 calls).
    The generator tracks scope (frames, knowledge, minds two deep) over
    chooses (every spec, several choices at once, calls of a model),
    knows, thinks, observes, observes-that/-event, wants and EU,
    forgets-about and snapshots (followed by observations since 1.3.3);
    returns over E, Pr (the posterior shortcut), Var, H, KL, `as` and
    imagine. Agreement: values within 1e-4 relative (float32 accumulation
    over random tables), or both reject with a memo error (memo.clj may
    not crash), messages compared.
    Findings are kept in `test/fixtures/fuzz.edn` and replayed without
    Python. First campaigns: 2,300 programs, one real disagreement (Var
    under zero mass, above).
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

  *Built 2026-10-24* (`test/memo/bounded_test.clj`):
  - **Brute force:** the tests enumerate the raw sample sequences on the
    host (m^k of them, per option where options have their own) and compare
    every model to within 1e-9: shared and per-option samples, EU per
    option, argmax with ties, a query inside a query (a future self sampling
    afresh per outcome), one-and-done and rejection after `:observes-that`,
    soft evidence and an observed choice, options considered (alone and with
    samples), a bounded agent modeled by another, an observer's posterior
    over a bounded agent's type, and Var, H, the posterior shortcut and KL
    as plug-in estimates.
  - **The exact limit:** the error against the exact table falls with k
    (1 to 64), by more than 8 times.
  - **Monte Carlo:** its mean over 1500 keys within five standard errors of
    the exact-over-samples cell (shared, per option, rejection, options); a
    key reproduces its table, another does not; E[ξ] of `mg/estimated` is
    the exact-over-samples cell; its joint sampler satisfies the thesis's
    Eq. 4.4, E[[value = v] / ξ] = 1.
  - **gen.clj's laws:** `laws/encapsulated` (less the same-key score law,
    as for gen.clj's own estimated densities) over `mg/estimated` and a
    model splicing it.
  - **Gradients** against finite differences, in the belief sampled and in
    the temperature, at an outcome of probability 0 too.
  - **The budget's checks, the limit, the report, the loud errors** for an
    estimate outside its agent's choice, each message compared.
  - **Planted bugs** (CLAUDE.md rule 7): 30, each caught (the gradient at a
    zero outcome, the multinomial coefficient, sharing without the options'
    or the enclosing queries' own samples, `:per-option` ignored, rejection
    without evidence, with evidence not moved by an observed choice, with
    soft evidence always kept, or dividing by k, unweighted draws, the
    options mask, a `:given` narrowed, Monte Carlo draws folded into
    constants, subsets not random, one draw reused, no 1/R, the consumer
    check, `estimated`'s joint sampler weighing on another table, the agent
    check, and, after the acceptance pass, the negative-belief and
    negative-acceptance checks, the report's callees and its report-only
    mode; after pass 2, a sampled guess, nested choices drawn, the event
    weights unchecked, and the report's budget check; after pass 3, coarse
    query deps, Monte Carlo inside another's mind, and the guess sampled).
  - **Acceptance pass 1** (2026-10-24, one agent, the Handbook and
    docstrings alone, ~70 models against its own brute force;
    `tmp/acceptance-m3/report.md`): everything checked agreed, except two
    defects, fixed: Monte Carlo enumerated (and kept) the count vectors it
    never uses, so its memory grew as k² until a raw OutOfMemoryError (now
    k = 10^5 in 16 ms), and negative weights gave tables above 1 (now loud
    errors). Its doc gaps are closed in the Handbook: EU's two queries,
    lost mass, `:knows` under rejection, `:options` at or above the number
    of options, which limit wins, the size formula, the report.
  - **Acceptance pass 2** (2026-10-24, one agent, the same rules;
    `tmp/acceptance-m3-pass2/`): pass 1's fixes held (Monte Carlo at k =
    10^5, negative weights, the report). It found Monte Carlo biased (z =
    −6 to −9) where the imagined future holds an observation, the guess
    being a nested bounded choice whose estimate the outer query sampled
    from: the guess is now exact and nested choices are exact under Monte
    Carlo. Also fixed: a weight above 1 accepted while the total stayed
    below 1, a report of an unknown agent, the lowering's errors without
    the agent or line, a limit beyond a long refused, rejection's category
    counted with nothing observed, caches bounded by size; the rest were
    doc gaps, closed.
  - **Acceptance pass 3** (2026-10-24, focused on Monte Carlo;
    `tmp/acceptance-m3-pass3/`): about 70 Monte Carlo checks against exact
    over samples (up to 2·10^6 keys), the exact limit on seven models, and
    brute force for EU and Predict with later observations. Two defects,
    fixed: Monte Carlo on an agent inside another's mind was biased (z =
    −30, −47; now exact there), and a later observation gave EU's samples
    per option where nothing depended on the option (query deps refined).
    Doc gaps closed: ties among estimates are float equality, `estimated`'s
    split key, a budget that bounds nothing.
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
## 11. Dependencies (decided, 2026-10-20)

- **The core:** Clojure and gen.clj, pinned to a commit, the same policy as
  GenGDT's. Nothing else.
- **Tests:** test.check and the Cognitect test runner.
- **Oracle tooling (optional, documented):** Python 3, and a virtual
  environment with the pinned memo release and JAX, for goldens and
  differential fuzzing. Tests that need it skip loudly without it.

## 12. Conventions (decided, 2026-10-20)

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
   *Decided 2026-10-22:* `memo.gen` (`dist` and `deterministic`), with the
   answers of §6.4; approved by the owner.
2. **The budget's place** (§7.3): call-time and keyed by agent, as proposed?
   Or also declarable in the model text? *Decided 2026-10-24:* call-time
   only, keyed by agent. *Built 2026-10-24* as `:memo/budget` in the call's
   trailing map (namespaced, §7.3).
3. **Choosing the bounded lowering** (§7.2): automatic below a size limit and
   reported, as proposed, or always declared by the modeler? And what limit?
   *Decided 2026-10-24:* exact over samples by default; Monte Carlo only
   when declared, so a space above the limit is a loud error otherwise. The
   limit is 10^6 states per agent query, overridable at the call. *Built
   2026-10-24* (§7.2, §7.5).
4. **Upstream:** does the exact core track new memo releases, with the pin
   moved deliberately? Or freeze at 1.3.2 until M3? *(2026-10-21: the owner
   moved the pin to 1.3.3 for upstream's snapshot fix, bean memoclj-0s5l;
   whether to track every release stays open.)* *Decided 2026-10-24
   (memoclj-v4kh):* tracked, deliberately. Each Python memo release gets a
   bean: the pin moved, the fixtures regenerated, differences into §2.4 and
   the ledger (§9). New constructs come in minor releases of memo.clj,
   upstream bug fixes in patch releases, each called out in the release
   notes.
5. **Continuous variables in minds:** memo is finite. Is discretizing enough,
   or should bounded minds (M3) admit gen.clj distributions inside an agent's
   model? That would go beyond memo's semantics, so it would be an extension
   by principle 1. *Decided 2026-10-24:* finite in M3 (discretize); the
   extension later, once a client needs it (§7.1).
11. **M3's choices made while building it** (§7.1–§7.5, built
    2026-10-24). *Decided 2026-10-24: the owner kept every recommendation
    below (memoclj-3eap); the sharing rule stays GenGDT's to choose for
    its imagined outcomes, default or `:per-option` (gengdt-lu5i).*
    *Those that change what a model predicts:*
    - **Every query the agent evaluates is a plug-in on k samples,** in
      every frame it owns, its future selves included (§7.1). So where a
      goal depends on something the agent will observe and something it
      never will, `EU`'s two queries average over every pair of their
      samples, not over k joint futures; and a later choice of the agent's
      that `EU` replays is bounded too, averaged exactly over its own
      samples: the present self anticipates a bounded future self.
      Alternatives: an exact future self (only the present self samples),
      or k joint futures (not built). Neither of GenGDT's models meets the
      first case: `cope`'s agent observes nothing before its goal (one
      query), and the first client's goal is over the outcome, which it
      observes. In `cope`, `future` is an agent of its own, bounded only if
      the budget names it.
    - **Options share samples** unless the agent's belief depends on the
      option; `:per-option true` gives each its own (§7.1, gengdt-lu5i).
    - **Observations** (§7.3): `:one-and-done`, the default, samples the
      exact posterior; `:rejection` samples the prior and keeps a sample
      with probability equal to an observation's weight. Weighting samples
      instead (self-normalized importance sampling) is not built; for hard
      evidence it coincides with rejection.
    - **`:options k`** draws k-subsets uniformly, of `:chooses` by weight,
      `:to-maximize`, `:to-minimize` or `:uniformly` only (§7.3).
    - **The budget reaches the memos called,** by agent name (§7.3): agents
      of the same name in different memos cannot be bounded apart.

    *Those that only allow or refuse programs, changing no table:*
    - **An estimate drives only its agent's own choice** (§7.4), elsewhere
      a loud error; a reported estimate is written as a choice. Lifting it
      later adds programs.
    - **Each part of a budget bounds something,** else a loud error
      (§7.3, memoclj-wxuc).
    - **The call option is `:memo/budget`,** not the `:budget` of the
      owner's decision (§13.2), since a call's trailing map also passes
      parameters by name; `:memo/limit` and `:memo/key` likewise (§7.3).
    - **The limit** (§7.2): checked per query and per choice ("per agent
      query" read as one choice), the agent's `:limit` winning over the
      call's, not applied under Monte Carlo.
6. **The name:** check that "memo.clj" does not collide with an existing
   project ("gen.clj" collided with probcomp's Gen.clj). *Checked and
   decided 2026-10-24 (memoclj-idgc):* no project of that name on GitHub,
   Clojars or Maven Central (Clojars has no `memo` artifact or group;
   Maven Central's `memo` artifacts are unrelated Android libraries); the
   closest is the owner's earlier port, robert-johansson/memo-clj. The
   name stays memo.clj.
7. **Git and tracking:** a repository and a beans prefix (`memoclj-`?) from
   the start, as gen.clj has. *Decided 2026-10-20:* this repository (no
   remote yet), beans with the prefix `memoclj-`, set up as gen.clj's (the
   private remote of the beans repository is pending). *Decided 2026-10-24
   (memoclj-okbr):* published as a private GitHub repository,
   robert-johansson/memo.clj, as gen.clj, memo-clj and GenGDT are.
8. **Float64 against float32 parity** for the constructs where the dtype
   changes the answer (§2.4): run those parity tests in float32, keep them
   out of the fuzzer's grammar, or accept documented differences?
   *What the corpus and the fuzzer do now (2026-10-20, for review):* the
   fuzzer keeps them out of its grammar (`:to-maximize` only over integer
   values, `exp` bounded, EU through `:wpp (exp ...)`), and classes KL
   against a zero as that known difference when both values are past
   float32's largest number; the demo tests accept documented differences
   (persuasion's ties, checked to be exact ties; an absolute tolerance
   where results cancel). *Decided 2026-10-24 (memoclj-mths):* documented
   differences, as above, made the rule; the Handbook says where float64
   and JAX's float32 differ.
9. **Zero-mass distributions:** memo returns them silently (§2.4). Keep
   that for parity, warn, or make it an error outside a parity mode? This
   is where principle 1 (same semantics) meets principle 6 (no silent wrong
   numbers). *Decided 2026-10-24 (memoclj-xz4k):* parity. Tables stay
   memo's (an all-zero distribution, silently). `memo.gen/dist` checks
   that a table sums to 1 at every call, so a memo used as a distribution
   cannot pass lost mass into a fit. *Corrected the same day (a side
   agent's note):* that guard covers distributions only.
   `memo.gen/deterministic` returns values (an `E`, an `EU`), which have
   no sum to check, so lost mass there still reaches a fit as a silent
   number (0, or an `E` over mass below 1); and neither guard sees an
   agent's own beliefs losing their mass inside the model (its estimate
   0, feeding its choice), which is memo's semantics. A check for
   `deterministic` (the observer's mass at the return, per cell) is
   memoclj-wbhh's, for when a client fits through expected values.
10. **NaN gradients where the values are right** (§2.4). *Decided
    2026-10-20:* the values keep matching JAX, and the M2 bridge makes a
    NaN gradient a loud error, so fitting never fails silently. Repairing
    such gradients (a zero-probability term contributing 0, which departs
    from JAX) may come later as an opt-in.
