# Changelog

What changed in memo.clj that code built on it (GenGDT, say) must know
about. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). memo.clj is a git
dependency: a release is a tag, taken as `{:git/url … :git/tag "v1.1.0"
:git/sha …}` (the Handbook's Setting Up gives the coordinate).

- **What 1.x promises:** the Handbook's [Stability](HANDBOOK.md#stability)
  lists what 1.0 freezes. Bounded minds are experimental, outside it.
- **New Python memo releases are followed deliberately** (DESIGN §13.4):
  new constructs come in minor releases, fixes in patch releases.

## [Unreleased]

### Changed

- "Observation of unknown choice" has a hint that names the fix when the
  value observed is an enum member (`(bob :observes alice.c :is
  Coin.TAILS)`, which memo reads as agent `Coin`'s choice `TAILS`) or a
  parameter: write `(bob :observes-that (= alice.c Coin.TAILS))`. The
  message is still Python memo's; other cases keep its hint.

## [1.1.0] - 2026-11-01

M4's parity half: the rest of Python memo's tooling, and memo.clj's
`work @`. Additions only, and fixes: nothing 1.0 froze changes. On gen.clj
v1.0.0.

### Added
- **Cost reflection** (M4, DESIGN §14.1): `(cost @(f args...))` and its
  siblings `time @`, `space @` and `size @` inside models, and
  `memo.core/cost` on the host (Python memo's `return_cost=True`).
  - `size` is Python memo's exactly.
  - `time` and `space` count memo.clj's own run, a documented difference
    from Python memo's XLA numbers (the Handbook's Differences).
  - A host function may declare its cost per cell with `^{:memo/cost n}`.
  - A bounded agent's budget is counted.
  - `(work @(f ...))` and `:work`, memo.clj's own: `time`, with each
    distinct call of a `^:cache` memo charged once, the work a run with its
    caches does. `time` charges them per call site, as Python memo does.
  - A count is kept until a memo, or a host function it reaches, changes.
    It is kept by the parameters it depends on, the others by their shape.
    The arguments are checked as a call's, a count kept or not.
- **`memo.core/rows`** (M4, DESIGN §14.2): a result as data, one map per
  cell, Python memo's `return_pandas` without a dataframe library.
- **`memo.core/*trace*`** (M4, DESIGN §14.3): a trace of every memo call,
  Python memo's `debug_trace`.
- **`^:memo/cache-table`** (M4, DESIGN §14.4): a pure, expensive host
  function's tables kept across calls, by its arguments' values; redefining
  it drops them.
- **`memo.pygen`** in `src`, experimental (M4, DESIGN §14.5): the
  transpiler, with `(memo.pygen/source vars)` giving a namespace's models,
  domains and helper functions as a Python memo module. It translates
  `cost @` and its siblings, and refuses `work @`, which Python memo lacks.
  It also refuses two names that become one Python name, names Python
  memo's models import for themselves (`Bool`, `pad`, `time`, ...), Python
  keywords, memo calls without axes (Python memo has no form for them), and
  a file edited since it was loaded. Files loaded by path (`clojure -M -i`)
  are read.

### Changed
- `(+ x)` is kept as Python memo's unary plus, where memo.clj dropped it.
  On a static boolean it now gives Python's 1, where it gave `true`.
- A `^:cache` memo's cache is keyed by its parameters' values, however they
  are passed: `(f 2)`, `(f {:n 2})` and a default are one entry, where they
  were separate entries (as in Python memo's `cache=True`). Results are
  unchanged.

### Fixed
- A `^:cache` memo and a `^:memo/cache-table` function took `-0.0` and
  `0.0` as the same argument, so a function that tells them apart (`1/x`,
  `atan2`) could read the other one's result.

## [1.0.1] - 2026-11-01

Documentation and licensing only: no change to code, results or the API.

### Added
- `LICENSE`: memo.clj is under the MIT License.
- `LICENSE-memo`: Python memo's MIT notice, for the parts of memo.clj
  derived from memo: the Handbook, the port of its test suite, the demo
  twins and their fixtures.
- `README.md`, with the logo (`memoclj.png`; its original,
  `memo_clj_logo.jpg`), and this changelog.

### Changed
- `deps.edn`: the `:dev` and `:mlx` aliases name a gen.clj checkout beside
  memo.clj (`../gen.clj`), not an absolute path.
- The oracle's setup is documented with memo-mcp, now public and under the
  MIT License (robert-johansson/memo-mcp).

## [1.0.0] - 2026-10-26

The first release: the exact core and the bridge, frozen.

### Added
- **The exact core.** Every statement and expression of Python memo 1.3.3
  except cost reflection, compiled to gen.array programs. It has parity
  against memo's test suite, its Handbook and its demos, compile errors
  with memo's messages, and differential fuzzing against Python memo.
  Where Python memo is wrong (DESIGN §2.4), memo.clj gives the correct
  value and records it.
- **`memo.gen`:**
  - `dist`, a memo as a gen.clj distribution, which checks that its table
    sums to 1;
  - `deterministic`, a memo as a deterministic generative function, which
    refuses lost mass in the queries its return reads (`:check-mass`).
- **Bounded minds** (experimental): `:memo/budget` in a call's trailing
  map, `memo.bounded`, `memo.gen/estimated`.
- **Pinned gen.clj** at its v1.0.0 release, with its laws module.
