# Changelog

What changed in memo.clj that code built on it (GenGDT, say) must know
about. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). memo.clj is a git
dependency: a release is a tag, taken as `{:git/url … :git/tag "v1.0.1"
:git/sha …}` (the Handbook's Setting Up gives the coordinate).

- **What 1.x promises:** the Handbook's [Stability](HANDBOOK.md#stability)
  lists what 1.0 freezes. Bounded minds are experimental, outside it.
- **New Python memo releases are followed deliberately** (DESIGN §13.4):
  new constructs come in minor releases, fixes in patch releases.

## [Unreleased]

### Added
- **Cost reflection** (M4, DESIGN §14.1): `(cost @(f args...))` and its
  siblings `time @`, `space @` and `size @` inside models, and
  `memo.core/cost` on the host (Python memo's `return_cost=True`).
  - `size` is Python memo's exactly.
  - `time` and `space` count memo.clj's own run, a documented difference
    from Python memo's XLA numbers (the Handbook's Differences).
  - A host function may declare its cost per cell with `^{:memo/cost n}`.
  - A bounded agent's budget is counted.
- **`memo.core/rows`** (M4, DESIGN §14.2): a result as data, one map per
  cell, Python memo's `return_pandas` without a dataframe library.

### Changed
- `(+ x)` is kept as Python memo's unary plus, where memo.clj dropped it.
  On a static boolean it now gives Python's 1, where it gave `true`.

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
