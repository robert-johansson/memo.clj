# memo.clj — guide for agents

memo.clj is memo (Chandra et al., PACMPL 2025; github.com/kach/memo) on the
JVM, built on gen.clj. The design and its open questions are in `DESIGN.md`;
read it before changing architecture. Its first client is GenGDT
(`~/code/GenGDT`), whose `notes/81-memo-clj-m1-answers.md` answered the
questions asked before M1. `HANDBOOK.md` is the language reference in
memo.clj's syntax, executable (`test/memo/handbook_doc_test.clj` runs it).

## The oracle and the references

- **Python memo 1.3.2** (kach/memo `711e4cd`) is the oracle. It lives in
  `memo-mcp/memo` (memo-mcp is the owner's MCP server, gitignored here), with
  a venv in `memo-mcp/venv`. The memo-dsl MCP tools run, compile and explain
  Python memo programs.
- **It is not always right.** DESIGN.md §2.4 lists its bugs and the
  semantics memo.clj must copy; §9's upstream ledger keeps the corrected
  values. A disagreement is triaged into the ledger or fixed here, never
  accepted silently.
- **memo-clj** (`~/code/memo-clj`) is an earlier port: lift its ideas and code
  only after reviewing them against Python memo's source, and never test
  against its numbers.

## Work tracking: beans

All work is tracked in beans, prefix `memoclj-` (`.beans/`, a private git
repository synced by hooks; see `.beans/hooks/README.txt`).
- `beans list --ready`, `beans show <id>`; put a bean in-progress before
  starting, tick its checklist, close it with a `## Summary of Changes`.
- In beans 0.4.2, `--ready` does not inherit blocking from a blocked parent:
  put `--blocked-by` on leaf tasks.

## Running

```
clojure -X:test                       # every test, on the JVM backend
clojure -X:test:fast                  # without the ^:slow tests (three demos, minutes on the JVM,
                                      #   demo-empowerment needing the 8 GB heap :test sets; the Python round trip, a fuzz campaign)
clojure -X:test:mlx                   # on MLX (Metal, float32)
clojure -J-Dgen.mlx.device=cpu -J-Dgen.mlx.dtypes=64 -X:test:mlx   # MLX's CPU, float64
clojure -X:dev:test                   # against gen.clj's working copy
memo-mcp/venv/bin/python -I test/fixtures/generate.py   # regenerate the fixtures
MEMO_DEMOS=demo-rsa,demo-fib memo-mcp/venv/bin/python -I test/fixtures/generate.py   # only those demos
clojure -X:test:fuzz :n 2000 :seed 1  # a differential fuzzing campaign against Python memo (memo.fuzz)
```

gen.clj is pinned to GenGDT's commit (`deps.edn`); move the two pins together.
After a fresh clone, run `clojure -X:deps prep` once (gen.clj compiles its
Java binding).

## Layers

| Layer | Namespaces |
|---|---|
| front end | `memo.parse` (forms → AST), `memo.ast`, `memo.frames`, `memo.analyze` (epistemic analysis: frames, known/deps, compile errors; emits the IR) |
| IR | `memo.ir`: single-assignment instructions over dims, with `check` |
| lowering | `memo.lower`: the exact lowering, IR → gen.array |
| API | `memo.core` (`defmemo`, calls, `print-table`, `program`), `memo.domains` (`defenum`), `memo.errors` |

## Rules

1. **Python memo's semantics, checked against it.** Every construct has
   parity tests against `test/fixtures/*.edn`, generated from a Python twin of
   each test model in `test/fixtures/generate.py`. Compile errors match
   Python memo's messages. Divergences are documented (DESIGN.md §2.4, §8).
2. **JAX's values, not gen.array's.** Where gen.array gives other values than
   JAX, `memo.lower` follows JAX. Booleans are the known case: gen.array's
   `(+ true true)` is 2 (Julia's rule), JAX's `True + True` is `True`. Inside
   a defmemo, `+` is memo's operator; in gen.clj code (and memo.clj's own
   array code) it is gen.array's, referred by the house style, so the same
   expression differs across that boundary. Find such differences with the
   MCP tools before assuming.
3. **Single assignment.** The IR never rebinds a symbol (`memo.ir/check`).
   Python memo's rebinding of observed values is its snapshot bug.
4. **Loud errors.** A construct not built yet, an operation JAX rejects, a
   value gen.array cannot compute as JAX does: each is an `ex-info` that says
   so, never a silent number.
5. **gen.clj's house style** where arrays are computed: refer gen.array's
   operators in place of clojure.core's (`(:refer-clojure :exclude [+ - * / <
   > <= >= == abs])`); never refer `max`, `min`, `take` or `flatten`.
6. **Tests run on every backend.** Read results with `memo.test-support/host`
   and compare with `close?` (float32 aware). Never call `array/item` inside a
   function being differentiated: it reads the primal and drops the gradient.
7. **A test must be able to fail.** Check new tests against a planted bug.
8. **Host functions** are tabulated on host numbers unless marked
   `^:memo/vectorized` (gen.array operators, elementwise); only those are
   differentiated, and an unmarked one given a differentiated value is a
   loud error (DESIGN.md §5). Inside a vectorized function, branch with
   `where`, never `if`: an `if` on an array takes one branch for all of it.
   Each call is spot-checked against host numbers on eager backends.
