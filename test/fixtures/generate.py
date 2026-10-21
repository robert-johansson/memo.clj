"""Reference outputs from Python memo for memo.clj's tests (DESIGN.md §9).

From memo.clj's root, with memo-mcp's environment (memo-mcp is gitignored):

    memo-mcp/venv/bin/python -I test/fixtures/generate.py

It imports memo from memo-mcp's checkout of kach/memo (not the venv's PyPI
package), records that checkout's commit, and writes test/fixtures/*.edn.
The tests read those files and never need Python.

Each model here has a twin with the same name, in kebab case, in the test
namespace named by its fixture file (slice1.edn: memo.slice1-test).
Results are recorded at the full shape of the memo's axes: Python memo leaves
size 1 the axes a result does not depend on, and memo.clj broadcasts.
"""

import json
import math
import os
import subprocess
import sys
from enum import IntEnum

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMO_SRC = os.path.join(ROOT, "memo-mcp", "memo")
sys.path.insert(0, MEMO_SRC)

import jax  # noqa: E402
import jax.numpy as np  # noqa: E402
import memo as memo_pkg  # noqa: E402
from memo import memo  # noqa: E402
from memo.core import MemoError  # noqa: E402

assert os.path.dirname(os.path.dirname(memo_pkg.__file__)) == MEMO_SRC, memo_pkg.__file__


# --- EDN ---------------------------------------------------------------------

class Kw(str):
    """An EDN keyword."""


def edn(x):
    if x is None:
        return "nil"
    if isinstance(x, Kw):
        return ":" + x
    if isinstance(x, dict):
        return "{" + "\n ".join(f"{edn(k)} {edn(v)}" for k, v in x.items()) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + " ".join(edn(v) for v in x) + "]"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        return json.dumps(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "##NaN"
        if math.isinf(x):
            return "##Inf" if x > 0 else "##-Inf"
        return repr(x)
    raise TypeError(f"no EDN for {x!r}")


def kebab(name):
    return Kw(name.replace("_", "-"))


def tolist(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return float(a)
    return [tolist(x) for x in a]


def oracle():
    commit = subprocess.run(["git", "-C", MEMO_SRC, "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return {Kw("memo"): memo_pkg.__version__, Kw("commit"): commit,
            Kw("jax"): jax.__version__, Kw("dtype"): "float32"}


def number(a):
    """An argument as recorded: Python's ints stay ints, as Python memo
    computes a parameter as the number it is given ({x << 4} needs one, and
    2 ** -1 is 0.5 while memo.clj's long ** gives 0)."""
    return a if isinstance(a, int) else float(a)


def case(f, *args):
    out = np.asarray(f(*args))
    shape = tuple(f._shape) if f._num_retvals == 1 else (f._num_retvals,) + tuple(f._shape)
    return {Kw("model"): kebab(f.__name__), Kw("args"): [number(a) for a in args],
            Kw("shape"): list(shape), Kw("value"): tolist(np.broadcast_to(out, shape))}


def compile_error(f, **opts):
    try:
        memo(f, **opts)
    except MemoError as e:
        return e.message
    raise AssertionError(f"{f.__name__} compiled, but should not")


def runtime_error(f, *args):
    try:
        f(*args)
    except MemoError as e:
        return e.message
    except Exception as e:
        return str(e)
    raise AssertionError(f"{f.__name__} ran, but should not")


def grad(f, weights, *args):
    """d/d(first argument) of sum(weights * f(*args)), by jax.grad."""
    w = np.array(weights)
    g = jax.grad(lambda a0: (w * f(a0, *args[1:])).sum())(args[0])
    return {Kw("model"): kebab(f.__name__), Kw("args"): [float(a) for a in args],
            Kw("weights"): tolist(w), Kw("value"): float(g)}


def ledger(wrong, right, weights, *args, why):
    """An upstream-ledger entry (DESIGN.md §9): Python memo's gradient of
    wrong is not the derivative of its values, and right is a twin with the
    same values whose gradient Python memo computes correctly."""
    g_wrong, g_right = grad(wrong, weights, *args), grad(right, weights, *args)
    return {Kw("model"): kebab(wrong.__name__), Kw("args"): g_wrong[Kw("args")],
            Kw("weights"): g_wrong[Kw("weights")], Kw("python"): g_wrong[Kw("value")],
            Kw("correct"): g_right[Kw("value")], Kw("why"): why}


def ledger_value(wrong, right, *args, why):
    """An upstream-ledger entry for values: Python memo's value of wrong (or
    its error) is not right, and right is a twin whose value is the correct
    one for wrong (computed correctly by Python memo)."""
    shape = tuple(wrong._shape) if wrong._num_retvals == 1 else (wrong._num_retvals,) + tuple(wrong._shape)
    try:
        python = tolist(np.broadcast_to(np.asarray(wrong(*args)), shape))
    except Exception as e:
        python = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    correct = tolist(np.broadcast_to(np.asarray(right(*args)), shape))
    return {Kw("model"): kebab(wrong.__name__), Kw("args"): [number(a) for a in args],
            Kw("python"): python, Kw("correct"): correct, Kw("why"): why}


def write(name, cases, errors, runtime_errors=(), grads=(), ledger=()):
    """errors: functions, or (function, memo options) pairs, that must not
    compile; runtime_errors: (memo, args) that compile and must fail when
    called; grads: from grad(); ledger: from ledger()."""
    path = os.path.join(ROOT, "test", "fixtures", name)
    errors = [e if isinstance(e, tuple) else (e, {}) for e in errors]
    data = {Kw("oracle"): oracle(), Kw("cases"): cases,
            Kw("errors"): {kebab(f.__name__): compile_error(f, **opts) for f, opts in errors},
            Kw("runtime-errors"): {kebab(f.__name__): runtime_error(f, *args)
                                   for f, *args in runtime_errors},
            Kw("grads"): list(grads),
            Kw("ledger"): list(ledger)}
    with open(path, "w") as out:
        out.write(";; Generated by test/fixtures/generate.py: do not edit.\n")
        out.write(edn(data) + "\n")
    print(f"wrote {path}: {len(cases)} cases, {len(errors)} errors")


# --- slice 1: one frame, a wpp choice, Pr and E ---------------------------------

class Coin(IntEnum):
    HEADS = 0
    TAILS = 1


N = np.arange(6)


@memo
def coin_flip[c: Coin](p):
    alice: chooses(c in Coin, wpp=p if c == {Coin.HEADS} else 1 - p)
    return Pr[alice.c == c]


@memo
def mean_roll(beta):
    alice: chooses(n in N, wpp=exp(beta * n))
    return E[alice.n]


@memo
def roll_table[n: N](beta):
    alice: chooses(n in N, wpp=exp(beta * n))
    return Pr[alice.n == n]
    return E[alice.n]


@memo
def roll_stats(beta):
    alice: chooses(n in N, wpp=exp(beta * n))
    return Var[alice.n]
    return Pr[alice.n > 2]
    return E[alice.n * alice.n]


@memo
def argmax_pick[n: N]():
    alice: chooses(n in N, to_maximize=-abs(n - 2.5))
    return Pr[alice.n == n]


@memo
def argmin_pick[n: N]():
    alice: chooses(n in N, to_minimize=(n - 2) ** 2)
    return Pr[alice.n == n]


@memo
def two_choices[x: Coin, y: Coin](p):
    alice: chooses(x in Coin, y in Coin, wpp=(p if x == y else 1 - p) * (1 + x))
    return Pr[alice.x == x, alice.y == y]


@memo
def two_agents(p):
    alice: chooses(x in Coin, wpp=p if x == {Coin.HEADS} else 1 - p)
    bob: chooses(y in Coin, wpp=1)
    return Pr[alice.x == bob.y]


@memo
def zero_mass():
    alice: chooses(n in N, wpp=0 * n)
    return E[alice.n] + Pr[alice.n == 0] + 10


@memo
def const_return[n: N, c: Coin]():
    return 1.0


@memo
def axis_return[n: N, c: Coin]():
    return n * 10 + c


@memo
def such_that_pick():
    alice: chooses(n in N, such_that=n < 3)
    return E[alice.n]


@memo
def uniform_pick():
    alice: chooses(n in N, uniformly)
    return E[alice.n]


@memo
def to_be_pick():
    alice: chooses(n in N, to_be=4)
    return E[alice.n]


@memo
def bool_sum():
    alice: chooses(n in N, wpp=(n == 0) + (n < 2))
    return Pr[alice.n == 0]


@memo
def default_param(beta=0.5):
    alice: chooses(n in N, wpp=exp(beta * n))
    return E[alice.n]


@memo
def exp_overflow(beta):
    alice: chooses(n in N, wpp=1.0 * exp(beta * n))
    return E[alice.n] + Pr[alice.n == 5]


@memo
def static_ite(p):
    alice: chooses(n in N, wpp=n + 1 if p > 0 else 6 - n)
    return E[alice.n]


@memo
def ops_pick[n: N]():
    alice: chooses(n in N, wpp=((n < 3) ^ (n % 2 == 0)) + (not (n == 5)) * 0.5 + n ** 2 / 10)
    return Pr[alice.n == n]


def return_uncertain():
    alice: chooses(n in N, uniformly)
    return alice.n


def unknown_choice():
    alice: chooses(n in N, wpp=m)
    return 1


def repeated_choice():
    alice: chooses(n in N, uniformly)
    alice: chooses(n in N, uniformly)
    return 1


def name_conflict(n):
    alice: chooses(n in N, uniformly)
    return 1


def other_agent_unknown():
    bob: chooses(m in N, uniformly)
    alice: chooses(n in N, wpp=bob.m)
    return 1


write("slice1.edn",
      [case(coin_flip, 0.3),
       case(mean_roll, 0.5),
       case(mean_roll, 1000.0),
       case(roll_table, -0.3),
       case(roll_stats, 0.7),
       case(argmax_pick),
       case(argmin_pick),
       case(two_choices, 0.2),
       case(two_agents, 0.3),
       case(zero_mass),
       case(const_return),
       case(axis_return),
       case(such_that_pick),
       case(uniform_pick),
       case(to_be_pick),
       case(bool_sum),
       case(default_param),
       case(default_param, 2.0),
       case(exp_overflow, 1000.0),
       case(static_ite, 1.0),
       case(static_ite, -1.0),
       case(ops_pick)],
      [return_uncertain, unknown_choice, repeated_choice, name_conflict,
       other_agent_unknown])


# --- slice 2: thinks, observes, observes-that/-event, knows, has-theory ---------

from memo import Variant  # noqa: E402

X = np.arange(3)


class Bool(IntEnum):
    NO = 0
    YES = 1


@memo
def observes_other():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    charlie: chooses(x in X, wpp=1)
    alice: observes [bob.x] is charlie.x
    return E[alice[bob.x]]


@memo
def observes_that_const():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes_that [bob.x == 0]
    return alice[E[bob.x]]


@memo
def observes_event_float():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes_event(wpp=bob.x / 3)
    return alice[E[bob.x]]


@memo
def observes_event_own():
    alice: chooses(x in X, wpp=1)
    alice: observes_event(wpp=x / 3)
    return E[alice.x]


@memo
def observes_event_void():
    alice: observes_event(wpp=3.14)
    return alice[2]


@memo
def post_diag_1[x: X]():
    bob: knows(x)
    bob: chooses(y in X, wpp=(x + y))
    return Pr[bob.y == x]


@memo
def post_diag_2[x: X]():
    bob: thinks[ alice: chooses(x in X, wpp=1) ]
    bob: observes [alice.x] is x
    bob: chooses(y in X, wpp=(alice.x + y))
    return Pr[bob.y == x]


@memo
def post_diag_3[x: X]():
    bob: chooses(x in X, wpp=1)
    bob: chooses(y in X, wpp=(x + y))
    return Pr[bob.y == x]


@memo
def jianlin[x: X]():
    a: thinks[ b: chooses(u in X, wpp=1.0) ]
    a: observes [b.u] is x
    a: chooses(v in X, wpp=b.u == v)
    return Pr[a.v == x]


@memo
def var_redundant_known():
    alice: chooses(coffee in Bool, uniformly)
    bob: knows(alice.coffee)
    return E[bob[Var[alice.coffee == 1]]]


@memo
def child_thinks_ok():
    child: has_theory(CHILD)
    child: thinks[ adult: chooses(x in X, wpp=1) ]
    return child[E[adult.x]]


@memo(variant=Variant.CHILD)
def child_toplevel():
    child: chooses(x in X, uniformly)
    return E[child.x]


@memo
def knows_copy[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: thinks[ charlie: knows(bob.x) ]
    alice: observes [bob.x] is x
    return alice[charlie[bob.x]]


@memo
def via_knows[y: X]():
    bob: chooses(y in X, wpp=y + 1)
    alice: knows(bob.y)
    alice: chooses(z in X, wpp=exp(bob.y * z))
    return Pr[alice.z == y]


@memo
def via_observes[y: X]():
    bob: chooses(y in X, wpp=y + 1)
    alice: thinks[ bob: chooses(y in X, wpp=1) ]
    alice: observes [bob.y] is bob.y
    alice: chooses(z in X, wpp=exp(bob.y * z))
    return Pr[alice.z == y]


@memo
def impossible_obs():
    bob: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(x in X, wpp=(x < 2)) ]
    alice: observes [bob.x] is bob.x
    return E[alice[Pr[bob.x == 2]]]


@memo
def impossible_obs_ll():
    bob: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(x in X, wpp=(x < 2)), bob: chooses(y in X, wpp=(y == x) + 1.0) ]
    alice: observes [bob.x] is bob.x
    return E[alice[E[bob.y]]]


@memo
def scramble_unread[s: X, t: X]():
    obs: knows(t)
    obs: knows(s)
    obs: thinks[ w: knows(s), w: chooses(x in X, wpp=2 if x == s else 1) ]
    obs: observes [w.x] is s
    return obs[Pr[w.x == t]]


Door = np.arange(3)


@memo
def monty_hall[pick: Door, reveal: Door, d: Door]():
    alice: thinks[ monty: chooses(prize in Door, wpp=1) ]
    alice: knows(pick)
    alice: thinks[
        monty: knows(pick),
        monty: chooses(reveal in Door, wpp=(reveal != prize and reveal != pick))
    ]
    alice: observes [monty.reveal] is reveal
    alice: knows(d)
    return alice[Pr[monty.prize == d]]


class Opinion(IntEnum):
    BAD_POLICY = 0
    GOOD_POLICY = 1


class Outcome(IntEnum):
    NO_SPENDING = 0
    SPENDING_INCREASE = 1


class Result(IntEnum):
    NO_SPENDING = 0
    SPENDING_INCREASE = 1


class Policy(IntEnum):
    CONSERVATIVE = 0
    LIBERAL = 1


class Economist(IntEnum):
    ALICE = 0
    BOB = 1


@memo
def polarization[e: Economist, d: Result]():
    economist: knows(e)
    economist: thinks[
        world: knows(e),
        world: chooses(v1 in Policy, wpp=(
            (0.9 if v1 == {Policy.LIBERAL} else 0.1) if e == {Economist.ALICE} else
            (0.1 if v1 == {Policy.LIBERAL} else 0.9))),
        world: chooses(v2 in Outcome, wpp=1),
        world: chooses(d in Result, wpp=0.9 if d == v2 else 0.1),
        world: chooses(h in Opinion, wpp=(
            0.5 if v2 == {Outcome.NO_SPENDING} else (
                (0.1 if h == {Opinion.GOOD_POLICY} else 0.9) if v1 == {Policy.CONSERVATIVE} else
                (0.9 if h == {Opinion.GOOD_POLICY} else 0.1))))
    ]
    economist: observes [world.d] is d
    return economist[Pr[world.h == 1]]


class Loc(IntEnum):
    BOX = 0
    BASKET = 1


class Action(IntEnum):
    ACT_STAY = 0
    ACT_MOVE = 1


class Obs(IntEnum):
    OBS_NONE = -1
    OBS_STAY = 0
    OBS_MOVE = 1


# demo-sally-anne.ipynb, with its host function do(l, a) written inline
@memo
def sally_anne[marble_pos_t0: Loc, obs: Obs, where_look: Loc]():
    child: knows(marble_pos_t0, obs, where_look)
    child: thinks[
        sally: knows(marble_pos_t0),
        sally: thinks[
            anne: knows(marble_pos_t0),
            anne: chooses(a in Action, wpp=0.01 if a == {Action.ACT_MOVE} else 0.99),
            anne: chooses(marble_pos_t1 in Loc,
                          wpp=(marble_pos_t0 if a == {Action.ACT_STAY} else 1 - marble_pos_t0) == marble_pos_t1),
            anne: chooses(o in Obs, wpp=1 if o == {Obs.OBS_NONE} or o == a else 0),
        ],
        sally: observes [anne.o] is obs,
        sally: chooses(where_look in Loc, wpp=Pr[anne.marble_pos_t1 == where_look])
    ]
    return child[Pr[sally.where_look == where_look]]


def obs_unmodeled[x: X]():
    alice: observes [bob.x] is x
    return 1


def obs_known[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes [bob.x] is x
    alice: observes [bob.x] is x
    return 1


def obs_unknown_source():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes [bob.x] is z
    return 1


def obs_domain_mismatch[c: Coin]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes [bob.x] is c
    return 1


def knows_unknown():
    alice: knows(bob.x)
    return 1


def uncertain_wpp():
    alice: thinks[ bob: chooses(y in X, wpp=1) ]
    alice: chooses(x in X, wpp=bob.y)
    return 1


def ask_unknown():
    alice: thinks[ bob: chooses(y in X, wpp=1) ]
    return alice[bob.y]


def child_thinks_err():
    child: has_theory(CHILD)
    child: thinks[ adult: thinks[ mom: chooses(x in X, wpp=1) ] ]
    return 0


def child_toplevel_err():
    child: thinks[ adult: chooses(x in X, wpp=1) ]
    return 0


def no_thinks_err():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    return 0


def no_knows_err[x: X]():
    alice: knows(x)
    return 0


write("slice2.edn",
      [case(observes_other),
       case(observes_that_const),
       case(observes_event_float),
       case(observes_event_own),
       case(observes_event_void),
       case(post_diag_1),
       case(post_diag_2),
       case(post_diag_3),
       case(jianlin),
       case(var_redundant_known),
       case(child_thinks_ok),
       case(child_toplevel),
       case(knows_copy),
       case(via_knows),
       case(via_observes),
       case(impossible_obs),
       case(impossible_obs_ll),
       case(scramble_unread),
       case(monty_hall),
       case(polarization),
       case(sally_anne)],
      [obs_unmodeled, obs_known, obs_unknown_source, obs_domain_mismatch,
       knows_unknown, uncertain_wpp, ask_unknown, child_thinks_err,
       (child_toplevel_err, {"variant": Variant.CHILD}),
       (no_thinks_err, {"variant": Variant.NO_THINKS}),
       (no_knows_err, {"variant": Variant.NO_KNOWS})])


# --- slice 3: memo calls, path conditions, recursion -----------------------------

class U(IntEnum):
    GREEN = 0b0001
    PINK = 0b0010
    SQUARE = 0b0100
    ROUND = 0b1000


class R(IntEnum):
    GREEN_SQUARE = U.GREEN | U.SQUARE
    GREEN_CIRCLE = U.GREEN | U.ROUND
    PINK_CIRCLE = U.PINK | U.ROUND


# demo-rsa.py, with its host function denotes(u, r) = (u & r) != 0 written in
# memo's arithmetic (every u is a power of two)
@memo
def L[u: U, r: R](beta, t):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=(((r - r % u) / u) % 2 == 1)
                         * (1 if t == 0 else exp(beta * L[u, r](beta, t - 1))))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L0[u: U, r: R]():
    listener: knows(u)
    listener: chooses(r in R, wpp=((r - r % u) / u) % 2 == 1)
    return Pr[listener.r == r]


@memo
def S1[r: R, u: U](alpha):
    speaker: knows(r)
    speaker: chooses(u in U, wpp=exp(alpha * log(L0[u, r]())))
    return Pr[speaker.u == u]


# S1 with its weights as L0 ** alpha: the same values, and a finite gradient
# where exp(alpha * log(L0)) has NaN (log 0 = -inf, and 0 * -inf in its chain)
@memo
def S1_pow[r: R, u: U](alpha):
    speaker: knows(r)
    speaker: chooses(u in U, wpp=L0[u, r]() ** alpha)
    return Pr[speaker.u == u]


@memo
def L1[u: U, r: R](alpha):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=S1[r, u](alpha))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def test_[x: X](t):
    return x


@memo
def memo_call_ellipsis(t=0):
    alice: chooses(x in X, wpp=test_[x](...))
    return E[alice.x]


@memo
def recursive_kwarg[x: X](a, b=0):
    return recursive_kwarg[x](a=a - 1, b=b + 1) if a > 0 else b


@memo
def kwarg():
    alice: chooses(x in X, wpp=1)
    return E[recursive_kwarg[alice.x](a=3)]


@memo
def kwarg_mixed():
    alice: chooses(x in X, wpp=1)
    return E[recursive_kwarg[alice.x](3, b=10)]


@memo
def multi_return_1[x: X]():
    return 1
    return 2


@memo
def multi_return_2[x: X]():
    return multi_return_1[1][x]()


@memo
def observes_call[x: X]():
    a: thinks[ b: chooses(x in X, wpp=1) ]
    a: observes [b.x] is x
    return a[b[test_[x](0)]]


@memo
def table2[x: X, c: Coin]():
    return x * 10 + c


@memo
def call_two_agents():
    alice: chooses(x in X, wpp=1)
    bob: chooses(c in Coin, wpp=c + 1)
    return E[table2[alice.x, bob.c]()]


@memo
def call_reversed[c: Coin]():
    alice: knows(c)
    alice: chooses(x in X, wpp=table2[x, c]() + 1)
    return E[alice.x]


Unit = [0]


@memo
def fib[a: Unit](n):
    return 1 if n < 2 else fib[a](n - 1) + fib[a](n - 2)


@memo
def multi_return_3[x: X]():
    return multi_return_1[x]()


@memo
def multi_return_4[x: X]():
    return multi_return_1[3][x]()


@memo
def multi_return_5[x: X]():
    return test_[0][x](0)


@memo
def call_domain_mismatch[c: Coin]():
    return test_[c](0)


def call_not_static():
    alice: chooses(x in X, wpp=test_[x](x))
    return 1


def call_kwarg_not_static():
    alice: chooses(x in X, wpp=test_[x](t=x))
    return 1


def call_unknown_choice():
    return test_[z](0)


def call_bad_axis[x: X]():
    return test_[x + 1](0)


W43 = [[0.3, -1.0, 0.7], [1.2, 0.4, -0.5], [-0.8, 0.9, 0.2], [0.5, -0.3, 1.1]]
W34 = [[0.3, 1.2, -0.8, 0.5], [-1.0, 0.4, 0.9, -0.3], [0.7, -0.5, 0.2, 1.1]]

write("slice3.edn",
      [case(L, 1.0, 0),
       case(L, 1.0, 1),
       case(L, 1.74, 2),
       case(L, 0.5, 3),
       case(L0),
       case(S1, 1.3),
       case(S1_pow, 1.3),
       case(L1, 1.3),
       case(test_, 0),
       case(memo_call_ellipsis),
       case(recursive_kwarg, 2),
       case(kwarg),
       case(kwarg_mixed),
       case(multi_return_1),
       case(multi_return_2),
       case(observes_call),
       case(table2),
       case(call_two_agents),
       case(call_reversed),
       case(fib, 0),
       case(fib, 1),
       case(fib, 10)],
      [call_not_static, call_kwarg_not_static, call_unknown_choice, call_bad_axis],
      runtime_errors=[(multi_return_3,), (multi_return_4,), (multi_return_5,),
                      (call_domain_mismatch,)],
      grads=[grad(L, W43, 1.74, 1),
             grad(L, W43, 0.6, 3),
             grad(L1, W43, 1.3),
             grad(S1, W34, 1.3),
             grad(S1_pow, W34, 1.3)])


# --- slice 4: host functions and the inline escape hatch ------------------------

from jax.scipy.stats.norm import pdf as norm_pdf  # noqa: E402

N5 = 5
N11 = np.arange(11)
B2 = np.arange(2)


@jax.jit
def plus_one(n):
    return n + 1


@memo
def inline_const():
    return {N5}


@memo
def inline_call():
    return plus_one({N5})


@memo
def inline_memo[x: X]():
    return test_[x]({N5})


@memo
def inline_param(p):
    return {p * 10} + 1


@jax.jit
def returns_scalar(x):
    return np.cos(x) + np.array([0, 1, 2])[x]


@memo
def ffi_ok():
    alice: chooses(x in X, wpp=1)
    return E[returns_scalar(alice.x)] + 12


@jax.jit
def returns_scalar_no_arg():
    return np.cos(3.14)


@memo
def ffi_ok_no_arg():
    return returns_scalar_no_arg() + 15


@jax.jit
def takes_nonstatic_arg(x, Z, y):
    return Z[x] * y


@memo
def ffi_static():
    alice: chooses(x in X, wpp=takes_nonstatic_arg(x, {np.array([0, 1, 2, 3, 4])}, 3.0))
    return E[alice.x]


@memo
def ffi_kwargs():
    agent: chooses(n in N11, wpp=norm_pdf(n, scale=1, loc=5))
    return E[agent.n]


@jax.jit
def kw_only_func(a, b):
    return a + b


@memo
def ffi_kwargs_only():
    alice: chooses(x in X, wpp=1)
    return E[kw_only_func(a=alice.x, b=1.0)]


@jax.jit
def denotes(u, r):
    return (u & r) != 0


# demo-rsa.py as written, with its host function
@memo
def L_ffi[u: U, r: R](beta, t):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=denotes(u, r) * (1 if t == 0 else exp(beta * L_ffi[u, r](beta, t - 1))))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@jax.jit
def do_action(l, a):
    return np.array([[0, 1], [1, 0]])[a, l]


# demo-sally-anne.ipynb as written (its do renamed do_action: do is a Clojure special form)
@memo
def sally_anne_ffi[marble_pos_t0: Loc, obs: Obs, where_look: Loc]():
    child: knows(marble_pos_t0, obs, where_look)
    child: thinks[
        sally: knows(marble_pos_t0),
        sally: thinks[
            anne: knows(marble_pos_t0),
            anne: chooses(a in Action, wpp=0.01 if a == {Action.ACT_MOVE} else 0.99),
            anne: chooses(marble_pos_t1 in Loc, wpp=do_action(marble_pos_t0, a) == marble_pos_t1),
            anne: chooses(o in Obs, wpp=1 if o == {Obs.OBS_NONE} or o == a else 0),
        ],
        sally: observes [anne.o] is obs,
        sally: chooses(where_look in Loc, wpp=Pr[anne.marble_pos_t1 == where_look])
    ]
    return child[Pr[sally.where_look == where_look]]


@jax.jit
def gauss(x, mu):
    return np.exp(-(x - mu) ** 2)


@memo
def ffi_param_grad(mu):
    alice: chooses(n in N11, wpp=gauss(n, mu))
    return E[alice.n]


@jax.jit
def sq(a):
    return a * a


@memo
def g_static_ffi(a):
    alice: chooses(x in B2, wpp=exp(sq(a) * x))
    return Pr[alice.x == 1]


@memo
def g_static_ref(a):
    alice: chooses(x in B2, wpp=exp(a * a * x))
    return Pr[alice.x == 1]


@memo
def h(a):
    alice: chooses(x in X, wpp=exp(a * x))
    return E[alice.x]


@memo
def calls_h(a):
    return h(a) * 2


@memo
def calls_h_ref(a):
    alice: chooses(x in X, wpp=exp(a * x))
    return E[alice.x] * 2


@jax.jit
def returns_nonscalar0():
    return np.array([0, 1])


@memo
def ffi_scalar0():
    return returns_nonscalar0()


@jax.jit
def returns_nonscalar1(x):
    return np.array([0, 1])


@memo
def ffi_scalar1():
    return returns_nonscalar1(1.0)


write("slice4.edn",
      [case(inline_const),
       case(inline_call),
       case(inline_memo),
       case(inline_param, 0.7),
       case(ffi_ok),
       case(ffi_ok_no_arg),
       case(ffi_static),
       case(ffi_kwargs),
       case(ffi_kwargs_only),
       case(L_ffi, 1.0, 0),
       case(L_ffi, 1.74, 2),
       case(sally_anne_ffi),
       case(ffi_param_grad, 4.3),
       case(g_static_ffi, 0.5),
       case(calls_h, 0.3)],
      [],
      runtime_errors=[(ffi_scalar0,), (ffi_scalar1,)],
      grads=[grad(ffi_param_grad, 1.0, 4.3),
             grad(L_ffi, W43, 1.74, 2)],
      ledger=[ledger(g_static_ffi, g_static_ref, 1.0, 0.5,
                     why="a host function of parameters only is evaluated outside jit and "
                         "converted with .item(), so its gradient is 0"),
              ledger(calls_h, calls_h_ref, 1.0, 0.3,
                     why="a memo without axes can only be called as a host function, so "
                         "the call's gradient is 0")])


# --- slice 5: imagine, snapshots, wants/EU/Predict -------------------------------

@memo
def observes_other_imagine():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    charlie: chooses(x in X, wpp=1)
    alice: observes [bob.x] is charlie.x
    return E[alice[imagine[env: knows(bob.x), bob.x + 1]]]


@memo
def imagine_ok():
    return alice[imagine[bob: chooses(y in X, wpp=1), E[bob.y]]]


@memo
def imagine_knows():
    alice: chooses(x in X, wpp=1)
    return E[alice[imagine[world: knows(x), world.x]]]


@memo
def imagine_knows_other[z: X]():
    alice: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(z in X, wpp=1) ]
    alice: observes [bob.z] is z
    return alice[imagine[world: knows(bob.z), world[bob.z]]]


@memo
def imagine_future_stress():
    alice: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(z in X, wpp=1) ]
    alice: snapshots_self_as(future_alice)
    return E[alice[imagine[
        world: knows(x, bob.z),
        world: chooses(z in X, wpp=1),
        future_alice: chooses(y in X, wpp=x + y),
        future_alice: thinks[ world: chooses(z in X, wpp=1) ],
        future_alice: observes [world.z] is world.z,
        E[future_alice.y + future_alice[world.z + y] + world.z + world[bob.z]]
    ]]]


@memo
def imagine_toplevel():
    return imagine[alice: chooses(x in X, wpp=1), E[alice.x]]


@memo
def predict_own():
    alice: chooses(x in X, to_maximize=Predict[x])
    return E[alice.x]


@memo
def predict():
    bob: chooses(z in X, wpp=1)
    bob: thinks[
        alice: chooses(x in X, to_maximize=Predict[z]),
        alice: chooses(y in X, to_maximize=Predict[z]),
        alice: thinks[ world: knows(x, y), world: chooses(z in X, wpp=z + x + y) ],
        alice: observes [world.z] is z,
        alice: chooses(z in X, wpp=world.z + z)
    ]
    return E[bob[E[alice.x]]]


@memo
def two_goals[x: X, y: X]():
    bob: thinks[
        alice: wants(u=x + y),
        alice: wants(v=x - y),
        alice: chooses(x in X, to_maximize=EU[u]),
        alice: chooses(y in X, to_maximize=EU[v])
    ]
    bob: knows(x, y)
    return bob[Pr[alice.x == x, alice.y == y]]


class Cups(IntEnum):
    Near_Vizzini = 0
    Near_Westley = 1


@memo
def vizzini_pick[cup: Cups](level):
    vizzini: wants(survive=my_cup != westley.poison)
    vizzini: thinks[
        westley: wants(kill=vizzini.my_cup == poison),
        westley: chooses(poison in Cups, to_maximize=EU[kill]),
        westley: thinks[
            vizzini: chooses(my_cup in Cups,
                             wpp=vizzini_pick[my_cup](level - 1) if level > 0 else my_cup == {Cups.Near_Vizzini})
        ]
    ]
    vizzini: chooses(my_cup in Cups, to_maximize=EU[survive])
    return Pr[vizzini.my_cup == cup]


from jax.scipy.stats.bernoulli import pmf as ber  # noqa: E402

Value = np.linspace(0, 1, 101)


@memo
def lemons_buyer():
    buyer: wants(payoff=(seller.quality * 1.5 - bid) if seller.sell else 0)
    buyer: chooses(bid in Value, to_maximize=EU[payoff])
    buyer: thinks[
        seller: given(quality in Value, wpp=1),
        seller: knows(bid),
        seller: chooses(sell in Bool, wpp=ber(sell, bid >= quality))
    ]
    return E[buyer.bid]


class Box(IntEnum):
    FST_BOX = 0
    SND_BOX = 1


class FoodStatus(IntEnum):
    EMPTY_BOX = 0
    FOODY_BOX = 1


class EvidenceType(IntEnum):
    CRUMBS = 0
    NOISES = 1
    VISUAL = 2


def p_evidence(et, fs):
    return np.array([[0.5, 0.6], [0.3, 0.6], [0.01, 0.9]])[et, fs]


@memo
def first_choice[et1: EvidenceType, b: Box]():
    chimp: knows(et1)
    chimp: wants(get_food=(world.box1_fs if b == {Box.FST_BOX} else world.box2_fs) == {FoodStatus.FOODY_BOX})
    chimp: thinks[
        world: chooses(box1_fs in FoodStatus, wpp=1),
        world: chooses(box2_fs in FoodStatus, wpp=1),
        world: knows(et1),
        world: chooses(e1 in FoodStatus, wpp=ber(e1, p_evidence(et1, box1_fs))),
    ]
    chimp: observes_that [world.e1 == {FoodStatus.FOODY_BOX}]
    chimp: chooses(b in Box, wpp=exp(10.0 * EU[get_food]))
    return Pr[chimp.b == b]


class Vocab(IntEnum):
    Ball = 0
    Bear = 1
    Duck = 2
    Milk = 3
    Shoe = 4


@memo
def novel_event_inform[fixed_event: Vocab, novel_event: Vocab, utterance: Vocab](beta):
    baby: knows(fixed_event, novel_event)
    baby: wants(inform=Pr[parent.referent_guess == novel_event])
    baby: thinks[
        parent: thinks[
            scene: chooses(fixed_event in Vocab, wpp=1),
            scene: chooses(novel_event in Vocab, wpp=1)
        ],
        parent: observes [scene.fixed_event] is fixed_event,
        parent: thinks[
            baby: knows(scene.fixed_event, scene.novel_event),
            baby: chooses(referent in Vocab, wpp=referent == scene.fixed_event or referent == scene.novel_event),
            baby: chooses(utterance in Vocab, wpp=(utterance == referent) + 0.25)
        ]
    ]
    baby: chooses(utterance in Vocab, wpp=exp(beta * EU[inform]))
    baby: thinks[
        parent: observes [baby.utterance] is utterance,
        parent: guesses(referent_guess in Vocab, wpp=Pr[referent_guess == baby.referent])
    ]
    return Pr[baby.utterance == utterance]


class Sp(IntEnum):
    Hungry = 0
    Sated = 1


class Ap(IntEnum):
    Feed = 0
    Sing = 1
    Ignore = 2


class Op(IntEnum):
    Crying = 0
    Quiet = 1


Bp = np.linspace(0, 1, 50)


@jax.jit
def get_belief(b, s):
    return np.array([b, 1 - b])[s]


@jax.jit
def tr(s, a, s_):
    z = np.array([[0.0, 1.0, 1.0], [0.0, 0.1, 0.1]])[s, a]
    return np.array([z, 1 - z])[s_]


@jax.jit
def obs_p(o, s, a):
    z = np.array([[0.8, 0.9, 0.8], [0.1, 0.0, 0.1]])[s, a]
    return np.array([z, 1 - z])[o]


@jax.jit
def reward(s, a):
    return np.array([-10, 0])[s] + np.array([-5, -0.5, 0])[a]


# demo-pomdp.ipynb (its names changed: S, A, O and R are taken in this file)
@memo
def Q_pomdp[b: Bp, a: Ap](t):
    alice: knows(b, a)
    alice: thinks[
        env: knows(b, a),
        env: chooses(s in Sp, wpp=get_belief(b, s)),
        env: chooses(s_ in Sp, wpp=tr(s, a, s_)),
        env: chooses(o in Op, wpp=obs_p(o, s_, a))
    ]
    alice: snapshots_self_as(future_alice)
    return alice[E[reward(env.s, a)] + (0.0 if t <= 0 else 0.9 * imagine[
        future_alice: observes [env.o] is env.o,
        future_alice: chooses(b_ in Bp, wpp=exp(-100.0 * abs(E[env.s_ == 0] - b_))),
        future_alice: chooses(a_ in Ap, to_maximize=Q_pomdp[b_, a_](t - 1)),
        E[future_alice[Q_pomdp[b_, a_](t - 1)]]
    ])]


# the snapshot bug (DESIGN.md §2.4), fixed in memo 1.3.3: a snapshot is
# unaffected by its owner's later observation
@memo
def snap_f[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: snapshots_self_as(past_alice)
    alice: observes [bob.x] is x
    return alice[past_alice[E[bob.x]]]


@memo
def snap_g[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1), bob: chooses(y in X, wpp=(y == x)) ]
    alice: snapshots_self_as(past_alice)
    alice: observes [bob.x] is x
    return alice[past_alice[E[bob.x + bob.y]]]


@memo
def snap_pr[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: snapshots_self_as(past_alice)
    alice: observes [bob.x] is x
    return alice[past_alice[Pr[bob.x == 2]]]


@memo
def eu_leak():
    world: chooses(s in B2, wpp=1)
    alice: thinks[ world: chooses(s in B2, wpp=1 + 2 * s) ]
    alice: wants(win=(world.s == g))
    alice: chooses(look in B2, to_maximize=EU[win] - 0.1 * look)
    alice: chooses(m in B2, wpp=exp(E[world.s] * m))
    alice: observes [world.s] is world.s
    alice: chooses(g in B2, wpp=Pr[world.s == g])
    return Pr[alice.m == 1]


def imagine_unknown_err():
    return alice[imagine[bob: chooses(y in X, wpp=1), bob.y]]


def predict_fail_knows():
    bob: chooses(z in X, wpp=1)
    bob: thinks[
        alice: chooses(x in X, to_maximize=Predict[x]),
        alice: knows(z)
    ]
    return E[bob[E[alice.x]]]


def unknown_goal():
    alice: chooses(x in X, to_maximize=EU[g])
    return 1


def no_wants_err():
    alice: wants(g=1)
    return 0


W555 = np.linspace(-1, 1, 125).reshape(5, 5, 5)

write("slice5.edn",
      [case(observes_other_imagine),
       case(imagine_ok),
       case(imagine_knows),
       case(imagine_knows_other),
       case(imagine_future_stress),
       case(imagine_toplevel),
       case(predict_own),
       case(predict),
       case(two_goals),
       case(vizzini_pick, 0),
       case(vizzini_pick, 1),
       case(vizzini_pick, 3),
       case(lemons_buyer),
       case(first_choice),
       case(novel_event_inform, 2.0),
       case(Q_pomdp, 0),
       case(Q_pomdp, 1),
       case(Q_pomdp, 2),
       case(snap_f),
       case(snap_g),
       case(snap_pr),
       case(eu_leak)],
      [imagine_unknown_err, predict_fail_knows, unknown_goal,
       (no_wants_err, {"variant": Variant.NO_WANTS})],
      grads=[grad(novel_event_inform, W555, 2.0)])


# --- slice 6: H, KL, forgets-about, cast, array parameters, cache --------------------

Rkl = np.linspace(-10, 10, 1000)


@jax.jit
def normpdf(x, mu, sigma):
    return jax.scipy.stats.norm.pdf(x, mu, sigma)


@memo
def kl():
    alice: chooses(p in Rkl, wpp=normpdf(p, 0, 1))
    alice: chooses(q in Rkl, wpp=normpdf(q, 0, 2))
    return KL[alice.p | alice.q]


@memo
def kl_victor[x: X]():
    bob: thinks[ alice: given(x in X, wpp=1), env: chooses(x in X, wpp=1) ]
    return bob[KL[alice.x | env.x]]


@memo
def entropy_uniform():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    return alice[H[bob.x]]


@memo
def entropy_joint(beta):
    alice: thinks[ bob: chooses(x in X, wpp=exp(beta * x)), bob: chooses(y in X, wpp=(y >= x)) ]
    return alice[H[bob.x, bob.y]]


@memo
def entropy_marginal():
    alice: thinks[ bob: chooses(x in X, wpp=1), bob: chooses(y in X, wpp=(y == x)) ]
    return alice[H[bob.y]]


is_prime = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
Qs = [lambda n: n == 7, lambda n: n > 6, lambda n: n % 2 == 0, lambda n: is_prime[n]]
Ne = np.arange(1, 6 + 1)
Qe = np.arange(len(Qs))
Ae = np.array([0, 1])


@jax.jit
def respond(q, a, n):
    return np.array([q_(n) for q_ in Qs])[q] == a


# demo-eig.ipynb (its N, Q, A renamed: taken in this file)
@memo
def eig[q: Qe]():
    alice: knows(q)
    alice: thinks[
        bob: chooses(n_red in Ne, wpp=1),
        bob: chooses(n_blu in Ne, wpp=1),
        bob: knows(q),
        bob: chooses(a in Ae, wpp=respond(q, a, n_red + n_blu))
    ]
    alice: snapshots_self_as(future_self)
    return alice[imagine[
        future_self: observes [bob.a] is bob.a,
        H[bob.n_red, bob.n_blu] - E[future_self[H[bob.n_red, bob.n_blu]]]
    ]]


@memo
def forgets_about_1():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in X, wpp=x + y)
    alice: forgets_about(x)
    return E[alice.y]


@memo
def forgets_about_multiple():
    alice: chooses(x in X, uniformly)
    alice: chooses(y in X, uniformly)
    alice: forgets_about(x, y)
    alice: chooses(z in Bool, uniformly)
    return E[alice.z]


@memo
def with_cast():
    cast: [alice, bob]
    alice: chooses(x in X, wpp=1)
    bob: knows(alice.x)
    return E[bob[alice.x]]


@jax.jit
def g_sum(n):
    return n.sum()


@jax.jit
def arr_at(arr, i):
    return arr[i]


@memo
def exotic_param(x: ... = X):
    return g_sum(x) + arr_at(x, 0)


@jax.jit
def exotic_kwarg_func(x, arr, scale=1.0):
    return arr[x] * scale


@memo
def exotic_param_kwarg(arr: ... = np.array([1, 2, 3])):
    alice: chooses(x in X, wpp=exotic_kwarg_func(x, arr=arr, scale=2.0))
    return E[alice.x]


@memo
def exotic_kwarg_recursive[x: X](k, arr: ...):
    alice: chooses(x in X, wpp=exotic_kwarg_func(x, arr=arr, scale=1.0))
    return exotic_kwarg_recursive[x](k=k - 1, arr=arr) if k > 0 else 1


@memo
def exotic_kwarg(arr: ... = np.array([1, 2, 3])):
    alice: chooses(x in X, wpp=1)
    return E[exotic_kwarg_recursive[alice.x](k=2, arr=arr)]


class DayAction(IntEnum):
    SH = 0
    GO = 1


class NightAction(IntEnum):
    Retire = 0
    Party = 1


class NightType(IntEnum):
    Responsible = 0
    Fun = 1


Pp = np.linspace(0, 1, 101)


def one_night_payoff(night_action):
    return np.array([1.0, 2.0])[night_action]


def one_day_payoff(day_action, night_action):
    return np.array([[+0, +0], [+1.0, -1.0]])[day_action, night_action]


# demo-penelope.ipynb (its P renamed)
@memo(cache=True)
def day_model[p: Pp, da: DayAction](n=0):
    day: knows(p, da)
    day: thinks[
        night: knows(p),
        night: given(t in NightType, wpp=ber(t, p)),
        night: chooses(na in NightAction,
                       wpp=(na == {NightAction.Retire}) if t == {NightType.Responsible}
                       else ber(na, night_model[0][p](n)))
    ]
    day: snapshots_self_as(next_day)
    return day[E[one_day_payoff(da, night.na)] + (0 if n <= 0 else imagine[
        next_day: observes [night.na] is night.na,
        next_day: chooses(p_ in Pp, to_minimize=abs(p_ - Pr[night.t == {NightType.Fun}])),
        next_day: chooses(da_ in DayAction, to_maximize=day_model[p_, da_](n - 1)),
        E[next_day[day_model[p_, da_](n - 1)]]
    ])]


@memo(cache=True)
def night_model[p: Pp](n=0):
    night: knows(p)
    night: wants(payoff=one_night_payoff(na) +
                 (0 if n <= 0 else E[night_model[1][p](n - 1) if day.da_ == {DayAction.GO} else 0]))
    night: chooses(sigma in Pp, to_maximize=EU[payoff])
    night: chooses(na in NightAction, wpp=ber(na, sigma))
    night: thinks[
        day: knows(p, sigma),
        day: thinks[
            night: knows(p, sigma),
            night: given(t in NightType, wpp=ber(t, p)),
            night: chooses(na in NightAction,
                           wpp=(na == {NightAction.Retire}) if t == {NightType.Responsible} else ber(na, sigma))
        ],
        day: observes [night.na] is na,
        day: chooses(p_ in Pp, to_minimize=abs(p_ - Pr[night.t == {NightType.Fun}])),
        day: chooses(da_ in DayAction, to_maximize=1 if n <= 0 else day_model[p_, da_](n - 1))
    ]
    return E[night.sigma]
    return E[night[EU[payoff]]]


def kl_fail_unknown():
    alice: chooses(p in X, wpp=1)
    alice: chooses(q in X, wpp=1)
    return KL[alice.r | alice.q]


def kl_fail_known[r: X]():
    alice: knows(r)
    alice: chooses(q in X, wpp=1)
    return KL[alice.r | alice.q]


def kl_fail_dom():
    alice: chooses(p in X, wpp=1)
    alice: chooses(q in Coin, wpp=1)
    return KL[alice.p | alice.q]


def entropy_unknown():
    return alice[H[bob.x]]


def entropy_known():
    bob: chooses(x in X, wpp=1)
    alice: knows(bob.x)
    return alice[H[bob.x]]


def forgets_scope_parent():
    alice: chooses(x in X, wpp=1)
    alice: forgets_about(x)
    return E[alice.x]


def forgets_scope_self():
    alice: chooses(x in X, wpp=1)
    alice: forgets_about(x)
    alice: chooses(y in X, wpp=x)
    return 0


def forgets_not_own[x: X]():
    alice: knows(x)
    alice: forgets_about(x)
    return 0


def cast_err():
    cast: [alice]
    alice: chooses(x in X, wpp=1)
    bob: chooses(y in X, wpp=1)
    return 1


def exotic_misused(arr: ...):
    return arr


write("slice6.edn",
      [case(kl),
       case(kl_victor),
       case(entropy_uniform),
       case(entropy_joint, 0.4),
       case(entropy_marginal),
       case(eig),
       case(forgets_about_1),
       case(forgets_about_multiple),
       case(with_cast),
       case(exotic_param),
       case(exotic_param_kwarg),
       case(exotic_kwarg),
       case(day_model, 0),
       case(night_model, 0),
       case(day_model, 1),
       case(night_model, 1)],
      [kl_fail_unknown, kl_fail_known, kl_fail_dom, entropy_unknown, entropy_known,
       forgets_scope_parent, forgets_scope_self, forgets_not_own, cast_err, exotic_misused],
      grads=[grad(entropy_joint, 1.0, 0.4)])


# --- the first acceptance pass's findings (bean memoclj-ai05) ---------------------
#
# Each model is named for its finding (t1_... for T1); memo.acceptance-test
# holds the twins, and memo.clj's own checks where Python memo is wrong.

X123 = np.array([1, 2, 3])
Signed = np.array([-2, -1, 0, 1, 2])


# T1: ** on literals and parameters is Python's (they are Python numbers)
@memo
def t1_lit_neg():
    return 2 ** -2


@memo
def t1_lit_pos():
    return 2 ** 3


@memo
def t1_param_neg(a):
    return a ** -1


@memo
def t1_param_exp(a):
    return 2 ** a


@memo
def t1_float_exp[x: X123]():
    return x ** -1.0


@memo
def t1_int_pos[x: X123]():
    return x ** 2


@memo
def t1_zero_neg():
    return 0 ** -1


# T2: the gradient of abs at 0 is JAX's, 1
@memo
def t2_abs(mu):
    return abs(mu)


@memo
def t2_near(mu):
    alice: chooses(x in X, wpp=exp(-abs(x - mu)))
    return E[alice.x]


@memo
def t2_abs_int[s: Signed]():
    return abs(s)


# T3: a member with an earlier member's value is its alias (IntEnum)
class Dupe(IntEnum):
    P = 1
    Q = 1
    R = 2


@memo
def t3_dupe_mean():
    alice: chooses(x in Dupe, uniformly)
    return E[alice.x]


@memo
def t3_dupe_to_be[x: Dupe]():
    alice: chooses(y in Dupe, to_be={Dupe.Q})
    return Pr[alice.y == x]


# T4: a host function's Ratio or big integer is a Python number (twins:
# Clojure's (/ v 2) gives a Ratio, (* v 3N) a BigInt)
def t4_half(v):
    return v / 2


def t4_triple(v):
    return v * 3


@memo
def t4_host_ratio[x: X]():
    return t4_half(x)


@memo
def t4_static_ratio():
    return t4_half(3)


@memo
def t4_inline_ratio():
    return {3 / 2}


@memo
def t4_host_bigint[x: X]():
    return t4_triple(x)


write("acceptance.edn",
      [case(t1_lit_neg),
       case(t1_lit_pos),
       case(t1_param_neg, 2),
       case(t1_param_neg, 2.0),
       case(t1_param_exp, -1),
       case(t1_param_exp, 3),
       case(t1_float_exp),
       case(t1_int_pos),
       case(t2_near, 2.0),
       case(t2_abs_int),
       case(t3_dupe_mean),
       case(t3_dupe_to_be),
       case(t4_host_ratio),
       case(t4_static_ratio),
       case(t4_inline_ratio),
       case(t4_host_bigint)],
      [],
      runtime_errors=[(t1_zero_neg,)],
      grads=[grad(t2_abs, 1.0, 0.0),
             grad(t2_abs, 1.0, -0.0),
             grad(t2_near, 1.0, 2.0),
             grad(t2_near, 1.0, 0.0)])


# --- the parity corpus: memo's own test suite, demo/test.py, case by case ---------
#
# test.py is run as it is, with memo_test replaced by a recorder, so the
# outcomes are Python memo's own. One change, recorded in the fixture:
# post_optim_distinctness ranges over 30 values, not Z's 1000. Its query
# repeats z, so Python memo's posterior shortcut does not apply and it builds
# the z x z1 x z2 table: 4 GB at float32 (Python memo, memo.clj on MLX) and
# more than 14 GB at float64 on gen.array's JVM backend, which fuses nothing.
# post_optim keeps Z: built naively its table would have 10^12 cells, so it
# checks that memo.clj takes the shortcut too.

import runpy  # noqa: E402

TESTPY_RECORDS = []


def _value_record(g, out, full=4096):
    shape = tuple(g._shape) if g._num_retvals == 1 else (g._num_retvals,) + tuple(g._shape)
    values = np.broadcast_to(np.asarray(out), shape)
    rec = {Kw("shape"): list(shape)}
    if values.size > full:
        flat = np.ravel(values)
        idx = np.linspace(0, flat.size - 1, 64).astype(int)
        rec[Kw("digest")] = {Kw("sum"): float(flat.astype(np.float64).sum()),
                             Kw("samples"): [[int(i), float(flat[i])] for i in idx]}
    else:
        rec[Kw("value")] = tolist(values)
    return rec


def recording_memo_test(mod, expect='pass', item=None, *args, **kwargs):
    def helper(f):
        name = f.__name__
        rec = {Kw("name"): kebab(name), Kw("expect"): expect,
               Kw("args"): [number(a) for a in args]}
        if kwargs.get("variant") is not None:
            rec[Kw("variant")] = kwargs["variant"].name
        if item is not None:
            rec[Kw("item")] = float(item)
        stage = "compile"
        try:
            memo_pkg.memo(f, install_module=mod.install, **kwargs)
            g = mod.__getattribute__(name)
            stage = "call"
            out = g(*args)
        except MemoError as e:
            rec.update({Kw("outcome"): "ce", Kw("stage"): stage, Kw("message"): e.message})
        except Exception as e:
            rec.update({Kw("outcome"): "re", Kw("stage"): stage,
                        Kw("message"): f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"})
        else:
            rec.update({Kw("outcome"): "pass"})
            rec.update(_value_record(g, out))
        TESTPY_RECORDS.append(rec)
        return f
    return helper


def run_testpy():
    import tempfile
    src = open(os.path.join(MEMO_SRC, "demo", "test.py")).read()
    old = ("def post_optim_distinctness[z: Z]():\n"
           "    alice: chooses(z1 in Z, wpp=1)\n"
           "    alice: chooses(z2 in Z, wpp=1)\n")
    assert src.count(old) == 1
    src = src.replace(old, old.replace(" Z", " Z30"))
    src = src.replace("Z = np.arange(1000)", "Z = np.arange(1000)\nZ30 = np.arange(30)")
    memo_pkg.memo_test = recording_memo_test
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memo_test_suite.py")
        with open(path, "w") as out:
            out.write(src)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(path, run_name="memo_test_suite")
    out_path = os.path.join(ROOT, "test", "fixtures", "testpy.edn")
    with open(out_path, "w") as out:
        out.write(";; Generated by test/fixtures/generate.py from memo's demo/test.py: do not edit.\n")
        out.write(edn({Kw("oracle"): oracle(), Kw("changes"): ["post_optim_distinctness over Z30 = np.arange(30), not Z"],
                       Kw("cases"): TESTPY_RECORDS}) + "\n")
    print(f"wrote {out_path}: {len(TESTPY_RECORDS)} cases")


run_testpy()


# --- the parity corpus: the Handbook, entry by entry -----------------------------
#
# Every code block of memo's Handbook.md, and every row of its tables that
# starts with code, is an entry: "heading/k" for the k-th block under a
# heading, "heading/code" for a row. handbook.py holds the twins, each
# recording the entries it ports; memo.handbook-test accounts for the rest.

import contextlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402


def slug(heading):
    return re.sub(r"[^a-z0-9]+", "-", heading.replace("`", "").lower()).strip("-")


def handbook_entries(path):
    entries, counts, heading, fence = [], {}, None, None
    for line in open(path, encoding="utf-8"):
        if line.startswith("```"):
            if fence is None:
                fence = []
            else:
                counts[heading] = counts.get(heading, 0) + 1
                entries.append({Kw("id"): f"{heading}/{counts[heading]}", Kw("text"): "".join(fence)})
                fence = None
        elif fence is not None:
            fence.append(line)
        elif m := re.match(r"#{1,3} (.*)", line):
            heading = slug(m.group(1))
        elif m := re.match(r"\| `([^`]*)`", line):
            entries.append({Kw("id"): f"{heading}/{m.group(1)}", Kw("text"): line.strip()})
    return entries


HANDBOOK_RECORDS = []


def hb_call(*args, **kwargs):
    return ("call", args, kwargs)


def hb_table(*args, **kwargs):
    return ("table", args, kwargs)


def hb_grads(*args, argnums=0):
    return ("grads", args, argnums)


def _error_text(e):
    return e.message if isinstance(e, MemoError) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def hb_twin(ids, *calls, **memo_opts):
    ids = [ids] if isinstance(ids, str) else ids

    def helper(f):
        rec = {Kw("name"): kebab(f.__name__), Kw("entries"): ids}
        if "variant" in memo_opts:
            rec[Kw("variant")] = memo_opts["variant"].name
        if memo_opts.get("cache"):
            rec[Kw("cache")] = True
        def install(src):  # into handbook.py's namespace: memo's own, a frame up, is generate.py's
            exec(src, f.__globals__)
            return f.__globals__
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                memo(f, install_module=install, **memo_opts)
            g = f.__globals__[f.__name__]
        except Exception as e:
            rec.update({Kw("outcome"): "ce", Kw("message"): _error_text(e)})
            HANDBOOK_RECORDS.append(rec)
            return f
        if out.getvalue():
            rec[Kw("compile-output")] = out.getvalue()
        rec[Kw("outcome")] = "compiled"
        if g.__doc__:
            rec[Kw("doc")] = g.__doc__
        rec[Kw("calls")] = []
        for kind, args, extra in calls:
            c = {Kw("kind"): kind, Kw("args"): [number(a) for a in args]}
            try:
                if kind == "grads":
                    c[Kw("argnums")] = list(extra) if isinstance(extra, tuple) else extra
                    v, gr = jax.value_and_grad(g, argnums=extra)(*args)
                    c.update({Kw("value"): float(v),
                              Kw("grad"): [float(x) for x in gr] if isinstance(gr, tuple) else float(gr)})
                else:
                    c[Kw("kwargs")] = {Kw(k): number(v) for k, v in extra.items()}
                    with contextlib.redirect_stdout(io.StringIO()) as out:
                        r = g(*args, **extra, print_table=(kind == "table"))
                    if kind == "table":
                        c[Kw("table")] = out.getvalue()
                    c.update(_value_record(g, r))
            except Exception as e:
                c[Kw("error")] = _error_text(e)
            rec[Kw("calls")].append(c)
        HANDBOOK_RECORDS.append(rec)
        return g
    return helper


def run_handbook():
    path = os.path.join(ROOT, "test", "fixtures", "handbook.py")
    runpy.run_path(path, run_name="memo_handbook",
                   init_globals={"twin": hb_twin, "call": hb_call, "table": hb_table, "grads": hb_grads})
    entries = handbook_entries(os.path.join(MEMO_SRC, "Handbook.md"))
    ids = {e[Kw("id")] for e in entries}
    unknown = {i for r in HANDBOOK_RECORDS for i in r[Kw("entries")]} - ids
    assert not unknown, f"twins port entries the Handbook does not have: {unknown}"
    out_path = os.path.join(ROOT, "test", "fixtures", "handbook.edn")
    with open(out_path, "w") as out:
        out.write(";; Generated by test/fixtures/generate.py from memo's Handbook.md and handbook.py: do not edit.\n")
        out.write(edn({Kw("oracle"): oracle(), Kw("entries"): entries, Kw("twins"): HANDBOOK_RECORDS}) + "\n")
    print(f"wrote {out_path}: {len(entries)} entries, {len(HANDBOOK_RECORDS)} twins")


run_handbook()


# --- the parity corpus: memo's 36 demos, as they are --------------------------------
#
# Each demo runs as written (a notebook's code cells in order, IPython magics
# removed, matplotlib headless), with memo replaced by a recorder: each model
# compiles as memo compiles it, and its calls from the demo's own code (not
# from other models, not under jax transformations) are recorded: the first
# DEMO_CALLS - 1 distinct ones and the latest. A model defined again under
# its name is recorded as name-2, name-3, ... demos/<demo>.edn holds what each demo's
# models computed; memo.demos.<demo>-test has their Clojure twins.
# MEMO_DEMOS=demo-rsa,demo-fib limits a run to those demos.

import time  # noqa: E402
import json as _json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402

DEMO_CALLS = 4
DEMO_FULL = 16384  # values recorded in full up to this many; a digest beyond
DEMO_DIR = os.path.join(MEMO_SRC, "demo")


def demo_source(path):
    """A demo as one Python script: a notebook's code cells, IPython syntax removed."""
    if not path.endswith(".ipynb"):
        return open(path, encoding="utf-8").read()
    cells = [("".join(c["source"])) for c in _json.load(open(path, encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    out = []
    for cell in cells:
        if cell.lstrip().startswith("%%"):
            magic, _, body = cell.lstrip().partition("\n")
            if not magic.startswith(("%%time", "%%timeit")):  # bash, html, ...: not Python
                continue
            cell = body  # the timed cell itself
        for line in cell.splitlines():
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            if stripped.startswith(("!", "%matplotlib", "%load_ext", "%config")):
                continue
            if stripped.startswith(("%time ", "%timeit ")):
                words = stripped.split(" ")[1:]
                while words and words[0].startswith("-"):
                    words = words[2:] if words[0] in ("-r", "-n") else words[1:]
                line = indent + " ".join(words)
            out.append(line)
        out.append("")
    return "\n".join(out)


def _concrete(x):
    if isinstance(x, jax.core.Tracer):
        return False
    if isinstance(x, (list, tuple)):
        return all(_concrete(v) for v in x)
    return True


def _arg(a):
    if isinstance(a, bool):
        return a
    if isinstance(a, int):
        return a
    a_ = np.asarray(a)
    return float(a_) if a_.ndim == 0 else tolist(a_)


def recording_memo(models):
    original = memo_pkg.memo

    def wrap(g):
        name = kebab(g.__name__)
        n = 1 + sum(1 for m in models if m[Kw("defined-as")] == name)
        rec = {Kw("name"): Kw(f"{name}-{n}") if n > 1 else name, Kw("defined-as"): name,
               Kw("calls"): [], Kw("ncalls"): 0}
        models.append(rec)
        seen = set()

        def wrapper(*args, **kwargs):
            out = g(*args, **kwargs)
            if ("return_aux" in kwargs or "return_cost" in kwargs  # a model's call, or cost @
                    or not _concrete(list(args) + list(kwargs.values()))):
                return out
            opts = {k: v for k, v in kwargs.items() if k in ("print_table", "return_pandas", "return_xarray")}
            kw = {k: v for k, v in kwargs.items() if k not in opts}
            call = {Kw("args"): [_arg(a) for a in args], Kw("kwargs"): {Kw(k): _arg(v) for k, v in kw.items()}}
            key = repr(call)
            rec[Kw("ncalls")] += 1
            if key in seen:
                return out
            seen.add(key)
            call.update(_value_record(g, out[0] if isinstance(out, tuple) else out, full=DEMO_FULL))
            calls = rec[Kw("calls")]
            if len(calls) == DEMO_CALLS:
                calls.pop()
            calls.append(call)
            return out

        wrapper.__dict__.update(g.__dict__)
        for attr in ("cache_clear", "cache_info"):  # cache=True
            if hasattr(g, attr):
                setattr(wrapper, attr, getattr(g, attr))
        wrapper.__name__, wrapper.__qualname__, wrapper.__doc__ = g.__name__, g.__qualname__, g.__doc__
        return wrapper

    def memo_(f=None, **opts):
        if f is None:
            return lambda f: memo_(f, **opts)

        def install(src):  # as memo's own exec, into the defining module's globals
            scope = {}
            exec(src, f.__globals__, scope)
            return scope
        holder = {}
        theirs = opts.pop("install_module", None)

        def capture(src):
            scope = theirs(src) if theirs else install(src)
            holder.update({f.__name__: scope[f.__name__], "__module__": scope})
            return holder
        try:
            original(f, install_module=capture, **opts)
        except MemoError as e:  # the demo's own (some demos show that a model must not compile)
            models.append({Kw("name"): kebab(f.__name__), Kw("defined-as"): kebab(f.__name__),
                           Kw("outcome"): "ce", Kw("message"): e.message})
            raise
        w = wrap(holder[f.__name__])
        if theirs:  # the demo's own module (make_module) calls it from there
            holder["__module__"][f.__name__] = w
        return w
    return memo_


def _stand_ins():
    """Modules some demos import for their notebook UI, which the recorder
    replaces: tqdm (a progress bar) returns its iterable; ipywidgets'
    interact calls the function once, with each widget's initial value (an
    iterable's first element, a (min, max) range's midpoint, a number as is)."""
    import types
    tq = types.ModuleType("tqdm")
    tq.tqdm = lambda it=None, *a, **k: it if it is not None else []
    tq.trange = lambda *a, **k: range(*a)
    for sub in ("auto", "notebook"):
        m = types.ModuleType(f"tqdm.{sub}")
        m.tqdm, m.trange = tq.tqdm, tq.trange
        setattr(tq, sub, m)
        sys.modules[f"tqdm.{sub}"] = m
    sys.modules["tqdm"] = tq

    def initial(v):
        if isinstance(v, tuple) and len(v) >= 2:
            return (v[0] + v[1]) / 2
        if isinstance(v, (int, float, str, bool)):
            return v
        return next(iter(v))

    def interact(f=None, **widgets):
        if f is None:
            return lambda f: interact(f, **widgets)
        import inspect
        params = inspect.signature(f).parameters
        f(**{k: initial(widgets.get(k, p.default)) for k, p in params.items()})
        return f
    w = types.ModuleType("ipywidgets")
    w.interact = interact
    sys.modules["ipywidgets"] = w


def run_demo(fn):
    """Run one demo with the recorder: its models' records and the outcome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = plt.savefig = matplotlib.figure.Figure.savefig = lambda *a, **k: None
    _stand_ins()
    models = []
    original = memo_pkg.memo
    memo_pkg.memo = recording_memo(models)
    t0 = time.time()
    outcome = {Kw("outcome"): "ran"}
    cwd = os.getcwd()
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=DEMO_DIR, prefix=".rec-", delete=False,
                                     encoding="utf-8") as tmp:
        tmp.write(demo_source(os.path.join(DEMO_DIR, fn)))
    limit = sys.getrecursionlimit()
    try:
        os.chdir(DEMO_DIR)
        sys.setrecursionlimit(20000)  # demo-mdp's Q(1000) recurses past Python's default 1000
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(tmp.name, run_name="__main__")
    except BaseException as e:  # a demo that stops part way still records what ran
        tb = traceback.extract_tb(e.__traceback__)
        line = next((fr.lineno for fr in reversed(tb) if fr.filename == tmp.name), None)
        outcome = {Kw("outcome"): "stopped", Kw("error"): _error_text(e), Kw("line"): line}
    finally:
        sys.setrecursionlimit(limit)
        os.chdir(cwd)
        os.unlink(tmp.name)
        memo_pkg.memo = original
        plt.close("all")
    outcome[Kw("seconds")] = round(time.time() - t0, 1)
    return models, outcome


def run_demos():
    only = [d for d in os.environ.get("MEMO_DEMOS", "").split(",") if d]
    names = sorted(fn for fn in os.listdir(DEMO_DIR)
                   if (fn.startswith("demo-") or fn == "Memonomicon.ipynb") and fn.endswith((".py", ".ipynb")))
    out_dir = os.path.join(ROOT, "test", "fixtures", "demos")
    os.makedirs(out_dir, exist_ok=True)
    for fn in names:
        demo = fn.rsplit(".", 1)[0]
        if only and demo not in only:
            continue
        models, outcome = run_demo(fn)
        path = os.path.join(out_dir, f"{demo}.edn")
        with open(path, "w") as out:
            out.write(f";; Generated by test/fixtures/generate.py from memo's demo/{fn}: do not edit.\n")
            out.write(edn({Kw("oracle"): oracle(), Kw("demo"): fn, **outcome, Kw("models"): models}) + "\n")
        print(f"wrote {path}: {len(models)} models, {sum(len(m.get(Kw('calls'), [])) for m in models)} calls, "
              f"{outcome[Kw('outcome')]} in {outcome[Kw('seconds')]} s"
              + (f" ({outcome[Kw('error')]}, line {outcome[Kw('line')]})" if Kw("error") in outcome else ""))


run_demos()
