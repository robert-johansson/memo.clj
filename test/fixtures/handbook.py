"""Twins of the memo Handbook's examples (memo-mcp/memo/Handbook.md).

Run by generate.py, which injects the recorder: `twin(ids, *calls, **memo
options)` compiles a model with memo, runs each call (`call(...)`,
`table(...)` for print_table=True, `grads(...)` for jax.value_and_grad) and
records the outcomes against the Handbook entries it ports (a code block,
"heading/k", or a row of a table, "heading/code"). A twin's ids are the
entries it shows working; a model the Handbook writes as a fragment is
completed as little as possible. Each twin has a Clojure twin of the same
name in memo.handbook-test.
"""

from enum import IntEnum

import jax
import jax.numpy as np
from jax.scipy.stats.bernoulli import pmf as ber_pmf
from memo import Variant

X = np.arange(3)
Y = np.arange(3)
N = np.arange(4)
A = np.arange(3)
Bias = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
Signed = np.array([-2, -1, 0, 1, 2])


class Action(IntEnum):
    WAIT = 0
    JUMP = 1
    RUN = 2


Actions = Action


class Roles(IntEnum):
    HOST = 0
    GUEST = 1
    COOK = 2


class Stores(IntEnum):
    BAKERY = 0
    BANK = 1
    CAFE = 2


class Bool(IntEnum):
    FALSE = 0
    TRUE = 1


class Loc(IntEnum):
    BASKET = 0
    BOX = 1


@jax.jit
def utility(a):
    return np.array([0.0, 1.0, 3.0])[a]


@jax.jit
def joint(x, y):
    return 1.0 + x * y


@jax.jit
def is_open(s):
    return s != 1


@jax.jit
def f(x):
    return np.cos(x)


@jax.jit
def g2(x, y):
    return np.exp(-(x - y) ** 2)


# --- Anatomy of a memo ------------------------------------------------------

@twin(["anatomy-of-a-memo/1", "anatomy-of-a-memo/model", "anatomy-of-a-memo/[x: X, y: Y]",
       "anatomy-of-a-memo/(a, b, c=3.14)", "anatomy-of-a-memo/''' ... '''",
       "anatomy-of-a-memo/alice: …", "anatomy-of-a-memo/return …"],
      call(0.5, -0.3), call(0.5, -0.3, 1.0), call(0.5, -0.3, c=2.0), call(a=0.5, b=-0.3))
def model[x: X, y: Y](a, b, c=3.14):
    ''' This is a model of ____! '''
    alice: chooses(x in X, wpp=exp(a * x))
    bob: chooses(y in Y, wpp=exp(b * y * c))
    return Pr[alice.x == x, bob.y == y]


# --- chooses -----------------------------------------------------------------

@twin("chooses/1", call(1.5), call(0.0))
def chooses_softmax[a: Actions](β):
    bob: chooses(a in Actions, wpp=exp(β*utility(a)))
    return Pr[bob.a == a]


@twin("chooses/2", call())
def chooses_bernoulli[x: Bool]():
    alice: chooses(x in Bool, wpp=ber_pmf(x, 0.9))
    return Pr[alice.x == x]


@twin("making-multiple-choices-simultaneously/1", call())
def chooses_joint[x: X, y: Y]():
    bob: chooses(x in X, y in Y, wpp=joint(x, y))
    return Pr[bob.x == x, bob.y == y]


@twin("deterministic-choice-with-to-maximize-and-to-minimize/1", call())
def chooses_to_maximize[a: Actions]():
    bob: chooses(a in Actions, to_maximize=utility(a))
    return Pr[bob.a == a]


@twin("deterministic-choice-with-to-maximize-and-to-minimize/1", call())
def chooses_to_minimize[a: Actions]():
    bob: chooses(a in Actions, to_minimize=utility(a))
    return Pr[bob.a == a]


@twin("deterministic-value-with-to-be/1", call())
def chooses_to_be[a: Actions]():
    bob: chooses(a in Actions, to_be={Action.JUMP})
    return Pr[bob.a == a]


@twin("deterministic-value-with-to-be/2", call())
def chooses_to_be_twice[a1: Actions, a2: Actions]():
    # Bob jumps twice
    bob: chooses(a1 in Actions, a2 in Actions, to_be={Action.JUMP})
    return Pr[bob.a1 == a1, bob.a2 == a2]


@twin("aliases-for-chooses/1", call())
def alias_given[r: Roles]():
    bob: given(r in Roles, wpp=1)
    return Pr[bob.r == r]


@twin("aliases-for-chooses/1", call())
def alias_draws[r: Roles]():
    bob: draws(r in Roles, wpp=r + 1)
    return Pr[bob.r == r]


@twin("aliases-for-chooses/1", call())
def alias_assigned[r: Roles]():
    bob: assigned(r in Roles, wpp=r + 1)
    return Pr[bob.r == r]


@twin("aliases-for-chooses/1", call())
def alias_guesses[r: Roles]():
    bob: guesses(r in Roles, wpp=r + 1)
    return Pr[bob.r == r]


@twin("aliases-for-chooses/2", call())
def chooses_such_that[s: Stores]():
    bob: chooses(s in Stores, such_that=is_open(s))
    return Pr[bob.s == s]


@twin("aliases-for-chooses/3", call())
def chooses_uniformly[s: Stores]():
    bob: chooses(s in Stores, uniformly)
    return Pr[bob.s == s]


# --- thinks, observes, knows ---------------------------------------------------

@twin("thinks/1", call())
def thinks_two_agents():
    bob: thinks[
        alice: chooses(x in X, wpp=1),
        charlie: chooses(y in Y, wpp=y + 1),
    ]
    return bob[E[alice.x + charlie.y]]


@twin("observes/1", call())
def observes_axis[y: X]():
    bob: thinks[alice: chooses(x in X, wpp=1), alice: chooses(z in X, wpp=1 + 2 * (x == z))]
    bob: observes [alice.x] is y
    return bob[E[alice.z]]


@twin("observes/1")
def observes_own_choice():
    # the Handbook: y may be "the observer's own choice"; a bare y names the
    # enclosing frame's choices, so bob's own is bob.y (the next twin)
    bob: thinks[alice: chooses(x in X, wpp=1), alice: chooses(z in X, wpp=1 + 2 * (x == z))]
    bob: chooses(y in X, wpp=y + 1)
    bob: observes [alice.x] is y
    return E[bob[E[alice.z]]]


@twin("observes/1", call())
def observes_own_choice_dotted():
    bob: thinks[alice: chooses(x in X, wpp=1), alice: chooses(z in X, wpp=1 + 2 * (x == z))]
    bob: chooses(y in X, wpp=y + 1)
    bob: observes [alice.x] is bob.y
    return E[bob[E[alice.z]]]


@twin("observes/1")
def observes_literal():
    bob: thinks[alice: chooses(x in X, wpp=1)]
    bob: observes [alice.x] is 7
    return 0


@twin("observes/2", call())
def observes_other_agent():
    charlie: chooses(y in X, wpp=y + 1)
    bob: thinks[alice: chooses(x in X, wpp=1), alice: chooses(z in X, wpp=1 + 2 * (x == z))]
    bob: observes [alice.x] is charlie.y
    return E[bob[E[alice.z]]]


@twin("advanced-observation-constructs/1", call())
def observes_that_bias():
    bob: thinks[coin: chooses(bias in Bias, wpp=1)]
    bob: observes_that [coin.bias > 0.5]
    return bob[E[coin.bias]]


@twin("advanced-observation-constructs/2", call())
def observes_event_bias():
    bob: thinks[coin: chooses(bias in Bias, wpp=1)]
    bob: observes_event(wpp=coin.bias)
    return bob[E[coin.bias]]


@twin("knows/1", call())
def knows_two[x: X]():
    alice: chooses(y in Y, wpp=y + 1)
    bob: knows(x, alice.y)
    bob: chooses(z in X, wpp=1 + 2 * (z == (x + alice.y) % 3))
    return E[bob.z]


@twin("knows/2", call())
def knows_short():
    alice: chooses(y in Y, wpp=y + 1)
    bob: knows(alice.y)
    return E[bob[E[alice.y]]]


@twin("knows/2", call())
def knows_expanded():
    alice: chooses(y in Y, wpp=y + 1)
    bob: thinks[ alice: chooses(y in Y, wpp=1) ]
    bob: observes [alice.y] is alice.y
    return E[bob[E[alice.y]]]


# --- snapshots, wants, forgets_about, has_theory, INSPECT -----------------------

@twin("snapshots-self-as/1", call())
def snapshot_aliases[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: snapshots_self_as(past_alice, younger_alice)
    return alice[past_alice[E[bob.x]] + younger_alice[Pr[bob.x == 2]]]


@twin("wants-and-eu/1", call())
def wants_to_win[a: A]():
    alice: thinks[ bob: chooses(score in N, wpp=1) ]
    alice: wants(win=score > bob.score)
    alice: chooses(a in A, to_maximize=EU[win])
    alice: given(score in N, wpp=exp(-(score - a - 1) ** 2))
    return Pr[alice.a == a]


@twin("forgets-about/1", call())
def forgets_about_not_yet():
    alice: chooses(x in Bool, uniformly)
    alice: chooses(y in Bool, to_be=x)
    return E[alice.y] + Var[alice.y]


@twin("forgets-about/2", call())
def forgets_about_x():
    alice: chooses(x in Bool, uniformly)
    alice: chooses(y in Bool, to_be=x)
    alice: forgets_about(x)
    return E[alice.y] + Var[alice.y]


@twin("forgets-about/2")
def forgets_about_then_x():
    alice: chooses(x in Bool, uniformly)
    alice: chooses(y in Bool, to_be=x)
    alice: forgets_about(x)
    return E[alice.x]


@twin("forgets-about/2", call())
def forgets_about_three():
    alice: chooses(x in Bool, uniformly)
    alice: chooses(y in Bool, wpp=1 + x)
    alice: chooses(z in Bool, wpp=1 + y)
    alice: chooses(w in Bool, to_be=z)
    alice: forgets_about(x, y, z)
    return E[alice.w]


@twin("has-theory/1")
def child_thinks_of_thinking():
    child: has_theory(CHILD)
    child: thinks[
        sally: thinks[ ball: chooses(loc in Loc, wpp=1) ]
    ]
    return 0


@twin("has-theory/1", call())
def child_thinks():
    child: has_theory(CHILD)
    child: thinks[
        sally: chooses(loc in Loc, wpp=1 + loc)
    ]
    return child[E[sally.loc]]


@twin("has-theory/1", variant=Variant.CHILD)
def child_variant_thinks():
    child: thinks[
        sally: chooses(loc in Loc, wpp=1)
    ]
    return 0


@twin("has-theory/1", call(), variant=Variant.CHILD)
def child_variant_chooses():
    child: chooses(loc in Loc, wpp=1 + loc)
    return E[child.loc]


@twin("inspect/1", call())
def inspect_alice():
    alice: chooses(x in X, wpp=1)
    alice: INSPECT()
    return E[alice.x]


# --- Expressions -------------------------------------------------------------

@twin("literals-and-parameters/1", call())
def literal():
    return 3.14


@twin("literals-and-parameters/2", call(1.0, 2.0, 3.0))
def parameters(a, b, c):
    return a + b * c


@twin("binary-operators/+", call())
def op_add[x: X, y: Y]():
    return x + y


@twin("binary-operators/-", call())
def op_sub[x: X, y: Y]():
    return x - y


@twin("binary-operators/*", call())
def op_mul[x: X, y: Y]():
    return x * y


@twin("binary-operators//", call())
def op_div[x: X, y: Y]():
    return x / (y + 1)


@twin("binary-operators/**", call())
def op_pow[x: X, y: Y]():
    return x ** (y + 0.5)


@twin("binary-operators/%", call())
def op_mod[x: Signed, y: Y]():
    return x % (y + 1)


@twin("binary-operators/%", call())
def op_mod_float[x: Signed]():
    return (x + 0.5) % 2


@twin("comparison-operators/==", call())
def op_eq[x: X, y: Y]():
    return x == y


@twin("comparison-operators/!=", call())
def op_neq[x: X, y: Y]():
    return x != y


@twin("comparison-operators/<", call())
def op_lt[x: X, y: Y]():
    return x < y


@twin("comparison-operators/<=", call())
def op_lte[x: X, y: Y]():
    return x <= y


@twin("comparison-operators/>", call())
def op_gt[x: X, y: Y]():
    return x > y


@twin("comparison-operators/>=", call())
def op_gte[x: X, y: Y]():
    return x >= y


@twin("logical-operators/and", call())
def op_and[x: X, y: Y]():
    return x > 0 and y > 1


@twin("logical-operators/or", call())
def op_or[x: X, y: Y]():
    return x > 0 or y > 1


@twin("logical-operators/^", call())
def op_xor[x: X, y: Y]():
    return (x > 0) ^ (y > 1)


@twin("logical-operators/^", call())
def op_xor_int[x: X, y: Y]():
    return x ^ y


@twin("unary-operators/-x", call())
def op_neg[x: X]():
    return -x


@twin("unary-operators/+x", call())
def op_uplus[x: X]():
    return +x


@twin("unary-operators/~x", call())
def op_inv[x: X]():
    return ~(x > 1)


@twin("unary-operators/~x", call())
def op_inv_int[x: X]():
    return ~x


@twin("built-in-functions/exp(x)", call())
def fn_exp[x: X]():
    return exp(x)


@twin("built-in-functions/log(x)", call())
def fn_log[x: X]():
    return log(x + 1)


@twin("built-in-functions/abs(x)", call())
def fn_abs[x: Signed]():
    return abs(x)


@twin("conditional-expressions/1", call(1.0, 2.0))
def conditional[x: X](a, b):
    return a if x > 1 else b


@twin("calling-custom-jax-functions/1", call())
def custom_function[x: X]():
    alice: chooses(y in Y, wpp=1 + f(y))
    return E[alice.y] + f(x)


# --- choice expressions, operators, queries, imagine ----------------------------

@twin("choice-expressions/1", call())
def own_choices[x: X, y: Y]():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, wpp=g2(x, y))
    return Pr[alice.x == x, alice.y == y]


@twin("choice-expressions/2", call())
def dot_notation():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, wpp=g2(x, y))
    return E[alice.x + alice.y]


@twin("choice-expressions/3", call())
def bracket_each():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, wpp=g2(x, y))
    return E[alice[x] + alice[y]]


@twin("choice-expressions/3", call())
def bracket_whole():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, wpp=g2(x, y))
    return E[alice[x + y]]


@twin("probabilistic-operators/1", call())
def expectation():
    alice: chooses(x in X, wpp=x + 1)
    bob: chooses(z in X, wpp=1)
    return E[alice.x + bob.z]


@twin("probabilistic-operators/2", call())
def variance():
    alice: chooses(y in Signed, wpp=1 + y * y)
    return Var[alice.y * 2]


@twin("probabilistic-operators/3", call())
def probability():
    alice: chooses(y in Signed, wpp=3 + y)
    return Pr[alice.y >= 0]


@twin("probabilistic-operators/4", call())
def joint_probability():
    a: chooses(x in Signed, wpp=1)
    b: chooses(y in Signed, wpp=3 + y)
    return Pr[a.x > 0, b.y < 2]


@twin("information-theoretic-operators/1", call())
def entropy():
    alice: chooses(x in X, wpp=1)
    bob: knows(alice.x)
    bob: chooses(y in Y, wpp=1 + 3 * (y == alice.x))
    return H[alice.x, bob.y]


@twin("information-theoretic-operators/2", call())
def kl_divergence():
    alice: chooses(x in X, wpp=x + 1)
    bob: chooses(y in X, wpp=1)
    return KL[alice.x | bob.y]


@twin("queries/1", call())
def query_variance():
    alice: chooses(x in Signed, wpp=1)
    return Var[alice[abs(x) * 2]]


@twin("queries/1", call())
def query_known():
    bob: chooses(y in N, wpp=y + 1)
    alice: knows(bob.y)
    return E[alice[bob.y == 3]]


@twin("queries/1")
def query_unknown():
    alice: thinks[bob: chooses(y in N, wpp=1)]
    return alice[bob.y == 3]


@twin("hypotheticals-with-imagine/1")
def imagine_as_written():
    # alice's answer depends on the imagined bob.y, unknown to the frame
    # that imagines: the top level needs its expectation (the next twin)
    alice: thinks[
        bob: chooses(x in N, wpp=1),
        bob: chooses(y in Y, wpp=1 + 3 * (y == x % 3)),
    ]
    return imagine[
        bob: chooses(y in Y, wpp=1),
        alice: observes [bob.y] is bob.y,
        alice[Pr[bob.x == 3]]
    ]


@twin("hypotheticals-with-imagine/1", call())
def imagine_observation():
    alice: thinks[
        bob: chooses(x in N, wpp=1),
        bob: chooses(y in Y, wpp=1 + 3 * (y == x % 3)),
    ]
    return imagine[
        bob: chooses(y in Y, wpp=1),
        alice: observes [bob.y] is bob.y,
        E[alice[Pr[bob.x == 3]]]
    ]


# --- calling other memos -------------------------------------------------------

@twin("calling-other-memos/1", call(1.0, 0.0, 3.1))
def f3[x: X](a, b, c):
    alice: chooses(x in X, wpp=exp(a * x + b))
    return Pr[alice.x == x] * c


@twin("calling-other-memos/1", call())
def g_calls_f():
    alice: chooses(x in X, wpp=f3[x](1.0, 0.0, 3.1))
    return E[alice.x]


@twin("calling-other-memos/2", call(1.0, 0.0, 3.1), call(-1.0, 2.0, 0.5))
def g_forwards(a, b, c):
    alice: chooses(x in X, wpp=f3[x](...))
    return E[alice.x]


@twin("calling-other-memos/3", call(3.14))
def g2_returns[x: X, y: Y](t):
    return x + y * t
    return x * y


@twin("calling-other-memos/3", call())
def g2_first():
    alice: chooses(x in X, y in Y, wpp=g2_returns[0][x, y](3.14))
    return E[alice.x + alice.y]


# --- referencing Python variables ------------------------------------------------

@twin("referencing-python-variables/1", call())
def inline_member():
    return {Action.WAIT}


@twin("referencing-python-variables/2", call(3))
def inline_shift(x):
    return {x << 4}


@twin("referencing-python-variables/2", call())
def inline_agent():
    alice: chooses(x in X, wpp=1)
    return {alice.x}


# --- running and configuring ------------------------------------------------------

@twin(["executing-a-memo/1", "executing-a-memo/2"], call(0.5, 2.0), table(0.5, 2.0))
def executed[x: X, y: Y](a, b):
    alice: chooses(x in X, wpp=exp(a * x))
    return Pr[alice.x == x] + b * y
    return E[alice.x]


@twin("decorator-options/1", call(0.5), call(0.5), cache=True)
def cached(a):
    alice: chooses(x in X, wpp=exp(a * x))
    return E[alice.x]


@twin("automatic-differentiation/1", call(0.5, -0.2), grads(0.5, -0.2), grads(0.5, -0.2, argnums=(0, 1)))
def differentiable(a, b):
    alice: chooses(x in X, wpp=exp(a * x + b * x * x))
    return E[alice.x]
