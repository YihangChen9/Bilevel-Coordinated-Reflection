"""Resource Contest baselines.

All baselines share the same env semantics (linear payoff, soft-clip to Mᵢ).
The lower level is an oracle agent (`xᵢ = Mᵢ`) — this isolates the upper-level
allocation signal so the comparison is apples-to-apples.

Methods:
- uniform:     τᵢ = C/N every round.            cumulative regret ~ linear in T.
- random:      τ ~ Dirichlet(1ⁿ), renormalized to C.   same expectation as uniform, higher variance.
- eps_greedy:  estimate M̂ᵢ from observed xᵢ; explore w.p. ε, else argmax M̂ᵢ.  sub-linear.
- oracle:      τ = C on argmax Mᵢ.            cumulative regret = 0 (lower bound).

Returns row dicts identical in shape to the LLM driver, so plots line up.
"""
import random
from dataclasses import dataclass
from typing import Callable

from scenarios.resource_contest.game import (
    GameParams, reward, optimal_reward, regret, clip_to_cap,
)


# ---- Lower-level (oracle agent: always plays its true cap) -----------------

def oracle_agent_action(cap: int, x_max: int) -> int:
    """An agent that knows its own cap and plays it. The interesting learning
    is at the upper level; this isolates that signal."""
    return min(cap, x_max)


# ---- Upper-level policies ---------------------------------------------------

UpperPolicy = Callable[[int, list[dict], GameParams, random.Random], list[float]]
"""Signature: (t, history, params, rng) -> tau vector summing to params.budget."""


def uniform_policy(t: int, history: list[dict], params: GameParams, rng: random.Random) -> list[float]:
    return [params.budget / params.n_agents] * params.n_agents


def random_policy(t: int, history: list[dict], params: GameParams, rng: random.Random) -> list[float]:
    raw = [rng.gammavariate(1.0, 1.0) for _ in range(params.n_agents)]
    s = sum(raw) or 1.0
    return [v * params.budget / s for v in raw]


def eps_greedy_policy(epsilon: float = 0.1, explore_rounds: int = 1) -> UpperPolicy:
    """Estimate M̂ᵢ = max observed xᵢ. Explore (uniform) for first `explore_rounds`
    rounds, then with prob ε explore uniform, else concentrate on argmax M̂."""
    def _policy(t: int, history: list[dict], params: GameParams, rng: random.Random) -> list[float]:
        if t < explore_rounds:
            return [params.budget / params.n_agents] * params.n_agents
        best = [0] * params.n_agents
        for rec in history:
            for i, xi in enumerate(rec["x_effective"][:params.n_agents]):
                if xi > best[i]:
                    best[i] = xi
        if rng.random() < epsilon:
            return [params.budget / params.n_agents] * params.n_agents
        argmax = max(range(params.n_agents), key=lambda i: best[i])
        tau = [0.0] * params.n_agents
        tau[argmax] = params.budget
        return tau
    return _policy


def oracle_policy(t: int, history: list[dict], params: GameParams, rng: random.Random) -> list[float]:
    argmax = max(range(params.n_agents), key=lambda i: params.max_actions[i])
    tau = [0.0] * params.n_agents
    tau[argmax] = params.budget
    return tau


# ---- Episode runner (no LLM) -----------------------------------------------

def run_episode_with_policy(
    params: GameParams,
    T: int,
    policy: UpperPolicy,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for t in range(T):
        caps = params.caps_at(t)
        R_star_t = optimal_reward(params, t)
        tau = policy(t, rows, params, rng)
        s = sum(max(0.0, v) for v in tau)
        if s == 0:
            tau = [params.budget / params.n_agents] * params.n_agents
        else:
            tau = [max(0.0, v) * params.budget / s for v in tau]

        x_input = [oracle_agent_action(caps[i], params.x_max)
                   for i in range(params.n_agents)]
        x_eff = [clip_to_cap(x_input[i], caps[i])[0]
                 for i in range(params.n_agents)]
        r = reward(tau, x_eff)
        g = regret(r, params, t)
        rows.append({
            "t": t, "tau": tau, "x_input": x_input, "x_effective": x_eff,
            "reward": r, "regret": g, "optimal": R_star_t,
        })
    return rows
