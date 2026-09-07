"""Benchmark configurations for Resource Contest experiments.

Each config is named and self-describing. We deliberately pick four regimes
that stress different aspects of the bilevel architecture:

  easy:    M widely separated → trivial bandit, fast convergence.
  hard:    M close together → noisy signal, needs more samples to disambiguate.
  many:    N=6 agents → larger search space, lower exploration efficiency.
  drift:   M changes at midpoint → tests adaptation to non-stationary env.

The `T` (horizon) and `seeds` are tuned per config so that:
  - non-LLM baselines run with multiple seeds for variance bars
  - LLM run is single-seed (LLM API costs); deterministic-ish under fixed prompt
"""
from dataclasses import dataclass

from scenarios.resource_contest.game import GameParams


@dataclass(frozen=True)
class ExpConfig:
    name: str
    params: GameParams
    T: int
    seeds: tuple[int, ...]          # for random / eps_greedy variance
    llm_seeds: tuple[int, ...] = (42,)  # for LLM (default single seed)
    description: str = ""


CONFIGS: list[ExpConfig] = [
    ExpConfig(
        name="easy",
        params=GameParams(
            n_agents=3,
            max_actions=(3, 5, 8),
            budget=1.0,
            x_max=10,
        ),
        T=15,
        seeds=(0, 1, 2, 3, 4),
        description="Wide cap gap [3,5,8]. Optimal R*=8. Should converge in ~2 rounds.",
    ),
    ExpConfig(
        name="hard",
        params=GameParams(
            n_agents=3,
            max_actions=(6, 7, 8),
            budget=1.0,
            x_max=10,
        ),
        T=20,
        seeds=(0, 1, 2, 3, 4),
        description="Caps close together [6,7,8]. Optimal R*=8 but uniform also yields 7. Tests precision.",
    ),
    ExpConfig(
        name="many",
        params=GameParams(
            n_agents=6,
            max_actions=(2, 4, 5, 6, 7, 9),
            budget=1.0,
            x_max=10,
        ),
        T=20,
        seeds=(0, 1, 2, 3, 4),
        description="N=6 agents, widest cap=9. More choices, larger exploration cost.",
    ),
    ExpConfig(
        name="drift",
        params=GameParams(
            n_agents=3,
            max_actions=(3, 5, 8),
            budget=1.0,
            x_max=10,
            # At t=10, the argmax shifts from agent_2 (M=8) to agent_0 (M=9).
            cap_schedule=((10, (9, 5, 4)),),
        ),
        T=20,
        seeds=(0, 1, 2, 3, 4),
        description="Non-stationary: argmax flips at t=10 (agent_2 → agent_0). Tests re-adaptation.",
    ),
]


def get_config(name: str) -> ExpConfig:
    for c in CONFIGS:
        if c.name == name:
            return c
    raise KeyError(f"unknown config {name!r}. Available: {[c.name for c in CONFIGS]}")
