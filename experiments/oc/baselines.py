"""Overcooked baselines.

  oracle:  GreedyHumanModel from overcooked_ai — perfect-information rollout
           via `OriginalAdapter.run_oracle_episode()`. Deterministic.

  random:  pick uniform-random low-level action per agent per step.
           Lower bound (no coordination at all).

  scripted (deliberately omitted): we wanted to add "one agent always cooks,
           other always delivers" but that requires layout-specific scripts;
           leaving it out keeps the baseline set platform-agnostic.

Both return dicts matching the bilevel_llm driver: {score, deliveries, steps}.
"""
import random

from scenarios.overcooked.common import Action
from scenarios.overcooked.config import OvercookedConfig


def run_random_episode(config: OvercookedConfig, seed: int = 0) -> dict:
    """Each step: each agent picks a uniform-random low-level action.
    No coordination, no awareness. Useful as a sanity-check lower bound."""
    from scenarios.overcooked.original_adapter import OriginalAdapter

    rng = random.Random(seed)
    adapter = OriginalAdapter(config)
    adapter.reset()

    actions_pool = list(Action)
    deliveries = 0
    for _ in range(config.max_steps):
        actions = {
            i: int(rng.choice(actions_pool)) for i in range(config.n_agents)
        }
        _, events = adapter.step(actions)
        if events.delivery:
            deliveries += 1
        if adapter.get_state().done:
            break

    state = adapter.get_state()
    return {
        "score": state.score,
        "deliveries": deliveries,
        "steps": state.step,
        "max_steps": config.max_steps,
    }


def run_oracle_episode(config: OvercookedConfig) -> dict:
    """Wraps OriginalAdapter.run_oracle_episode() for uniform API."""
    from scenarios.overcooked.original_adapter import OriginalAdapter

    adapter = OriginalAdapter(config)
    result = adapter.run_oracle_episode()
    return {
        "score": result["score"],
        "deliveries": result["deliveries"],
        "steps": result["steps"],
        "max_steps": config.max_steps,
    }
