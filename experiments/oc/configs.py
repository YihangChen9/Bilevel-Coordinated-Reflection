"""Benchmark configurations for Overcooked experiments.

We pick three layouts that each stress different aspects of coordination:

  cramped_room          — tiny shared space, frequent agent collisions.
                          Tests: spatial coordination & blocking-avoidance.
  asymmetric_advantages — each agent has natural specialty (one side near pots,
                          one near delivery). Tests: role-assignment quality.
  centre_pots           — pots in the centre, agents approach from both sides.
                          Tests: shared-resource contention.

Note: hard-coordination layouts (coordination_ring, forced_coordination,
bottleneck, counter_circuit) were considered but excluded — the GreedyHumanModel
oracle fails to deliver anything on them even at 400 steps, so there is no
meaningful baseline to compare against.

Budget: max_steps=200 (small enough to keep LLM cost reasonable, large enough
for ≥3 deliveries on each layout under oracle play).
"""
from dataclasses import dataclass

from scenarios.overcooked.config import OvercookedConfig


@dataclass(frozen=True)
class OCExpConfig:
    name: str
    config: OvercookedConfig
    random_seeds: tuple[int, ...]      # for random/non-LLM baselines
    llm_seeds: tuple[int, ...] = (42,) # single seed for LLM (cost)
    description: str = ""


CONFIGS: list[OCExpConfig] = [
    OCExpConfig(
        name="cramped_room",
        config=OvercookedConfig(
            layout="cramped_room",
            max_steps=200,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=25,
        ),
        random_seeds=(0, 1, 2),
        description="Tiny kitchen, collisions matter most.",
    ),
    OCExpConfig(
        name="asymmetric_advantages",
        config=OvercookedConfig(
            layout="asymmetric_advantages",
            max_steps=200,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=25,
        ),
        random_seeds=(0, 1, 2),
        description="Each agent has natural specialty — role assignment matters.",
    ),
    OCExpConfig(
        name="centre_pots",
        config=OvercookedConfig(
            layout="centre_pots",
            max_steps=200,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=25,
        ),
        random_seeds=(0, 1, 2),
        description="Central pots, shared-resource contention.",
    ),
    # ── Hard-coordination layouts (added 2026-07-23) ──
    # These were originally excluded because the GreedyHumanModel *oracle*
    # can't achieve ≥3 deliveries on them (that's exactly WHY they're hard:
    # naive greedy play deadlocks at the bottleneck). We include them anyway —
    # the comparison of interest is bilevel-LLM vs random, and the gate on/off
    # ablation; the greedy "oracle" is reported as-is (may be low/zero) and is
    # not the baseline we lean on for these.
    OCExpConfig(
        name="coordination_ring",
        config=OvercookedConfig(
            layout="coordination_ring",
            max_steps=400,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=30,
        ),
        random_seeds=(0, 1, 2),
        description="Ring topology — agents must rotate the same direction or deadlock.",
    ),
    OCExpConfig(
        name="forced_coordination",
        config=OvercookedConfig(
            layout="forced_coordination",
            max_steps=400,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=30,
        ),
        random_seeds=(0, 1, 2),
        description="Split kitchen — neither agent can deliver alone; hand-off is mandatory.",
    ),
    OCExpConfig(
        name="counter_circuit",
        config=OvercookedConfig(
            layout="counter_circuit_o_1order",
            max_steps=400,
            n_agents=2,
            cook_time=20,
            recipe_size=3,
            stuck_threshold=30,
        ),
        random_seeds=(0, 1, 2),
        description="Long counter loop — passing over the counter beats walking around.",
    ),
]


def get_config(name: str) -> OCExpConfig:
    for c in CONFIGS:
        if c.name == name:
            return c
    raise KeyError(f"unknown config {name!r}. Available: {[c.name for c in CONFIGS]}")
