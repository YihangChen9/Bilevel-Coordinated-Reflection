"""Configuration for Overcooked scenarios."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OvercookedConfig:
    layout: str = "cramped_room"
    max_steps: int = 400
    n_agents: int = 2
    cook_time: int = 20
    recipe_size: int = 3
    stuck_threshold: int = 25  # >20 (cook time) to avoid false-positive on waiting for pot
    delivery_reward: float = 20.0
    macros_per_call: int = 1
    n_planning_rounds: int = 1  # K rounds of workboard negotiation before executing
