"""Resource Contest game numerics.

Linear payoff: reward = Σ τᵢ · xᵢ, with τ on the simplex Σ τ = C and
xᵢ ∈ {0, …, x_max} clipped silently to the hidden cap Mᵢ.
"""
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameParams:
    n_agents: int
    max_actions: tuple[int, ...]  # Mᵢ at t=0 (hidden from orchestrator + agent)
    budget: float = 1.0           # C
    x_max: int = 10               # visible input upper bound
    # Optional non-stationary schedule: list of (t_start, new_max_actions).
    # Each entry replaces max_actions from that round onward. None = stationary.
    cap_schedule: tuple[tuple[int, tuple[int, ...]], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.max_actions) != self.n_agents:
            raise ValueError(
                f"max_actions length {len(self.max_actions)} != n_agents {self.n_agents}"
            )
        if self.cap_schedule is not None:
            for t_start, caps in self.cap_schedule:
                if len(caps) != self.n_agents:
                    raise ValueError(
                        f"cap_schedule entry at t={t_start} has {len(caps)} caps "
                        f"but n_agents={self.n_agents}"
                    )

    def caps_at(self, t: int) -> tuple[int, ...]:
        """Return Mᵢ vector active at round t (handles non-stationary schedule)."""
        active = self.max_actions
        if self.cap_schedule:
            for t_start, caps in self.cap_schedule:
                if t >= t_start:
                    active = caps
        return active


def reward(tau: list[float], x_effective: list[int]) -> float:
    """Per-round reward Σ τᵢ · xᵢ, where xᵢ is the env-clipped effective action."""
    return sum(t * x for t, x in zip(tau, x_effective))


def optimal_reward(params: GameParams, t: int = 0) -> float:
    """R*_t = C · max(Mᵢ(t)). The bandit's lower bound on regret at round t."""
    return params.budget * max(params.caps_at(t))


def regret(reward_t: float, params: GameParams, t: int = 0) -> float:
    """Per-round regret at round t. Always ≥ 0."""
    return max(0.0, optimal_reward(params, t) - reward_t)


def clip_to_cap(x_input: int, cap: int) -> tuple[int, bool]:
    """Soft-reject clip: returns (effective_x, was_clipped)."""
    if x_input < 0:
        return 0, True
    if x_input > cap:
        return cap, True
    return x_input, False


_ACTION_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_action(text: str, lo: int = 0, hi: int = 10) -> int | None:
    """Extract first integer from a worker's final answer. Float-tolerant, clamped."""
    if text is None:
        return None
    s = str(text).strip().strip('"').strip("'")
    m = _ACTION_RE.search(s)
    if not m:
        return None
    val = round(float(m.group(0)))
    return max(lo, min(hi, val))
