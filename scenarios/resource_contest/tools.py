"""RC tools. The headline tool is `commit_action`:

- Each worker may call it up to MAX_ATTEMPTS times per round (binary-search the cap).
- The env silently clips x → min(x, Mᵢ); the tool tells the agent whether it
  was clipped (so they can binary-search) but NEVER the exact Mᵢ.
- The last accepted `effective_x` is what the env uses for reward.
"""
from dataclasses import dataclass, field

from memory.base import MemoryPort
from worker.tool import Tool

from scenarios.resource_contest.game import clip_to_cap


MAX_ATTEMPTS = 3


@dataclass
class AgentEnvState:
    """Per-agent per-round state: tracks attempts and the final committed x."""
    cap: int                            # Mᵢ
    attempts: int = 0
    last_effective: int = 0
    last_input: int | None = None
    history: list[tuple[int, int, bool]] = field(default_factory=list)  # (input, effective, clipped)


def make_rc_toolset_for_agent(
    workboard: MemoryPort, agent_id: int, env_state: AgentEnvState, x_max: int = 10,
) -> list[Tool]:
    section = f"agent_{agent_id}"

    def _commit_action(action: int) -> str:
        if env_state.attempts >= MAX_ATTEMPTS:
            return (
                f"rejected: agent_{agent_id} has used all {MAX_ATTEMPTS} attempts this round. "
                f"Final committed x={env_state.last_effective}."
            )
        x_in = max(0, min(int(action), x_max))
        x_eff, clipped = clip_to_cap(x_in, env_state.cap)
        env_state.attempts += 1
        env_state.last_input = x_in
        env_state.last_effective = x_eff
        env_state.history.append((x_in, x_eff, clipped))
        workboard.edit(section, f"action={x_eff} (attempt={env_state.attempts})")
        remain = MAX_ATTEMPTS - env_state.attempts
        if clipped:
            return (
                f"committed: agent_{agent_id} x_input={x_in} was CLIPPED to "
                f"effective_x={x_eff} (your cap is below {x_in}). "
                f"Attempts remaining: {remain}."
            )
        return (
            f"committed: agent_{agent_id} x_input={x_in} accepted as effective_x={x_eff} "
            f"(within your cap). Attempts remaining: {remain}."
        )

    def _submit_reflection(content: str) -> str:
        """Write a free-form reflection to the workboard under
        `agent_<i>_reflection`. The driver may snapshot and rollback this
        section if J(m_e) does not ascend (gated-write ablation)."""
        text = (content or "").strip()
        if not text:
            return "rejected: reflection is empty"
        if len(text) > 600:
            text = text[:600] + " […]"
        workboard.edit(f"{section}_reflection", text)
        return f"reflection committed for agent_{agent_id} ({len(text)} chars)"

    return [
        Tool(
            name="commit_action",
            description=(
                f"Commit integer action in [0, {x_max}] for this round. The env "
                f"silently clips to your hidden cap Mᵢ; the response tells you if "
                f"clipping happened so you can binary-search. Up to {MAX_ATTEMPTS} "
                f"calls per round; final committed value counts for reward."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "integer",
                        "description": f"Integer in [0, {x_max}].",
                        "minimum": 0,
                        "maximum": x_max,
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            func=_commit_action,
        ),
        Tool(
            name="submit_reflection",
            description=(
                "Write a short reflection to the shared workboard so other "
                "workers and the orchestrator can benefit (e.g. 'agent_2 "
                "discovered M=8'). The driver may reject this write if it does "
                "not add information (J-ascent gate)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Natural-language reflection (≤600 chars).",
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            func=_submit_reflection,
        ),
    ]
