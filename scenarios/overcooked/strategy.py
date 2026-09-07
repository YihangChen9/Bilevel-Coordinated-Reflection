"""OvercookedStrategy: assigns roles to workers based on kitchen state."""
from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING

from memory.base import MemoryPort
from orchestrator.llm import LLM
from orchestrator.orchestrator import OrchContext, Subtask
from scenarios.overcooked.renderer import render_global

if TYPE_CHECKING:
    from scenarios.overcooked.common import GameState


_STRATEGY_PROMPT = """\
You are the head chef coordinating a kitchen team in an Overcooked game.

=== DISH PIPELINE ===
One dish = 3 onions in a pot → auto-cook ~20 steps → plate picks up cooked dish → deliver to goal → +20 pts.
The workers know the detailed rules. Your job is to assign WHO does WHAT so the pipeline flows smoothly.

=== CURRENT KITCHEN STATE ===
{state_text}

=== PREVIOUS PLANS & GATED REFLECTIONS ===
(Entries tagged "[gated-write OK]" are reflections from deliveries that
strictly improved the actual-vs-optimal step gap — prefer their assignments.)
{history_block}

=== REPLAN REASON ===
{replan_reason}

You have {n_agents} workers. Assign each a role based on what the kitchen needs RIGHT NOW:
  - "gatherer": pick up onions from pile and place in pot (repeat 3 times to fill pot)
  - "deliverer": grab plate, wait for pot to finish, plate the dish, deliver to goal
  - "flexible": help wherever the bottleneck is

Think about:
  - Is the pot empty? → need a gatherer to fill it with 3 onions
  - Is the pot cooking? → gatherer can start prepping for next dish; deliverer should grab a plate and wait
  - Is the pot ready? → deliverer urgently needs to plate and deliver
  - Is an agent stuck? → reassign them to a different task

Output ONLY a JSON object:
{{"assignments": [{{"agent_id": 0, "role": "gatherer", "detail": "fill pot_0 with 3 onions"}}, ...], "reasoning": "..."}}
"""


def _extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    l, r = s.find("{"), s.rfind("}")
    if l < 0 or r < l:
        return {}
    try:
        return json.loads(s[l:r + 1])
    except json.JSONDecodeError:
        return {}


class OvercookedStrategy:
    """DecomposeStrategy implementation for Overcooked."""

    def __init__(self, llm: LLM, n_agents: int, log_memory: MemoryPort):
        self.llm = llm
        self.n_agents = n_agents
        self.log = log_memory
        self.last_state_text: str = ""

    def decompose_with_state(
        self, state_text: str, replan_reason: str, ctx: OrchContext
    ) -> list[Subtask]:
        """Called by the driver with the current rendered state."""
        self.last_state_text = state_text
        history = self.log.read()[-2000:] if self.log.read() else "(first plan)"

        prompt = _STRATEGY_PROMPT.format(
            state_text=state_text,
            history_block=history,
            replan_reason=replan_reason,
            n_agents=self.n_agents,
        )
        try:
            resp = self.llm.complete([{"role": "user", "content": prompt}])
            parsed = _extract_json(resp.content)
        except Exception as e:
            print(f"warning: OvercookedStrategy failed: {e}", file=sys.stderr)
            parsed = {}

        assignments = parsed.get("assignments", [])
        if len(assignments) != self.n_agents:
            assignments = [
                {"agent_id": i, "role": "flexible", "detail": "do what's needed"}
                for i in range(self.n_agents)
            ]

        reasoning = parsed.get("reasoning", "")
        subs = []
        for a in assignments:
            i = a.get("agent_id", len(subs))
            role_name = a.get("role", "flexible")
            detail = a.get("detail", "")
            description = (
                f"You are Agent {i} in the Overcooked kitchen.\n"
                f"Your assigned role: {role_name}\n"
                f"Detail: {detail}\n"
                f"Strategist note: {reasoning}\n\n"
                f"Use do_plan() with commands like: pickup ingredient_0, place_in pot_0, "
                f"pickup plate_pile, deliver, wait_near pot_0.\n"
                f"Observe after each action. When your current work is complete, final='done'."
            )
            subs.append(Subtask(index=i, description=description, role_name="cook_worker"))
        return subs

    def decompose(self, task: str, ctx: OrchContext) -> list[Subtask]:
        """Standard DecomposeStrategy interface. `task` is the rendered state text."""
        return self.decompose_with_state(task, "initial plan", ctx)
