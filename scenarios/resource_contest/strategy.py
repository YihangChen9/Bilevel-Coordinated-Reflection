"""RCStrategy — upper-level allocation.

Reads orch_log history of `(τ, x, reward)`. Asks the LLM to propose a τ vector
summing to budget C that concentrates on the agent with the largest observed
xᵢ. Falls back to uniform if the LLM output is unusable.
"""
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from memory.base import MemoryPort
from orchestrator.llm import LLM
from orchestrator.orchestrator import OrchContext, Subtask

from scenarios.resource_contest.game import GameParams


_RECORD_RE = re.compile(
    r"reward=(?P<reward>-?\d+(?:\.\d+)?)\s+"
    r"tau=(?P<tau>\[[^\]]*\])\s+"
    r"x=(?P<x>\[[^\]]*\])"
)


def _parse_float_list(s: str) -> list[float]:
    return [float(p.strip()) for p in s.strip("[]").split(",") if p.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(round(float(p.strip()))) for p in s.strip("[]").split(",") if p.strip()]


@dataclass
class History:
    records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_log(cls, log_text: str, tail: int = 10) -> "History":
        records: list[dict[str, Any]] = []
        for m in _RECORD_RE.finditer(log_text):
            records.append({
                "reward": float(m["reward"]),
                "tau": _parse_float_list(m["tau"]),
                "x": _parse_int_list(m["x"]),
            })
        return cls(records=records[-tail:])

    def best_observed_x(self, n: int, window: int | None = None) -> list[int]:
        """Highest xᵢ observed for each agent (=0 if never seen).

        If `window` is set, max is taken only over the last `window` records —
        used by the `sliding_max_k5` ablation to make the summary forget old
        observations (e.g. after a cap-flip in the `drift` config).
        """
        recs = self.records[-window:] if window is not None else self.records
        best = [0] * n
        for rec in recs:
            for i, xi in enumerate(rec["x"][:n]):
                if xi > best[i]:
                    best[i] = xi
        return best


_DECOMPOSE_PROMPT = """\
You are the strategic planner of a Resource Contest game.

Game:
- {n} agents. Each agent i has a HIDDEN cap M_i ∈ [0, 10]. You don't know M_i.
- Per round you pick a budget allocation τ = (τ_0, …, τ_{n_minus_1}) with τ_i ≥ 0 and Σ τ_i = {budget}.
- Agent i submits an integer x_i; env silently clips to min(x_i, M_i).
- Shared reward = Σ τ_i · x_i. Your job is to maximize cumulative reward over rounds.

Key insight (LINEAR payoff over the simplex): the optimum is to put ALL budget
on the single agent with the largest x_i. There's no benefit to spreading.
But you have to figure out who has the largest cap first.

Recent rounds (most recent last; empty if t=0):
{history_block}

Best observed x per agent so far:
{best_block}

Output ONLY a JSON object:
{{"tau": [<{n} floats, all ≥ 0, summing to {budget}>], "reasoning": "<one short sentence>"}}

Suggested policy:
- t == 0: explore uniformly: τ_i = {budget}/{n} for all i.
- t ≥ 1: argmax = agent with largest observed x. Put most budget there.
  Reserve a small slice (~10%) for the next-best for continued exploration
  if your top observation is still uncertain (e.g. only 1 datapoint).
- Once you're confident (3+ rounds with same argmax), concentrate fully: τ = budget on the argmax, 0 elsewhere.
"""


def _format_history(h: History) -> str:
    if not h.records:
        return "(no prior rounds)"
    return "\n".join(
        f"  t-{len(h.records)-i}: reward={r['reward']:.3f} tau={[f'{t:.2f}' for t in r['tau']]} x={r['x']}"
        for i, r in enumerate(h.records)
    )


def _format_best(best: list[int]) -> str:
    return ", ".join(f"agent_{i}={v}" for i, v in enumerate(best))


def _extract_json_object(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(s[start:i + 1])
    raise ValueError(f"no JSON object in: {text[:200]}")


def _safe_float(v: object, fb: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fb
    return f if math.isfinite(f) else fb


def _normalize_to_simplex(tau: list[float], budget: float) -> list[float]:
    """Project (clip to ≥ 0, renormalize to sum=budget). Empty / all-zero → uniform."""
    n = len(tau)
    if n == 0:
        return []
    clipped = [max(0.0, t) for t in tau]
    s = sum(clipped)
    if s == 0:
        return [budget / n] * n
    return [t * budget / s for t in clipped]


class RCStrategy:
    """Implements DecomposeStrategy.decompose for Resource Contest."""

    def __init__(
        self,
        llm: LLM,
        params: GameParams,
        log_memory: MemoryPort,
        last_k: int = 10,
        best_window: int | None = None,
        worker_blind: bool = False,
        workboard_for_read: MemoryPort | None = None,
    ):
        self.llm = llm
        self.params = params
        self.log = log_memory
        self.last_k = last_k
        self.best_window = best_window
        self.worker_blind = worker_blind
        # Optional: read workboard (m_e) and inject into strategy prompt — closes
        # the algorithmic loop from Phase II reflection-writes to Phase III meta-
        # reflection on the leader side.
        self.workboard_for_read = workboard_for_read
        self.last_tau: list[float] = []

    def decompose(self, task: str, ctx: OrchContext) -> list[Subtask]:
        history = History.from_log(self.log.read(), tail=self.last_k)
        best = history.best_observed_x(self.params.n_agents, window=self.best_window)

        history_block = _format_history(history)
        if self.workboard_for_read is not None:
            wb_text = self.workboard_for_read.read().strip()
            if wb_text:
                history_block += (
                    "\n\nWorkboard (live notes from workers this episode):\n"
                    + wb_text
                )

        prompt = _DECOMPOSE_PROMPT.format(
            n=self.params.n_agents,
            n_minus_1=self.params.n_agents - 1,
            budget=self.params.budget,
            history_block=history_block,
            best_block=_format_best(best),
        )
        try:
            resp = self.llm.complete([{"role": "user", "content": prompt}])
            parsed = _extract_json_object(resp.content)
        except Exception as e:  # noqa: BLE001 — strategy must not crash a round
            print(
                f"warning: RCStrategy LLM/parse failed ({type(e).__name__}: {e}); using uniform fallback",
                file=sys.stderr,
            )
            parsed = {}

        raw = parsed.get("tau") or []
        if len(raw) != self.params.n_agents:
            tau = [self.params.budget / self.params.n_agents] * self.params.n_agents
        else:
            tau = _normalize_to_simplex(
                [_safe_float(v, 0.0) for v in raw], self.params.budget
            )
        self.last_tau = list(tau)
        reasoning = str(parsed.get("reasoning", ""))

        subs: list[Subtask] = []
        for i in range(self.params.n_agents):
            if self.worker_blind:
                # Minimum-information brief: only the per-round goal, no formula,
                # no mention of M_i / clipping / tau_i / monotonicity / optimal x.
                description = (
                    f"You are agent {i}. Commit an integer in [0, 10] via\n"
                    f"commit_action. Maximize the effective_x the env returns."
                )
                role_name = "rc_player_blind"
            else:
                description = (
                    f"You are agent {i} of {self.params.n_agents} in Resource Contest.\n"
                    f"Your allocation this round: τ_{i} = {tau[i]:.3f}\n"
                    f"Strategist note: {reasoning}\n\n"
                    f"Goal: maximize τ_{i} · effective_x. Your hidden cap M_{i} ∈ [0, 10].\n"
                    f"If your scratch records M_{i}, commit it directly. Else commit 10 once\n"
                    f"to discover your cap, write it to scratch, and respond with the\n"
                    f"effective integer the env reported."
                )
                role_name = "rc_player"
            subs.append(Subtask(index=i, description=description, role_name=role_name))
        return subs
