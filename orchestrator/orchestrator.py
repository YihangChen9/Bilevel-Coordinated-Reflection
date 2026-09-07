import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from memory.base import MemoryPort
from orchestrator.llm import LLM
from worker.role import DEFAULT_ROLE, Role
from worker.worker import Worker


@dataclass
class Subtask:
    index: int
    description: str
    role_name: str | None = None  # filled by strategy or left default


@dataclass
class OrchContext:
    task: str
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecomposeStrategy(Protocol):
    def decompose(self, task: str, ctx: OrchContext) -> list[Subtask]: ...


_DEFAULT_DECOMPOSE_PROMPT = """\
You are a task decomposer. Given a user task, split it into 1-5 self-contained
subtasks that can be executed in parallel. Each subtask must be independently
completable and state its own goal.

Return ONLY a JSON array like:
[{{"description": "..."}}, {{"description": "..."}}]

Task: {task}
"""


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON array in LLM output: {text[:200]}")
    return json.loads(m.group(0))


def default_decompose(llm: LLM, task: str) -> list[Subtask]:
    prompt = _DEFAULT_DECOMPOSE_PROMPT.format(task=task)
    resp = llm.complete([{"role": "user", "content": prompt}])
    items = _extract_json_array(resp.content)
    return [Subtask(index=i, description=item["description"]) for i, item in enumerate(items)]


WorkerFactory = Callable[[Role, MemoryPort, Subtask], Worker]


def _annotate_with_round(
    subtasks: list[Subtask], round_idx: int, n_rounds: int
) -> list[Subtask]:
    """When n_rounds > 1, prepend a `[Round r+1/K]` marker to each subtask's description.
    When n_rounds == 1, returns the list unchanged (byte-for-byte)."""
    if n_rounds == 1:
        return subtasks
    prefix = f"[Round {round_idx + 1}/{n_rounds}]\n"
    return [
        Subtask(
            index=s.index,
            description=prefix + s.description,
            role_name=s.role_name,
        )
        for s in subtasks
    ]


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        worker_factory: WorkerFactory,
        workboard: MemoryPort,
        decompose_strategy: DecomposeStrategy | None = None,
        log_memory: MemoryPort | None = None,
        scratch_memory: MemoryPort | None = None,
        role_registry: dict[str, Role] | None = None,
        max_parallel: int = 10,
        clear_workboard_per_round: bool = True,
    ):
        self.llm = llm
        self.worker_factory = worker_factory
        self.workboard = workboard
        self.decompose_strategy = decompose_strategy
        self.log_memory = log_memory
        self.scratch_memory = scratch_memory
        self.role_registry = role_registry or {"default": DEFAULT_ROLE}
        self.max_parallel = max_parallel
        self.clear_workboard_per_round = clear_workboard_per_round

    def _decompose(self, task: str) -> list[Subtask]:
        if self.decompose_strategy is not None:
            return self.decompose_strategy.decompose(task, OrchContext(task=task))
        return default_decompose(self.llm, task)

    def _resolve_role(self, name: str | None) -> Role:
        return self.role_registry.get(name or "default", DEFAULT_ROLE)

    async def _run_subtask(self, sub: Subtask, sem: asyncio.Semaphore) -> tuple[int, str]:
        role = self._resolve_role(sub.role_name)
        async with sem:
            worker = self.worker_factory(role, self.workboard, sub)
            result = await asyncio.to_thread(worker.run, sub.description)
        return sub.index, result.final_answer

    async def _arun_rounds(self, task: str, n_rounds: int) -> list[list[tuple[int, str]]]:
        """Core multi-round loop. Clears memories at start of the timestep, decomposes
        once (strategy LLM called once regardless of K), then dispatches workers K times
        with workboard persisting across rounds. Returns K lists of (index, final_answer)
        sorted by index per round."""
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

        if self.clear_workboard_per_round:
            self.workboard.clear()
        if self.scratch_memory is not None:
            self.scratch_memory.clear()

        base_subtasks = self._decompose(task)
        sem = asyncio.Semaphore(self.max_parallel)
        all_rounds: list[list[tuple[int, str]]] = []

        for round_idx in range(n_rounds):
            round_subtasks = _annotate_with_round(base_subtasks, round_idx, n_rounds)
            pairs = await asyncio.gather(
                *[self._run_subtask(s, sem) for s in round_subtasks]
            )
            pairs.sort(key=lambda p: p[0])
            all_rounds.append(pairs)

            if self.log_memory is not None:
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                summary = task if len(task) <= 80 else task[:77] + "..."
                suffix = f" [round {round_idx + 1}/{n_rounds}]" if n_rounds > 1 else ""
                aggregated = "\n\n".join(f"### subtask {i}\n{ans}" for i, ans in pairs)
                body = (
                    f"### Decomposition\n"
                    + "\n".join(f"- subtask_{s.index}: {s.description}" for s in base_subtasks)
                    + f"\n\n### Final output\n{aggregated}\n\n### Score\nn/a\n"
                )
                self.log_memory.append(f"Task {ts} — \"{summary}\"{suffix}", body)

        return all_rounds

    def run(self, task: str) -> str:
        """Back-compat: returns the aggregated string form."""
        pairs = asyncio.run(self._arun_rounds(task, n_rounds=1))[0]
        return "\n\n".join(f"### subtask {i}\n{ans}" for i, ans in pairs)

    def run_pairs(self, task: str) -> list[tuple[int, str]]:
        """Milestone-2 structured single-round entry point."""
        return asyncio.run(self._arun_rounds(task, n_rounds=1))[0]

    def run_rounds(self, task: str, n_rounds: int = 1) -> list[list[tuple[int, str]]]:
        """Multi-round entry point. Returns K lists of sorted (index, final_answer) pairs."""
        return asyncio.run(self._arun_rounds(task, n_rounds=n_rounds))
