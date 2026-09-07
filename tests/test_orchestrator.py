import json
from pathlib import Path

from env.local import LocalEnv
from memory.markdown import MarkdownMemory
from orchestrator.llm import FakeLLM
from orchestrator.orchestrator import Orchestrator
from worker.react import ReactAgent
from worker.role import DEFAULT_ROLE
from worker.worker import Worker


def _empty_toolset(env):
    return []


def test_orchestrator_end_to_end(tmp_path: Path):
    # Orchestrator LLM: one decomposition call returning 2 subtasks
    orch_llm = FakeLLM([json.dumps([{"description": "A"}, {"description": "B"}])])

    # Each worker gets its own LLM script — build a factory that rotates them.
    scripts = [
        [json.dumps({"thought": "t", "final": "answerA"})],
        [json.dumps({"thought": "t", "final": "answerB"})],
    ]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)

    def factory(role, wb, sub):
        script = scripts.pop(0)
        return Worker(
            agent=ReactAgent(FakeLLM(script), max_steps=3),
            role=role,
            tools=_empty_toolset,
            env=env,
            workboard=wb,
        )

    orch = Orchestrator(
        llm=orch_llm,
        worker_factory=factory,
        workboard=workboard,
        log_memory=log,
    )
    out = orch.run("do a thing")
    assert "answerA" in out
    assert "answerB" in out
    # order preserved
    assert out.index("answerA") < out.index("answerB")
    # log updated
    assert "Decomposition" in log.read()
    assert "subtask_0: A" in log.read()


def test_orchestrator_respects_strategy(tmp_path: Path):
    from orchestrator.orchestrator import Subtask

    class FixedStrategy:
        def decompose(self, task, ctx):
            return [Subtask(index=0, description="only")]

    scripts = [[json.dumps({"thought": "t", "final": "done"})]]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=2),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    # orch_llm never called because strategy short-circuits decompose
    orch = Orchestrator(
        llm=FakeLLM([]),
        worker_factory=factory,
        workboard=workboard,
        decompose_strategy=FixedStrategy(),
    )
    out = orch.run("task")
    assert "done" in out


def test_orchestrator_run_pairs_returns_structured(tmp_path: Path):
    """run_pairs skips aggregated-string assembly and returns structured tuples."""
    orch_llm = FakeLLM([json.dumps([{"description": "A"}, {"description": "B"}])])
    scripts = [
        [json.dumps({"thought": "t", "final": "5"})],
        [json.dumps({"thought": "t", "final": "7"})],
    ]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=3),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard)
    pairs = orch.run_pairs("task")
    assert pairs == [(0, "5"), (1, "7")]
