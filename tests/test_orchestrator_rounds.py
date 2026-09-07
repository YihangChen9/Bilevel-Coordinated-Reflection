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


def test_run_rounds_k1_equivalent_to_run_pairs(tmp_path: Path):
    """n_rounds=1 returns the same shape as run_pairs, wrapped in a length-1 outer list."""
    orch_llm = FakeLLM([json.dumps([{"description": "A"}, {"description": "B"}])])
    scripts = [
        [json.dumps({"thought": "t", "final": "X"})],
        [json.dumps({"thought": "t", "final": "Y"})],
    ]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=3),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard)
    rounds = orch.run_rounds("task", n_rounds=1)
    assert len(rounds) == 1
    assert rounds[0] == [(0, "X"), (1, "Y")]


def test_run_rounds_k2_dispatches_workers_twice(tmp_path: Path):
    """With n_rounds=2, workers are dispatched twice and the workboard persists between."""
    orch_llm = FakeLLM([json.dumps([{"description": "A"}])])

    round0_script = [
        json.dumps({"thought": "write r0", "action": {"tool": "edit_workboard", "args": {"section": "agent_0", "content": "r0"}}}),
        json.dumps({"thought": "done", "final": "round0"}),
    ]
    round1_script = [
        json.dumps({"thought": "read", "action": {"tool": "read_workboard", "args": {}}}),
        json.dumps({"thought": "done", "final": "round1_saw_r0"}),
    ]
    scripts = [round0_script, round1_script]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=5),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard)
    rounds = orch.run_rounds("task", n_rounds=2)
    assert len(rounds) == 2
    assert rounds[0] == [(0, "round0")]
    assert rounds[1] == [(0, "round1_saw_r0")]


def test_run_rounds_logs_one_entry_per_round(tmp_path: Path):
    """orch_log receives K entries when n_rounds=K."""
    orch_llm = FakeLLM([json.dumps([{"description": "A"}])])
    scripts = [
        [json.dumps({"thought": "t", "final": "a"})],
        [json.dumps({"thought": "t", "final": "b"})],
        [json.dumps({"thought": "t", "final": "c"})],
    ]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=2),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard, log_memory=log)
    orch.run_rounds("task", n_rounds=3)

    text = log.read()
    assert text.count("## Task ") == 3
    assert "[round 1/3]" in text
    assert "[round 2/3]" in text
    assert "[round 3/3]" in text


def test_round_marker_only_when_k_gt_1(tmp_path: Path):
    """When n_rounds=1, the subtask description passed to workers must NOT have a round prefix."""
    orch_llm = FakeLLM([json.dumps([{"description": "plain desc"}])])
    received_descriptions: list[str] = []

    class CaptureAgent:
        def run(self, task, tools, ctx):
            from worker.base import AgentResult
            received_descriptions.append(task)
            return AgentResult(final_answer="ok", steps=[], stopped_reason="final")

    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")

    def factory(role, wb, sub):
        return Worker(agent=CaptureAgent(), role=role, tools=_empty_toolset, env=env, workboard=wb)

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard)
    orch.run_rounds("task", n_rounds=1)

    assert received_descriptions == ["plain desc"]
    received_descriptions.clear()

    orch_llm2 = FakeLLM([json.dumps([{"description": "plain desc"}])])
    orch2 = Orchestrator(llm=orch_llm2, worker_factory=factory, workboard=workboard)
    orch2.run_rounds("task", n_rounds=2)

    assert len(received_descriptions) == 2
    assert received_descriptions[0].startswith("[Round 1/2]")
    assert received_descriptions[1].startswith("[Round 2/2]")


def test_annotate_with_round_is_identity_when_k_equals_1():
    """When n_rounds == 1, the helper must return the input list object itself —
    not a copy — so the zero-round-overhead contract is visible."""
    from orchestrator.orchestrator import Subtask, _annotate_with_round

    xs = [Subtask(index=0, description="x", role_name=None)]
    assert _annotate_with_round(xs, 0, 1) is xs


def test_run_rounds_k1_log_has_no_round_suffix(tmp_path: Path):
    """Back-compat: n_rounds=1 log entry title must NOT contain '[round 1/1]'."""
    orch_llm = FakeLLM([json.dumps([{"description": "A"}])])
    scripts = [[json.dumps({"thought": "t", "final": "done"})]]
    env = LocalEnv(root=tmp_path)
    workboard = MarkdownMemory(tmp_path / "wb.md")
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)

    def factory(role, wb, sub):
        return Worker(
            agent=ReactAgent(FakeLLM(scripts.pop(0)), max_steps=2),
            role=role, tools=_empty_toolset, env=env, workboard=wb,
        )

    orch = Orchestrator(llm=orch_llm, worker_factory=factory, workboard=workboard, log_memory=log)
    orch.run_rounds("task", n_rounds=1)

    text = log.read()
    assert text.count("## Task ") == 1
    assert "[round" not in text  # no suffix at all when K=1
