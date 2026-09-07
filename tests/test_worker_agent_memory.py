import json
from pathlib import Path

from env.local import LocalEnv
from memory.markdown import MarkdownMemory
from orchestrator.llm import FakeLLM
from worker.react import ReactAgent
from worker.role import DEFAULT_ROLE
from worker.worker import Worker


def _empty_toolset(env):
    return []


def test_worker_with_agent_memory_injects_scratch_tools(tmp_path: Path):
    wb = MarkdownMemory(tmp_path / "wb.md")
    sc = MarkdownMemory(tmp_path / "sc.md", persistent=True)

    llm = FakeLLM([
        json.dumps({"thought": "save", "action": {"tool": "write_scratch", "args": {"content": "my note"}}}),
        json.dumps({"thought": "verify", "action": {"tool": "read_scratch", "args": {}}}),
        json.dumps({"thought": "done", "final": "ok"}),
    ])
    w = Worker(
        agent=ReactAgent(llm, max_steps=4),
        role=DEFAULT_ROLE,
        tools=_empty_toolset,
        env=LocalEnv(root=tmp_path),
        workboard=wb,
        agent_memory=sc,
    )
    result = w.run("task")
    assert result.final_answer == "ok"
    assert "my note" in sc.read()


def test_worker_without_agent_memory_has_no_scratch_tools(tmp_path: Path):
    """Backward compat: agent_memory=None means no scratch tools (milestone-1 behavior)."""
    wb = MarkdownMemory(tmp_path / "wb.md")
    llm = FakeLLM([
        json.dumps({"thought": "try", "action": {"tool": "write_scratch", "args": {"content": "x"}}}),
        json.dumps({"thought": "fallback", "final": "no scratch"}),
    ])
    w = Worker(
        agent=ReactAgent(llm, max_steps=3),
        role=DEFAULT_ROLE,
        tools=_empty_toolset,
        env=LocalEnv(root=tmp_path),
        workboard=wb,
    )
    result = w.run("task")
    assert result.final_answer == "no scratch"
    assert "unknown tool" in result.steps[0]["observation"].lower()
