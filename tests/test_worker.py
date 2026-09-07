import json
from pathlib import Path

from env.local import LocalEnv
from memory.markdown import MarkdownMemory
from orchestrator.llm import FakeLLM
from worker.react import ReactAgent
from worker.role import Role, DEFAULT_ROLE
from worker.worker import Worker


def _noop_toolset(env):
    return []


def test_worker_injects_workboard_tools(tmp_path: Path):
    wb = MarkdownMemory(tmp_path / "wb.md")
    llm = FakeLLM([
        json.dumps({"thought": "write wb", "action": {"tool": "edit_workboard", "args": {"section": "P", "content": "hello"}}}),
        json.dumps({"thought": "done", "final": "ok"}),
    ])
    w = Worker(
        agent=ReactAgent(llm, max_steps=4),
        role=DEFAULT_ROLE,
        tools=_noop_toolset,
        env=LocalEnv(root=tmp_path),
        workboard=wb,
    )
    result = w.run("task")
    assert result.final_answer == "ok"
    assert "hello" in wb.read()


def test_worker_role_filters_tools(tmp_path: Path):
    wb = MarkdownMemory(tmp_path / "wb.md")
    # Allow only read_workboard — edit_workboard must be unavailable
    role = Role(name="reader", system_prompt="read only", allowed_tools=["read_workboard"])
    llm = FakeLLM([
        json.dumps({"thought": "try edit", "action": {"tool": "edit_workboard", "args": {"section": "x", "content": "y"}}}),
        json.dumps({"thought": "fallback", "final": "blocked"}),
    ])
    w = Worker(
        agent=ReactAgent(llm, max_steps=3),
        role=role,
        tools=_noop_toolset,
        env=LocalEnv(root=tmp_path),
        workboard=wb,
    )
    result = w.run("task")
    assert result.final_answer == "blocked"
    # edit_workboard was filtered → observation says "unknown tool"
    assert "unknown tool" in result.steps[0]["observation"].lower()


def test_worker_role_selector(tmp_path: Path):
    wb = MarkdownMemory(tmp_path / "wb.md")
    picked = Role(name="picked", system_prompt="picked", allowed_tools=None)

    class Sel:
        def select(self, task, ctx):
            return picked

    llm = FakeLLM([json.dumps({"thought": "t", "final": "done"})])
    w = Worker(
        agent=ReactAgent(llm, max_steps=2),
        role=Sel(),
        tools=_noop_toolset,
        env=LocalEnv(root=tmp_path),
        workboard=wb,
    )
    result = w.run("task")
    assert result.final_answer == "done"
