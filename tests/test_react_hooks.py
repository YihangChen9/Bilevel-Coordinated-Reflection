import json

from orchestrator.llm import FakeLLM
from worker.base import AgentContext
from worker.hooks import StepInfo
from worker.react import ReactAgent
from worker.tool import Tool


def _echo_tool():
    return Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        func=lambda msg: f"got:{msg}",
    )


def test_pre_step_hook_modifies_messages():
    captured: list[int] = []

    def hook(messages, info):
        captured.append(info.step_idx)
        out = [dict(m) for m in messages]
        out[0]["content"] += " [HOOKED]"
        return out

    llm = FakeLLM([json.dumps({"thought": "t", "final": "done"})])
    agent = ReactAgent(llm=llm, max_steps=2, pre_step_hook=hook)
    ctx = AgentContext(role_name="r", system_prompt="sys")
    agent.run("task", [_echo_tool()], ctx)

    assert captured == [0]
    assert "[HOOKED]" in llm.calls[0][0]["content"]


def test_post_step_hook_receives_populated_step_info():
    seen: list[StepInfo] = []

    def hook(info: StepInfo) -> None:
        seen.append(info)

    llm = FakeLLM([
        json.dumps({"thought": "t", "action": {"tool": "echo", "args": {"msg": "hi"}}}),
        json.dumps({"thought": "t", "final": "ok"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=3, post_step_hook=hook)
    ctx = AgentContext(role_name="r", system_prompt="sys")
    agent.run("task", [_echo_tool()], ctx)

    assert len(seen) == 2
    assert seen[0].step_idx == 0
    assert seen[0].parsed_action == {"tool": "echo", "args": {"msg": "hi"}}
    assert seen[0].observation is not None
    assert "got:hi" in seen[0].observation
    assert seen[1].step_idx == 1
    assert seen[1].parsed_action is None  # final-only step


def test_pre_step_hook_exception_does_not_crash():
    def bad(messages, info):
        raise RuntimeError("boom")

    llm = FakeLLM([json.dumps({"thought": "t", "final": "ok"})])
    agent = ReactAgent(llm=llm, max_steps=2, pre_step_hook=bad)
    ctx = AgentContext(role_name="r", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "ok"


def test_post_step_hook_exception_does_not_crash():
    def bad(info):
        raise RuntimeError("boom")

    llm = FakeLLM([json.dumps({"thought": "t", "final": "ok"})])
    agent = ReactAgent(llm=llm, max_steps=2, post_step_hook=bad)
    ctx = AgentContext(role_name="r", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "ok"


import pytest


def test_pre_step_hooks_list_runs_in_order():
    order: list[str] = []

    def h1(messages, info):
        order.append("h1")
        out = [dict(m) for m in messages]
        out[0]["content"] += " [H1]"
        return out

    def h2(messages, info):
        order.append("h2")
        out = [dict(m) for m in messages]
        out[0]["content"] += " [H2]"
        return out

    llm = FakeLLM([json.dumps({"thought": "t", "final": "done"})])
    agent = ReactAgent(llm=llm, max_steps=2, pre_step_hooks=[h1, h2])
    ctx = AgentContext(role_name="r", system_prompt="sys")
    agent.run("task", [_echo_tool()], ctx)

    assert order == ["h1", "h2"]
    content = llm.calls[0][0]["content"]
    assert content.index("[H1]") < content.index("[H2]")


def test_post_step_hooks_list_both_receive_step_info():
    seen1: list[int] = []
    seen2: list[int] = []

    def h1(info):
        seen1.append(info.step_idx)

    def h2(info):
        seen2.append(info.step_idx)

    llm = FakeLLM([
        json.dumps({"thought": "t", "action": {"tool": "echo", "args": {"msg": "hi"}}}),
        json.dumps({"thought": "t", "final": "ok"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=3, post_step_hooks=[h1, h2])
    ctx = AgentContext(role_name="r", system_prompt="sys")
    agent.run("task", [_echo_tool()], ctx)

    assert seen1 == [0, 1]
    assert seen2 == [0, 1]


def test_react_rejects_both_singular_and_plural_pre_hook():
    with pytest.raises(ValueError):
        ReactAgent(
            llm=FakeLLM([]),
            max_steps=1,
            pre_step_hook=lambda m, i: m,
            pre_step_hooks=[lambda m, i: m],
        )


def test_react_rejects_both_singular_and_plural_post_hook():
    with pytest.raises(ValueError):
        ReactAgent(
            llm=FakeLLM([]),
            max_steps=1,
            post_step_hook=lambda i: None,
            post_step_hooks=[lambda i: None],
        )
