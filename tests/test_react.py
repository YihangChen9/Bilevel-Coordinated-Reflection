import json
from pathlib import Path

from orchestrator.llm import FakeLLM
from worker.base import AgentContext
from worker.react import ReactAgent
from worker.tool import Tool


def _echo_tool():
    return Tool(
        name="echo",
        description="echo back the input",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        func=lambda msg: f"got:{msg}",
    )


def test_react_single_tool_then_final():
    llm = FakeLLM([
        json.dumps({"thought": "use echo", "action": {"tool": "echo", "args": {"msg": "hi"}}}),
        json.dumps({"thought": "done", "final": "answer=got:hi"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=5)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "answer=got:hi"
    assert result.stopped_reason == "final"
    assert len(result.steps) == 2  # tool step + final step


def test_react_hits_max_steps():
    loop_msg = json.dumps({"thought": "spin", "action": {"tool": "echo", "args": {"msg": "x"}}})
    llm = FakeLLM([loop_msg, loop_msg, loop_msg])
    agent = ReactAgent(llm=llm, max_steps=2)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 2


def test_react_unknown_tool_becomes_observation():
    llm = FakeLLM([
        json.dumps({"thought": "bad", "action": {"tool": "nope", "args": {}}}),
        json.dumps({"thought": "recover", "final": "ok"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=3)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "ok"
    # step 1 recorded an error observation
    assert "unknown tool" in result.steps[0]["observation"].lower()


def test_react_malformed_json_becomes_observation():
    llm = FakeLLM([
        "not json at all",
        json.dumps({"thought": "recover", "final": "done"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=3)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "done"
    assert "json" in result.steps[0]["observation"].lower()


def test_react_tolerates_code_fence():
    llm = FakeLLM([
        '```json\n' + json.dumps({"thought": "t", "final": "done"}) + '\n```',
    ])
    agent = ReactAgent(llm=llm, max_steps=2)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "done"
    assert result.stopped_reason == "final"


def test_react_tolerates_leading_prose():
    llm = FakeLLM([
        "Sure, here you go:\n" + json.dumps({"thought": "t", "final": "ok"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=2)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "ok"


def test_react_records_raw_on_every_step():
    final_raw = json.dumps({"thought": "t", "final": "x"})
    llm = FakeLLM([final_raw])
    agent = ReactAgent(llm=llm, max_steps=2)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.steps[0]["raw"] == final_raw


def test_react_rejects_non_dict_json():
    llm = FakeLLM([
        json.dumps(["not", "an", "object"]),
        json.dumps({"thought": "t", "final": "recover"}),
    ])
    agent = ReactAgent(llm=llm, max_steps=3)
    ctx = AgentContext(role_name="t", system_prompt="sys")
    result = agent.run("task", [_echo_tool()], ctx)
    assert result.final_answer == "recover"
    assert "object" in result.steps[0]["observation"].lower()
