from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.mock_adapter import MockAdapter
from scenarios.overcooked.planner import MockPlanner
from scenarios.overcooked.tools import make_overcooked_tools


def test_observe_returns_text():
    adapter = MockAdapter()
    adapter.reset()
    planner = MockPlanner(adapter)
    tools = {t.name: t for t in make_overcooked_tools(adapter, agent_id=0, planner=planner)}
    result = tools["observe"].call()
    assert isinstance(result, str)
    assert "Agent 0" in result or "agent 0" in result.lower()


def test_get_status_returns_dict():
    adapter = MockAdapter()
    adapter.reset()
    planner = MockPlanner(adapter)
    tools = {t.name: t for t in make_overcooked_tools(adapter, agent_id=0, planner=planner)}
    result = tools["get_status"].call()
    assert "score" in result
    assert "step" in result


def test_do_plan_executes_and_returns_observation():
    adapter = MockAdapter()
    adapter.reset()
    planner = MockPlanner(adapter)
    tools = {t.name: t for t in make_overcooked_tools(adapter, agent_id=0, planner=planner)}
    result = tools["do_plan"].call(cmd="pickup ingredient_0")
    assert isinstance(result, str)
    # After executing, agent should have moved or interacted
    state = adapter.get_state()
    # Agent 0 should have tried to pick up ingredient
    assert "Agent 0" in result or "agent 0" in result.lower()
