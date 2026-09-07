from scenarios.overcooked.common import Action, AgentState, Position
from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.mock_adapter import MockAdapter
from scenarios.overcooked.planner import MockPlanner


def test_planner_pickup_returns_action_sequence():
    adapter = MockAdapter()
    state = adapter.reset()
    planner = MockPlanner(adapter)
    actions = planner.plan(agent_id=0, cmd="pickup ingredient_0")
    assert isinstance(actions, list)
    assert all(isinstance(a, int) for a in actions)
    assert len(actions) > 0
    # Last action should be INTERACT
    assert actions[-1] == Action.INTERACT


def test_planner_deliver_returns_action_sequence():
    adapter = MockAdapter()
    state = adapter.reset()
    planner = MockPlanner(adapter)
    actions = planner.plan(agent_id=0, cmd="deliver")
    assert isinstance(actions, list)
    assert actions[-1] == Action.INTERACT


def test_planner_wait_returns_stay_actions():
    adapter = MockAdapter()
    state = adapter.reset()
    planner = MockPlanner(adapter)
    actions = planner.plan(agent_id=0, cmd="wait_near pot_0")
    assert all(a == Action.STAY for a in actions) or actions[-1] == Action.STAY


def test_planner_unknown_command_returns_stay():
    adapter = MockAdapter()
    state = adapter.reset()
    planner = MockPlanner(adapter)
    actions = planner.plan(agent_id=0, cmd="dance wildly")
    assert actions == [Action.STAY]
