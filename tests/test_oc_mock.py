from scenarios.overcooked.common import Action, Position
from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.mock_adapter import MockAdapter


def test_mock_reset_returns_initial_state():
    adapter = MockAdapter()
    state = adapter.reset()
    assert state.step == 0
    assert state.score == 0
    assert len(state.agents) == 2
    assert len(state.pots) == 1
    assert state.pots[0].is_idle


def test_mock_agent_moves():
    adapter = MockAdapter()
    adapter.reset()
    state, _ = adapter.step({0: Action.RIGHT, 1: Action.STAY})
    assert state.agents[0].pos == Position(2, 2)
    assert state.agents[1].pos == Position(1, 3)


def test_mock_full_delivery_pipeline():
    """Walk through: pickup ingredient × 3 → pot cooks → pickup plate → pickup dish → deliver."""
    config = OvercookedConfig(max_steps=100, cook_time=3, recipe_size=3, n_agents=2)
    adapter2 = MockAdapter(config)
    adapter2.reset()

    # Directly manipulate agent state for a delivery test
    adapter2._agents[0].pos = Position(2, 2)  # near ingredient
    adapter2._agents[0].direction = Action.UP  # face ingredient at (2,1)
    adapter2._agents[0].holding = ""

    # Pick up 3 ingredients and place in pot
    for _ in range(3):
        adapter2._agents[0].pos = Position(2, 2)
        adapter2._agents[0].direction = Action.UP
        adapter2.step({0: Action.INTERACT, 1: Action.STAY})  # pickup ingredient
        assert adapter2._agents[0].holding == "ingredient_0"
        adapter2._agents[0].pos = Position(2, 1)
        adapter2._agents[0].direction = Action.RIGHT  # face pot at (3,1)
        adapter2.step({0: Action.INTERACT, 1: Action.STAY})  # place in pot
        assert adapter2._agents[0].holding == ""

    # Pot should be cooking now
    state = adapter2.get_state()
    assert state.pots[0].is_cooking

    # Wait for cooking
    for _ in range(config.cook_time):
        adapter2.step({0: Action.STAY, 1: Action.STAY})
    state = adapter2.get_state()
    assert state.pots[0].is_ready

    # Agent 1 picks up plate, then dish, then delivers
    adapter2._agents[1].pos = Position(2, 3)
    adapter2._agents[1].direction = Action.RIGHT  # face plate pile at (3,3)
    adapter2.step({0: Action.STAY, 1: Action.INTERACT})  # pickup plate
    assert adapter2._agents[1].holding == "plate"

    adapter2._agents[1].pos = Position(2, 1)
    adapter2._agents[1].direction = Action.RIGHT  # face pot at (3,1)
    adapter2.step({0: Action.STAY, 1: Action.INTERACT})  # pickup dish
    assert adapter2._agents[1].holding == "dish"

    adapter2._agents[1].pos = Position(2, 2)
    adapter2._agents[1].direction = Action.RIGHT  # face goal at (3,2)
    adapter2.step({0: Action.STAY, 1: Action.INTERACT})  # deliver
    assert adapter2._agents[1].holding == ""
    assert adapter2.get_state().score == 20


def test_mock_stuck_detection():
    config = OvercookedConfig(max_steps=50, stuck_threshold=5, n_agents=2)
    adapter = MockAdapter(config)
    adapter.reset()
    for _ in range(5):
        _, events = adapter.step({0: Action.STAY, 1: Action.STAY})
    assert 0 in events.stuck_agents
    assert 1 in events.stuck_agents


def test_mock_episode_terminates():
    config = OvercookedConfig(max_steps=10, n_agents=2)
    adapter = MockAdapter(config)
    adapter.reset()
    for _ in range(10):
        state, _ = adapter.step({0: Action.STAY, 1: Action.STAY})
    assert state.done
