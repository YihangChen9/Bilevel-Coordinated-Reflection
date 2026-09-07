from scenarios.overcooked.common import AgentState, Position
from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.mock_adapter import MockAdapter
from scenarios.overcooked.renderer import render_global, render_agent_view


def test_render_global_contains_key_info():
    adapter = MockAdapter()
    state = adapter.reset()
    text = render_global(state)
    assert "step 0" in text.lower()
    assert "score 0" in text.lower()
    assert "agent" in text.lower() or "A0" in text
    assert "pot" in text.lower() or "P0" in text


def test_render_agent_view_contains_agent_state():
    adapter = MockAdapter()
    state = adapter.reset()
    text = render_agent_view(state, agent_id=0)
    assert "Agent 0" in text or "agent 0" in text.lower()
    assert "holding" in text.lower() or "empty" in text.lower()


def test_render_global_shows_pot_cooking():
    adapter = MockAdapter()
    adapter.reset()
    adapter._pots[0].ingredients = 3
    adapter._pots[0].cooking_timer = 10
    state = adapter.get_state()
    text = render_global(state)
    assert "cooking" in text.lower()
