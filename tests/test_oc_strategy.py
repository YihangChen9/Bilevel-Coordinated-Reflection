import json

from memory.markdown import MarkdownMemory
from orchestrator.llm import FakeLLM
from orchestrator.orchestrator import OrchContext
from scenarios.overcooked.strategy import OvercookedStrategy


def test_strategy_returns_n_subtasks(tmp_path):
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)
    llm = FakeLLM([json.dumps({
        "assignments": [
            {"agent_id": 0, "role": "gatherer", "detail": "onions to pot_0"},
            {"agent_id": 1, "role": "deliverer", "detail": "plate and serve"},
        ],
        "reasoning": "split pipeline",
    })])
    strat = OvercookedStrategy(llm=llm, n_agents=2, log_memory=log)
    subs = strat.decompose("Kitchen 5x5, step 0...", OrchContext(task="t"))
    assert len(subs) == 2
    assert subs[0].role_name == "cook_worker"
    assert "gatherer" in subs[0].description
    assert "deliverer" in subs[1].description


def test_strategy_fallback_on_bad_llm_output(tmp_path):
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)
    llm = FakeLLM(["this is not json at all"])
    strat = OvercookedStrategy(llm=llm, n_agents=2, log_memory=log)
    subs = strat.decompose("state", OrchContext(task="t"))
    assert len(subs) == 2
    assert "flexible" in subs[0].description


def test_strategy_decompose_with_state(tmp_path):
    log = MarkdownMemory(tmp_path / "log.md", persistent=True)
    llm = FakeLLM([json.dumps({
        "assignments": [
            {"agent_id": 0, "role": "gatherer", "detail": "get onions"},
        ],
        "reasoning": "replan after delivery",
    })])
    strat = OvercookedStrategy(llm=llm, n_agents=1, log_memory=log)
    subs = strat.decompose_with_state("kitchen state", "delivery completed", OrchContext(task="t"))
    assert len(subs) == 1
    assert strat.last_state_text == "kitchen state"
