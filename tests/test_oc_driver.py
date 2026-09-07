import json
from pathlib import Path

from orchestrator.llm import FakeLLM
from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.driver import run_episode_with_llms
from scenarios.overcooked.mock_adapter import MockAdapter
from scenarios.overcooked.planner import MockPlanner


def _worker_script_pickup_and_done():
    """Worker: observe → do_plan(pickup ingredient_0) → final done."""
    return [
        json.dumps({"thought": "check state", "action": {"tool": "observe", "args": {}}}),
        json.dumps({"thought": "go get ingredient", "action": {"tool": "do_plan", "args": {"cmd": "pickup ingredient_0"}}}),
        json.dumps({"thought": "done for now", "final": "done"}),
    ]


def test_driver_runs_without_crash(tmp_path: Path):
    """Basic integration: driver runs a short episode without errors."""
    config = OvercookedConfig(max_steps=20, cook_time=3, n_agents=2, stuck_threshold=15)
    adapter = MockAdapter(config)
    planner = MockPlanner(adapter)

    # Need enough LLM scripts: orchestrator (1 plan) + workers (2 per macro round) × several rounds
    orch_scripts = [
        json.dumps({
            "assignments": [
                {"agent_id": 0, "role": "gatherer", "detail": "pickup ingredients"},
                {"agent_id": 1, "role": "deliverer", "detail": "plate and deliver"},
            ],
            "reasoning": "split work",
        }),
    ] * 5  # enough for replans

    worker_scripts = [_worker_script_pickup_and_done() for _ in range(20)]  # enough for all rounds
    worker_iter = iter(worker_scripts)

    result = run_episode_with_llms(
        adapter=adapter,
        planner=planner,
        config=config,
        orch_llm=FakeLLM(orch_scripts),
        make_worker_llm=lambda: FakeLLM(next(worker_iter)),
        workspace=tmp_path,
    )

    assert "score" in result
    assert "steps" in result
    assert result["steps"] <= config.max_steps
    assert result["replans"] >= 1

    # orch_log should have at least one entry
    log = (tmp_path / ".orchestrator-log.md").read_text(encoding="utf-8")
    assert "OC step=" in log
