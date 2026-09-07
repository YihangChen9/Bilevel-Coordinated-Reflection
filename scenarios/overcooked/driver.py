"""Overcooked episode driver: macro-round loop with event-driven replanning."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from memory.markdown import MarkdownMemory
from orchestrator.llm import LLM, LiteLLM
from orchestrator.orchestrator import OrchContext, Subtask
from worker.hooks import inject_workboard_on_start
from worker.react import ReactAgent
from worker.worker import Worker

from scenarios.overcooked.common import Action, StepEvents
from scenarios.overcooked.config import OvercookedConfig
from scenarios.overcooked.renderer import render_global, render_agent_view, log_game_state
from scenarios.overcooked.role import COOK_WORKER
from scenarios.overcooked.strategy import OvercookedStrategy
from scenarios.overcooked.tools import make_overcooked_tools


def _steps_to_delivery(adapter):
    """Resolve min-steps-to-delivery in preference order:

    1. Ground-truth lookup table (deterministic, exact) — `gt_steps_to_delivery()`,
       requires `adapter.load_ground_truth_table(...)` to have been called.
    2. GreedyHumanModel rollout — `min_steps_to_delivery()`,  fallback when no
       GT table is loaded.  KNOWN ISSUE: stochastic + stateful, ±20 step noise.

    Returns int (≥1) when a delivery is reachable, else None.
    Treats the legacy `-1` sentinel as None to spare downstream code that
    historically negated it into J=+1.
    """
    if hasattr(adapter, "gt_steps_to_delivery"):
        v = adapter.gt_steps_to_delivery()
        if v is not None:
            return v
    if hasattr(adapter, "min_steps_to_delivery"):
        v = adapter.min_steps_to_delivery()
        if v is None or v < 0:
            return None
        return v
    return None


def run_episode_with_llms(
    adapter,
    planner,
    config: OvercookedConfig,
    orch_llm: LLM,
    make_worker_llm: Callable[[], LLM],
    workspace: Path,
    log_filename: str = ".orchestrator-log.md",
    workboard_filename: str = ".workboard.md",
    scratch_prefix: str = ".agent-",
    reset_orch_log: bool = False,
    reset_scratches: bool = False,
    gate: bool = True,
) -> dict:
    """Run one full Overcooked episode. Returns {score, steps, deliveries, replans}.

    ``gate`` toggles the J-ascent gated-write mechanism (Algorithm Phase II/III).
    When True (default), workboard writes and orch_log reflections are only
    committed if J strictly ascends; when False, every write is committed
    unconditionally (the non-gated ablation arm for the paper comparison).

    The driver:
    1. Resets the environment
    2. Calls orchestrator strategy to assign roles
    3. Dispatches workers to execute macro actions (each worker runs a short ReAct loop)
    4. After each worker batch completes, checks for replan events
    5. Repeats until episode ends (max_steps reached)
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    workboard = MarkdownMemory(workspace / workboard_filename)
    orch_log = MarkdownMemory(workspace / log_filename, persistent=True)
    scratches = [
        MarkdownMemory(workspace / f"{scratch_prefix}{i}.md", persistent=True)
        for i in range(config.n_agents)
    ]

    # Optional fresh-start: persistent memories normally accumulate across runs
    # (algorithm's slow-timescale strategy evolution). For test/debug runs that
    # want clean output, callers can reset these explicitly. `overwrite` bypasses
    # the persistent guard since the caller is asking for a clean slate.
    if reset_orch_log:
        orch_log.overwrite("")
    if reset_scratches:
        for s in scratches:
            s.overwrite("")

    game_log_path = workspace / "game_log.jsonl"
    if game_log_path.exists():
        game_log_path.unlink()

    strategy = OvercookedStrategy(
        llm=orch_llm, n_agents=config.n_agents, log_memory=orch_log,
    )

    state = adapter.reset()
    replan_reason = "episode start"
    n_replans = 0
    n_deliveries = 0

    # Gated-ascent (Algorithm: Bilevel Coordinated Reflection):
    # J(m_e) proxy = -gap_to_optimal_steps. We only commit a reflection to
    # orch_log (m_o) when actual delivery steps shrink the gap vs history best.
    best_delivery_gap: int | None = None
    optimal_at_replan: int | None = None
    step_at_replan: int = 0

    # J-history: per-worker-turn (step, agent, j_before, j_after, committed).
    # Collected regardless of `gate` so the paper's J table / convergence plot
    # can compare gated vs non-gated trajectories.
    j_history: list[dict] = []
    gate_stats = {"writes_seen": 0, "writes_committed": 0, "writes_discarded": 0}

    while not state.done:
        # Orchestrator plans (or replans)
        state_text = render_global(state)
        subtasks = strategy.decompose_with_state(state_text, replan_reason, OrchContext(task=state_text))
        n_replans += 1

        # Write plan to workboard
        workboard.clear()
        for sub in subtasks:
            workboard.append(f"agent_{sub.index}", sub.description)

        # Log the plan
        orch_log.append(
            f"OC step={state.step} replan={replan_reason}",
            f"state: {state_text}\nassignments: {json.dumps([s.description[:100] for s in subtasks])}",
        )

        # Log initial state for this planning window (with frame image if available)
        _replan_img = None
        if hasattr(adapter, "render_state_image"):
            frames_dir = game_log_path.parent / "frames"
            frames_dir.mkdir(exist_ok=True)
            img_file = frames_dir / f"step_{state.step:04d}.png"
            img_file.write_bytes(adapter.render_state_image())
            _replan_img = str(img_file.relative_to(game_log_path.parent))
        _opt = _steps_to_delivery(adapter)
        log_game_state(game_log_path, state, event=f"replan: {replan_reason}", frame_image=_replan_img, optimal_remaining=_opt)
        # Snapshot for gated-ascent comparison at next delivery
        optimal_at_replan = _opt if (_opt is not None and _opt > 0) else None
        step_at_replan = state.step

        # ── Multi-round planning (K rounds of observe + workboard negotiation) ──
        # Workers see the kitchen and each other's intentions via workboard BEFORE
        # executing any actions. Only observe/get_status/workboard/scratch tools are
        # available (no do_plan). This is the "cheap talk" phase.
        if config.n_planning_rounds > 1:
            for planning_round in range(config.n_planning_rounds - 1):
                for agent_id in range(config.n_agents):
                    worker_llm = make_worker_llm()
                    # Planning-only tools: observe + get_status (no do_plan)
                    plan_tools = [t for t in make_overcooked_tools(adapter, agent_id, planner)
                                  if t.name != "do_plan"]

                    planning_brief = (
                        f"[Planning round {planning_round + 1}/{config.n_planning_rounds}]\n"
                        f"You are NOT executing yet — just observing and coordinating.\n"
                        f"Read the workboard to see your teammates' intentions. "
                        f"Update the workboard with YOUR plan for this window.\n"
                        f"Respond final='planned' when done.\n\n"
                        + (subtasks[agent_id].description if agent_id < len(subtasks) else "flexible")
                    )

                    worker = Worker(
                        agent=ReactAgent(
                            llm=worker_llm, max_steps=4,
                            pre_step_hook=inject_workboard_on_start(workboard),
                        ),
                        role=COOK_WORKER,
                        tools=lambda _env, _t=plan_tools: list(_t),
                        env=None,
                        workboard=workboard,
                        agent_memory=scratches[agent_id],
                    )
                    worker.run(planning_brief)

                log_game_state(game_log_path, state,
                               event=f"planning round {planning_round + 1}/{config.n_planning_rounds}")

        # ── Execution loop: workers take macro actions until next replan event ──
        replan_reason = ""
        macro_rounds = 0
        max_macro_rounds = 50
        last_macro_cmd: dict[int, str] = {}  # track each agent's last macro for stuck detection

        while not state.done and not replan_reason and macro_rounds < max_macro_rounds:
            for agent_id in range(config.n_agents):
                if state.done:
                    break

                pos_before = state.agents[agent_id].pos if agent_id < len(state.agents) else None

                worker_llm = make_worker_llm()
                oc_tools = make_overcooked_tools(adapter, agent_id, planner, game_log_path=game_log_path)

                worker = Worker(
                    agent=ReactAgent(
                        llm=worker_llm, max_steps=5,
                        pre_step_hook=inject_workboard_on_start(workboard),
                    ),
                    role=COOK_WORKER,
                    tools=lambda _env, _t=oc_tools: list(_t),
                    env=None,
                    workboard=workboard,
                    agent_memory=scratches[agent_id],
                )

                sub = subtasks[agent_id] if agent_id < len(subtasks) else subtasks[-1]

                # ── Per-turn gated-ascent for workboard (Algorithm Phase II) ──
                # Snapshot workboard text + J = -min_steps_to_delivery() before
                # the worker acts. After the turn, accept the worker's workboard
                # writes only if J strictly ascended (gap to optimal shrunk).
                # The env action itself is irrevocable; only the markdown notes
                # get rolled back when the action didn't help.
                wb_snapshot = workboard.read()
                j_before = _steps_to_delivery(adapter)

                result = worker.run(sub.description)

                # Extract the last do_plan cmd from worker's steps
                for step in reversed(result.steps):
                    action = step.get("action") or {}
                    if action.get("tool") == "do_plan":
                        last_macro_cmd[agent_id] = (action.get("args") or {}).get("cmd", "")
                        break

                state = adapter.get_state()

                # Evaluate J after the turn. Lower min_steps = higher J.
                # If the worker modified the workboard but the gap didn't shrink,
                # discard the writes (rollback to snapshot). Skip the gate when
                # we can't compute J (mock adapters) or the env can't reach a
                # delivery (j == -1).
                if j_before is not None and j_before > 0:
                    j_after = _steps_to_delivery(adapter)
                    wb_after = workboard.read()
                    if wb_after != wb_snapshot:
                        gate_stats["writes_seen"] += 1
                        ascended = (
                            j_after is not None and j_after > 0 and j_after < j_before
                        )
                        committed = ascended or (not gate)
                        if not committed:
                            # gate ON and no ascent → roll the writes back
                            workboard.overwrite(wb_snapshot)
                            gate_stats["writes_discarded"] += 1
                            print(
                                f"workboard gated-write discarded: agent={agent_id} "
                                f"j_before={j_before} j_after={j_after} (no ascent)",
                                file=sys.stderr,
                            )
                        else:
                            gate_stats["writes_committed"] += 1
                            print(
                                f"workboard write {'gated-OK' if gate else 'ungated'}: "
                                f"agent={agent_id} j_before={j_before} j_after={j_after}",
                                file=sys.stderr,
                            )
                        j_history.append({
                            "step": int(state.step), "agent": int(agent_id),
                            "j_before": int(j_before),
                            "j_after": int(j_after) if j_after is not None else None,
                            "ascended": bool(ascended), "committed": bool(committed),
                        })

            # After all workers acted, check events
            state = adapter.get_state()
            if state.score > n_deliveries * config.delivery_reward:
                new_deliveries = int(state.score / config.delivery_reward) - n_deliveries
                n_deliveries += new_deliveries

                # ── Gated-ascent reflection write (Algorithm Phase III) ──
                # Compare actual delivery steps vs oracle-optimal at replan.
                # Only commit the reflection to orch_log when the gap shrinks
                # (J strictly ascends); otherwise discard to filter noisy plans.
                if optimal_at_replan is not None:
                    actual_steps = max(1, state.step - step_at_replan)
                    gap = actual_steps - optimal_at_replan
                    prev_best_str = (
                        str(best_delivery_gap) if best_delivery_gap is not None else "inf"
                    )
                    ascended = best_delivery_gap is None or gap < best_delivery_gap
                    if ascended or (not gate):
                        tag = "gated-write OK" if (gate and ascended) else "ungated"
                        orch_log.append(
                            f"OC reflection step={state.step} delivery#{n_deliveries} [{tag}]",
                            (
                                f"actual_steps={actual_steps}, optimal={optimal_at_replan}, "
                                f"gap={gap} (prev_best={prev_best_str}). "
                                f"Successful assignments: "
                                f"{json.dumps([s.description[:120] for s in subtasks])}"
                            ),
                        )
                        if ascended:
                            best_delivery_gap = gap
                    else:
                        print(
                            f"gated-write discarded: gap={gap} >= best_gap={prev_best_str}",
                            file=sys.stderr,
                        )

                replan_reason = f"delivery completed (total: {n_deliveries})"

            # Smart stuck detection:
            # Position unchanged for N steps AND last macro was NOT a deliberate wait
            for aid in range(config.n_agents):
                hist = adapter._position_history.get(aid, []) if hasattr(adapter, '_position_history') else []
                if len(hist) >= config.stuck_threshold:
                    recent = hist[-config.stuck_threshold:]
                    if len(set((p.x, p.y) for p in recent)) == 1:
                        last_cmd = last_macro_cmd.get(aid, "")
                        if last_cmd.startswith("wait"):
                            continue  # deliberate wait, not stuck
                        replan_reason = f"agent {aid} stuck (last cmd: {last_cmd})"
                        break

            macro_rounds += 1

        if not replan_reason and not state.done:
            replan_reason = "macro rounds exhausted"

    return {
        "score": state.score,
        "steps": state.step,
        "deliveries": n_deliveries,
        "replans": n_replans,
        "gate": gate,
        "gate_stats": gate_stats,
        "j_history": j_history,
    }


def run_episode(
    config: OvercookedConfig,
    workspace: Path | str = "workspace/overcooked",
    trace_path: Path | str | None = None,
    reset_orch_log: bool = False,
    reset_scratches: bool = False,
    gate: bool = True,
) -> dict:
    """Convenience wrapper with LiteLLM + real or mock adapter."""
    workspace = Path(workspace)
    if trace_path is None:
        trace_path = workspace / "llm_trace.jsonl"
    trace_path = Path(trace_path)
    if trace_path.exists():
        trace_path.unlink()

    # Prefer original overcooked_ai (best planner + pot readability),
    # fall back to jaxmarl, then MockAdapter.
    try:
        from scenarios.overcooked.original_adapter import OriginalAdapter
        from scenarios.overcooked.planner import MockPlanner
        adapter = OriginalAdapter(config)
        planner = MockPlanner(adapter)
        # Auto-load ground-truth V table if one exists for this layout.
        gt_path = Path("experiments/oc/ground_truth") / f"{config.layout}.pkl"
        if gt_path.exists():
            n = adapter.load_ground_truth_table(gt_path)
            print(f"Loaded GT V table from {gt_path} ({n:,} entries)", file=sys.stderr)
        print("Using original overcooked_ai environment", file=sys.stderr)
    except ImportError:
        try:
            from scenarios.overcooked.adapter import OvercookedAdapter, JaxPlanner
            adapter = OvercookedAdapter(config)
            planner = JaxPlanner(adapter)
            print("Using JaxMARL environment", file=sys.stderr)
        except ImportError:
            from scenarios.overcooked.mock_adapter import MockAdapter
            from scenarios.overcooked.planner import MockPlanner
            print("warning: no real env available, using MockAdapter", file=sys.stderr)
            adapter = MockAdapter(config)
            planner = MockPlanner(adapter)

    worker_llms: dict[int, LiteLLM] = {}

    def _make_worker_llm() -> LiteLLM:
        # Round-robin tagging
        if not hasattr(_make_worker_llm, "_counter"):
            _make_worker_llm._counter = 0
        i = _make_worker_llm._counter % config.n_agents
        _make_worker_llm._counter += 1
        if i not in worker_llms:
            worker_llms[i] = LiteLLM(trace_path=trace_path, tag=f"worker_{i}")
        return worker_llms[i]

    return run_episode_with_llms(
        adapter=adapter,
        planner=planner,
        config=config,
        orch_llm=LiteLLM(trace_path=trace_path, tag="orchestrator"),
        make_worker_llm=_make_worker_llm,
        workspace=workspace,
        reset_orch_log=reset_orch_log,
        reset_scratches=reset_scratches,
        gate=gate,
    )
