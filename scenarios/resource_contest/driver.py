"""RC episode driver — runs T rounds of Resource Contest with full bilevel LLM stack."""
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_REFLECTION_RE = re.compile(
    r"##\s*agent_(\d+)_reflection\s*\n(.*?)(?=^##|\Z)",
    re.DOTALL | re.MULTILINE,
)
_CAP_CLAIM_RE = re.compile(
    r"(?:M[_\s]*\d?|cap)\s*[=≥]\s*(\d+)|cap\s+is\s+(\d+)",
    re.IGNORECASE,
)


def _rollback_reflections(current: str, snapshot: str) -> str:
    """Roll the ## agent_<i>_reflection sections back to those in `snapshot`,
    preserving all other sections (notably ## agent_<i> action= lines) from
    `current`. Used by the gated-workboard ablation when J does not ascend."""
    # Strip reflection sections from current
    stripped = _REFLECTION_RE.sub("", current).rstrip() + "\n"
    # Extract reflection sections from snapshot
    snap_reflections = "".join(
        f"## agent_{m.group(1)}_reflection\n{m.group(2).rstrip()}\n\n"
        for m in _REFLECTION_RE.finditer(snapshot)
    )
    return (stripped.rstrip() + "\n\n" + snap_reflections).strip() + "\n"


def J_workboard(m_e_text: str) -> int:
    """Cheap surrogate for J(m_e) := E[U(x) | τ, m_e] in RC.

    Counts the number of agents whose `## agent_<i>_reflection` block contains
    a concrete cap claim (e.g., 'M=8', 'cap = 5', 'M >= 3'). Higher value =
    more discovered structure visible on the workboard. This is computable
    purely from text (no extra LLM call) and monotone-increases when workers
    publish new information about caps."""
    if not m_e_text:
        return 0
    agents_with_claim: set[int] = set()
    for m in _REFLECTION_RE.finditer(m_e_text):
        agent_id = int(m.group(1))
        body = m.group(2)
        if _CAP_CLAIM_RE.search(body):
            agents_with_claim.add(agent_id)
    return len(agents_with_claim)

from env.local import LocalEnv
from memory.markdown import MarkdownMemory
from orchestrator.llm import LLM, LiteLLM
from orchestrator.orchestrator import Orchestrator, Subtask
from worker.hooks import inject_workboard_on_start
from worker.react import ReactAgent
from worker.worker import Worker

from scenarios.resource_contest.game import (
    GameParams, reward, optimal_reward, regret, parse_action, clip_to_cap,
)
from scenarios.resource_contest.role import RC_PLAYER, RC_PLAYER_BLIND
from scenarios.resource_contest.strategy import RCStrategy
from scenarios.resource_contest.tools import AgentEnvState, make_rc_toolset_for_agent


@dataclass(frozen=True)
class AblationConfig:
    """Ablation switches for RC. The `default` instance reproduces the
    pre-ablation behaviour exactly (write everything, tail=10, clear workboard
    every round, workboard hook on, no episode-start resets beyond legacy
    reset_memory flag).
    """
    name: str = "default"
    # Group A — write/read end
    write_gate: str = "no_gate"          # "no_gate" | "j_ascent"
    last_k: int = 10                     # how many records strategy LLM is shown
    best_window: int | None = None       # window for best_observed_x (None = full last_k)
    # Group B — memory clear / hook
    clear_workboard_per_round: bool = True
    enable_workboard_hook: bool = True
    reset_scratches_at_start: bool = False
    reset_orch_log_at_start: bool = False
    # Group C — worker-side information minimality
    worker_blind: bool = False           # strip formula/cap/optimal-x leakage from worker prompt
    # Group D — algorithm-faithful J-gate on workboard reflection writes
    # When True, the driver snapshots workboard before each worker turn, lets
    # the worker call submit_reflection, then recomputes J(m_e) = "number of
    # discovered cap claims in workboard". If J does not strictly ascend after
    # the worker's reflection, the reflection text is rolled back (the
    # commit_action env effect is NOT rolled back).
    gate_workboard_writes: bool = False
    # Group D — also feed workboard contents (m_e) into the strategy LLM prompt,
    # so reflections written by workers actually influence the leader's
    # next-round τ decision. Closes the algorithmic Phase II → Phase III loop.
    strategy_reads_workboard: bool = False


ABLATION_DEFAULT = AblationConfig()


def run_episode_with_llms(
    params: GameParams,
    T: int,
    workspace: Path,
    orch_llm: LLM,
    make_worker_llm: Callable[[], LLM],
    log_filename: str = ".orchestrator-log.md",
    workboard_filename: str = ".workboard.md",
    scratch_prefix: str = ".agent-",
    reset_memory: bool = True,
    ablation: AblationConfig = ABLATION_DEFAULT,
) -> list[dict]:
    """Run T rounds. Returns one dict per round: {t, tau, x_input, x_effective, reward, regret}."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    workboard = MarkdownMemory(workspace / workboard_filename)
    orch_log = MarkdownMemory(workspace / log_filename, persistent=True)
    scratches = [
        MarkdownMemory(workspace / f"{scratch_prefix}{i}.md", persistent=True)
        for i in range(params.n_agents)
    ]
    # Legacy reset_memory wipes both; ablation flags allow fine-grained control.
    if reset_memory or ablation.reset_orch_log_at_start:
        orch_log.overwrite("")
    if reset_memory or ablation.reset_scratches_at_start:
        for s in scratches:
            s.overwrite("")

    env = LocalEnv(root=workspace)

    # Per-round, per-agent env state. Re-created every round.
    # `current_caps` is mutated by the main loop before each orch.run_pairs() call
    # so worker_factory always reads the active caps for the current round.
    current_env_states: list[AgentEnvState | None] = [None] * params.n_agents
    current_caps: list[int] = list(params.max_actions)

    def worker_factory(role, wb, sub: Subtask) -> Worker:
        i = sub.index
        state = AgentEnvState(cap=current_caps[i])
        current_env_states[i] = state
        rc_tools = make_rc_toolset_for_agent(
            workboard=wb, agent_id=i, env_state=state, x_max=params.x_max,
        )

        def toolset(_env):
            return list(rc_tools)

        pre_hook = inject_workboard_on_start(wb) if ablation.enable_workboard_hook else None

        return Worker(
            agent=ReactAgent(
                llm=make_worker_llm(),
                max_steps=8,
                pre_step_hook=pre_hook,
            ),
            role=role,
            tools=toolset,
            env=env,
            workboard=wb,
            agent_memory=scratches[i],
        )

    strategy = RCStrategy(
        llm=orch_llm,
        params=params,
        log_memory=orch_log,
        last_k=ablation.last_k,
        best_window=ablation.best_window,
        worker_blind=ablation.worker_blind,
        workboard_for_read=workboard if ablation.strategy_reads_workboard else None,
    )
    orch = Orchestrator(
        llm=orch_llm,
        worker_factory=worker_factory,
        workboard=workboard,
        decompose_strategy=strategy,
        log_memory=None,  # custom log entries below for parseable history
        max_parallel=params.n_agents,
        role_registry={"rc_player": RC_PLAYER, "rc_player_blind": RC_PLAYER_BLIND},
        clear_workboard_per_round=ablation.clear_workboard_per_round,
    )

    rows: list[dict] = []
    best_reward_so_far: float | None = None  # for ablation.write_gate == "j_ascent"
    j_persistent: int = 0  # J(m_e) accumulated across rounds for gated-workboard ablation
    n_reflections_kept: int = 0
    n_reflections_rolled_back: int = 0

    for t in range(T):
        # Reset env-states and refresh caps for non-stationary envs
        current_env_states[:] = [None] * params.n_agents
        new_caps = params.caps_at(t)
        for i, c in enumerate(new_caps):
            current_caps[i] = c
        R_star_t = optimal_reward(params, t)

        # Group D — algorithm-faithful J-gate (Phase II).
        # Snapshot workboard BEFORE this round's worker batch. After they run,
        # if J(m_e_after) did not strictly ascend over j_persistent, roll the
        # ## agent_<i>_reflection sections back to the snapshot. commit_action
        # side effects (## agent_<i> action=K) are kept — only reflection
        # candidates c_i are gated.
        if ablation.gate_workboard_writes:
            wb_snapshot_pre = workboard.read()
        else:
            wb_snapshot_pre = None

        pairs = orch.run_pairs(f"round {t}")
        tau = list(strategy.last_tau) or [params.budget / params.n_agents] * params.n_agents

        if ablation.gate_workboard_writes and wb_snapshot_pre is not None:
            j_after = J_workboard(workboard.read())
            if j_after > j_persistent:
                j_persistent = j_after
                n_reflections_kept += 1
                print(
                    f"  workboard gated-write OK: J {j_persistent} (round {t})",
                    file=sys.stderr,
                )
            else:
                # Rollback ONLY the reflection sections, preserve commit_action lines.
                new_wb_text = _rollback_reflections(workboard.read(), wb_snapshot_pre)
                workboard.overwrite(new_wb_text)
                n_reflections_rolled_back += 1
                print(
                    f"  workboard gated-write DISCARDED: J unchanged at {j_persistent} (round {t})",
                    file=sys.stderr,
                )

        # Authoritative effective_x comes from env_state (what env actually used),
        # not from parsing the worker's final answer — agents may hallucinate.
        x_input = [0] * params.n_agents
        x_eff = [0] * params.n_agents
        for i, st in enumerate(current_env_states):
            if st is None:
                # Worker factory wasn't called for this agent (shouldn't happen) —
                # treat as zero commit.
                continue
            x_input[i] = st.last_input if st.last_input is not None else 0
            x_eff[i] = st.last_effective

        # Cross-check: if a worker never called commit_action, their final answer
        # might still be a number — fall back to parsing + clipping.
        for idx, body in pairs:
            if current_env_states[idx] is None or current_env_states[idx].attempts == 0:
                a = parse_action(body, lo=0, hi=params.x_max)
                if a is not None:
                    x_input[idx] = a
                    x_eff[idx], _ = clip_to_cap(a, new_caps[idx])

        r = reward(tau, x_eff)
        g = regret(r, params, t)

        # Write-gate (Group A ablation): "j_ascent" only appends when r strictly
        # exceeds the running best. "no_gate" always appends. Either way, the row
        # is always recorded in the returned trace — gating only affects what the
        # strategy LLM sees on the next round.
        if ablation.write_gate == "j_ascent":
            gated_in = best_reward_so_far is None or r > best_reward_so_far
        else:
            gated_in = True
        if gated_in:
            orch_log.append(
                f"RC t={t}",
                f"reward={r:.4f} tau={[round(t_,3) for t_ in tau]} x={x_eff} x_input={x_input}",
            )
        if best_reward_so_far is None or r > best_reward_so_far:
            best_reward_so_far = r

        print(
            f"  t={t}: tau={[f'{x:.2f}' for x in tau]}  x_eff={x_eff}  "
            f"reward={r:.3f}  regret={g:.3f}  gated_in={gated_in}",
            file=sys.stderr,
        )

        rows.append({
            "t": t, "tau": tau, "x_input": x_input, "x_effective": x_eff,
            "reward": r, "regret": g, "optimal": R_star_t,
            "gated_in_orch_log": gated_in,
            "j_workboard": j_persistent,
        })

    if ablation.gate_workboard_writes:
        print(
            f"workboard gating summary: J_final={j_persistent}, "
            f"kept={n_reflections_kept}, rolled_back={n_reflections_rolled_back}",
            file=sys.stderr,
        )
    return rows


def run_episode(
    params: GameParams,
    T: int,
    workspace: Path | str = "workspace/rc",
    trace_path: Path | str | None = None,
    reset_memory: bool = True,
    ablation: AblationConfig = ABLATION_DEFAULT,
    model: str | None = None,
) -> list[dict]:
    """LiteLLM-backed convenience wrapper.

    If `model` is provided, it overrides the LLM_MODEL env var for this episode's
    orchestrator + worker LLMs (used by the ablation runner to fan out across
    multiple models in one process).
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if trace_path is None:
        trace_path = workspace / "llm_trace.jsonl"
    trace_path = Path(trace_path)
    if trace_path.exists():
        trace_path.unlink()

    worker_llms: dict[int, LiteLLM] = {}
    counter = {"n": 0}

    def _next_worker_llm() -> LiteLLM:
        i = counter["n"] % params.n_agents
        counter["n"] += 1
        if i not in worker_llms:
            worker_llms[i] = LiteLLM(model=model, trace_path=trace_path, tag=f"worker_{i}")
        return worker_llms[i]

    return run_episode_with_llms(
        params=params,
        T=T,
        workspace=workspace,
        orch_llm=LiteLLM(model=model, trace_path=trace_path, tag="orchestrator"),
        make_worker_llm=_next_worker_llm,
        reset_memory=reset_memory,
        ablation=ablation,
    )
