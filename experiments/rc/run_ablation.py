"""RC ablation runner — sweeps Group A (write/read gate) and Group B (memory clear) variants.

For each (model, variant, config) it runs one RC episode and writes:
  experiments/rc/ablation_results/<model_slug>/<variant>/<config>.json

with fields: model, variant, ablation, config, rows, summary, elapsed_sec.

Workspaces (full trace, including .workboard.md / .agent-*.md / .orchestrator-log.md /
llm_trace.jsonl) are kept under:
  workspace/rc_ablation/<model_slug>/<variant>/<config>/seed_<S>/

Concurrency: --concurrency N spawns N episodes in parallel (default 4). Each
episode is fully isolated via its own workspace dir and its own LiteLLM instance,
so concurrent runs don't share state.

Usage:
  uv run python -m experiments.rc.run_ablation \
      --models Kimi-K2.6 \
      --concurrency 4

  # Smoke test
  uv run python -m experiments.rc.run_ablation \
      --models Kimi-K2.6 --variants j_ascent --configs easy --concurrency 1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from experiments.rc.configs import CONFIGS, ExpConfig, get_config
from scenarios.resource_contest.driver import AblationConfig, run_episode


# ────────────────────────────────────────────────────────────────────────────
# Variant matrix
# ────────────────────────────────────────────────────────────────────────────

# Group A — write/read gate.    All 4 configs.
# Group B — memory clear/hook.  easy + drift only.
VARIANTS: dict[str, AblationConfig] = {
    # baseline (shared by both groups)
    "default": AblationConfig(name="default"),
    # Group A
    "j_ascent":         AblationConfig(name="j_ascent", write_gate="j_ascent"),
    "recency_k5":       AblationConfig(name="recency_k5", last_k=5),
    "sliding_max_k5":   AblationConfig(name="sliding_max_k5", last_k=10, best_window=5),
    # Group B
    "workboard_off":    AblationConfig(name="workboard_off", enable_workboard_hook=False),
    "workboard_keep":   AblationConfig(name="workboard_keep", clear_workboard_per_round=False),
    "scratch_reset":    AblationConfig(name="scratch_reset", reset_scratches_at_start=True),
    "orch_log_reset":   AblationConfig(name="orch_log_reset", reset_orch_log_at_start=True),
    # Group C — worker-side information minimality
    "worker_blind":     AblationConfig(name="worker_blind", worker_blind=True),
    # Group D — algorithm-faithful J-gate on workboard reflection writes.
    # Reflections must persist across rounds for the gate decision to matter,
    # so we also disable per-round workboard clearing here. commit_action
    # side-effects (`## agent_<i>` action= lines) get overwritten in place by
    # the next round's commit, so they don't accumulate; only the gated
    # `## agent_<i>_reflection` sections build up monotonically.
    "gated_workboard":  AblationConfig(
        name="gated_workboard",
        gate_workboard_writes=True,
        clear_workboard_per_round=False,
    ),
    # Group D — fully closed algorithm loop: J-gate on workboard + strategy
    # actually reads workboard. The reflections workers write now propagate
    # to the leader's next-round τ decision.
    "algorithm_complete": AblationConfig(
        name="algorithm_complete",
        gate_workboard_writes=True,
        clear_workboard_per_round=False,
        strategy_reads_workboard=True,
    ),
}

GROUP_A = ["default", "j_ascent", "recency_k5", "sliding_max_k5"]
GROUP_B = ["default", "workboard_off", "workboard_keep", "scratch_reset", "orch_log_reset"]
GROUP_C = ["default", "worker_blind"]
GROUP_D = ["default", "gated_workboard", "algorithm_complete"]
GROUP_A_CONFIGS = ["easy", "hard", "many", "drift"]
GROUP_B_CONFIGS = ["easy", "drift"]
GROUP_C_CONFIGS = ["easy", "hard", "many", "drift"]
GROUP_D_CONFIGS = ["easy", "hard", "many", "drift"]


def build_jobs(models: list[str], variants: list[str] | None, configs: list[str] | None) -> list[dict]:
    """Returns a list of {model, variant, config} jobs based on the variant/config
    membership of each variant in Group A vs Group B."""
    jobs: list[dict] = []
    for model in models:
        for v_name in VARIANTS:
            if variants is not None and v_name not in variants:
                continue
            # Pick the config set per group; `default` belongs to both.
            cfg_names: set[str] = set()
            if v_name in GROUP_A:
                cfg_names.update(GROUP_A_CONFIGS)
            if v_name in GROUP_B:
                cfg_names.update(GROUP_B_CONFIGS)
            if v_name in GROUP_C:
                cfg_names.update(GROUP_C_CONFIGS)
            if v_name in GROUP_D:
                cfg_names.update(GROUP_D_CONFIGS)
            for c_name in sorted(cfg_names):
                if configs is not None and c_name not in configs:
                    continue
                jobs.append({"model": model, "variant": v_name, "config": c_name})
    return jobs


# ────────────────────────────────────────────────────────────────────────────
# One job
# ────────────────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def run_one(job: dict, *, results_root: Path, workspace_root: Path) -> dict:
    model = job["model"]
    variant = job["variant"]
    cfg_name = job["config"]
    ablation = VARIANTS[variant]
    cfg = get_config(cfg_name)
    seed = cfg.llm_seeds[0]  # 1 seed per ablation cell

    model_slug = slugify(model)
    workspace = workspace_root / model_slug / variant / cfg_name / f"seed_{seed}"
    # Wipe any leftover state from a prior run on this cell — orch_log /
    # scratches / workboard / llm_trace contaminate fresh ablation if kept.
    # The driver itself only unlinks llm_trace; persistent memories survive,
    # which biases the LLM with prior observations.
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    result_dir = results_root / model_slug / variant
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{cfg_name}.json"

    t0 = time.time()
    error: str | None = None
    rows: list[dict] = []
    try:
        rows = run_episode(
            params=cfg.params,
            T=cfg.T,
            workspace=workspace,
            ablation=ablation,
            model=model,
            reset_memory=False,  # let ablation flags drive resets
        )
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[FAIL] {model}/{variant}/{cfg_name}: {error[:200]}", file=sys.stderr)

    elapsed = time.time() - t0
    total_reward = sum(r["reward"] for r in rows) if rows else 0.0
    optimal_total = sum(r["optimal"] for r in rows) if rows else 0.0
    final_regret = optimal_total - total_reward

    out = {
        "model": model,
        "variant": variant,
        "ablation": asdict(ablation),
        "config": {
            "name": cfg.name, "T": cfg.T, "description": cfg.description,
            "n_agents": cfg.params.n_agents,
            "max_actions": list(cfg.params.max_actions),
            "budget": cfg.params.budget,
            "x_max": cfg.params.x_max,
            "cap_schedule": (
                [[t, list(caps)] for t, caps in cfg.params.cap_schedule]
                if cfg.params.cap_schedule else None
            ),
        },
        "seed": seed,
        "summary": {
            "total_reward": total_reward,
            "optimal_total": optimal_total,
            "final_regret": final_regret,
            "n_rounds_recorded": len(rows),
            "n_rounds_gated_in": sum(1 for r in rows if r.get("gated_in_orch_log", True)),
        },
        "rows": rows,
        "elapsed_sec": elapsed,
        "error": error,
        "workspace": str(workspace),
    }
    result_path.write_text(json.dumps(out, indent=2, default=str))
    status = "OK" if error is None else "FAIL"
    print(
        f"[{status}] {model}/{variant}/{cfg_name}: total_reward={total_reward:.1f} "
        f"regret={final_regret:.2f} elapsed={elapsed:.0f}s gated_in={out['summary']['n_rounds_gated_in']}/{len(rows)}",
        file=sys.stderr,
    )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Kimi-K2.6"],
                    help="Model name(s) as known by the LLM proxy (LLM_MODEL).")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variants to run (default: all in VARIANTS).")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Subset of configs to run (default: per-group defaults).")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Number of episodes to run in parallel.")
    ap.add_argument("--dry-run", action="store_true", help="Print the job list and exit.")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "ablation_results")
    ap.add_argument("--workspace-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "workspace" / "rc_ablation")
    args = ap.parse_args()

    load_dotenv()

    jobs = build_jobs(args.models, args.variants, args.configs)
    print(f"Planned {len(jobs)} jobs across {len(args.models)} model(s):", file=sys.stderr)
    for j in jobs:
        print(f"  - {j['model']}/{j['variant']}/{j['config']}", file=sys.stderr)
    if args.dry_run or not jobs:
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.workspace_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    completed: list[dict] = []
    failed: list[dict] = []

    if args.concurrency <= 1:
        for job in jobs:
            r = run_one(job, results_root=args.results_dir, workspace_root=args.workspace_dir)
            (failed if r.get("error") else completed).append(r)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(run_one, job,
                          results_root=args.results_dir,
                          workspace_root=args.workspace_dir): job
                for job in jobs
            }
            for fut in cf.as_completed(futures):
                r = fut.result()
                (failed if r.get("error") else completed).append(r)

    elapsed = time.time() - t0
    print(f"\n=== ALL DONE in {elapsed:.0f}s — {len(completed)} OK, {len(failed)} FAIL ===",
          file=sys.stderr)
    # Write a summary index.
    index = {
        "models": args.models,
        "n_jobs": len(jobs),
        "n_ok": len(completed),
        "n_fail": len(failed),
        "elapsed_sec": elapsed,
        "jobs": [
            {
                "model": r["model"], "variant": r["variant"],
                "config": r["config"]["name"],
                "total_reward": r["summary"]["total_reward"],
                "final_regret": r["summary"]["final_regret"],
                "error": r.get("error"),
            }
            for r in (completed + failed)
        ],
    }
    (args.results_dir / "index.json").write_text(json.dumps(index, indent=2, default=str))
    print(f"→ {args.results_dir / 'index.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
