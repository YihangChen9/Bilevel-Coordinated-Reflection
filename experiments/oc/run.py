"""Runs the Overcooked experiment suite.

For each config × method:
  - executes the episode
  - records {score, deliveries, steps, max_steps}

Output: experiments/oc/results/<config_name>.json

Usage:
  uv run python -m experiments.oc.run                  # all configs, with LLM
  uv run python -m experiments.oc.run --no-llm         # baselines only (fast)
  uv run python -m experiments.oc.run --config cramped_room
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from experiments.oc.baselines import run_oracle_episode, run_random_episode
from experiments.oc.configs import CONFIGS, OCExpConfig, get_config


RESULTS_DIR = Path(__file__).parent / "results"


def run_baselines(cfg: OCExpConfig) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    # Oracle is deterministic — one run.
    print(f"  [oracle] running...", file=sys.stderr)
    t0 = time.time()
    out["oracle"] = [run_oracle_episode(cfg.config)]
    print(f"  [oracle] done ({time.time()-t0:.1f}s): {out['oracle'][0]}", file=sys.stderr)

    # Random — multi-seed for variance.
    out["random"] = []
    for seed in cfg.random_seeds:
        t0 = time.time()
        r = run_random_episode(cfg.config, seed=seed)
        out["random"].append(r)
        print(f"  [random seed={seed}] done ({time.time()-t0:.1f}s): {r}", file=sys.stderr)

    return out


def run_llm(cfg: OCExpConfig) -> list[dict]:
    from scenarios.overcooked.driver import run_episode

    runs: list[dict] = []
    for seed in cfg.llm_seeds:
        workspace = Path(f"workspace/oc_exp/{cfg.name}/seed_{seed}").resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = run_episode(
            config=cfg.config, workspace=workspace,
            reset_orch_log=True, reset_scratches=True,
        )
        elapsed = time.time() - t0
        print(f"  [llm seed={seed}] done ({elapsed:.1f}s): {r}", file=sys.stderr)
        r["elapsed_sec"] = elapsed
        runs.append(r)
    return runs


def serialize_cfg(cfg: OCExpConfig) -> dict:
    c = cfg.config
    return {
        "name": cfg.name,
        "description": cfg.description,
        "random_seeds": list(cfg.random_seeds),
        "llm_seeds": list(cfg.llm_seeds),
        "config": {
            "layout": c.layout,
            "max_steps": c.max_steps,
            "n_agents": c.n_agents,
            "cook_time": c.cook_time,
            "recipe_size": c.recipe_size,
            "delivery_reward": c.delivery_reward,
        },
    }


def run_config(cfg: OCExpConfig, with_llm: bool) -> dict:
    t0 = time.time()
    print(f"\n=== {cfg.name} ({cfg.description}) ===", file=sys.stderr)
    results = run_baselines(cfg)
    if with_llm:
        load_dotenv()
        try:
            results["bilevel_llm"] = run_llm(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  [llm] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results["bilevel_llm"] = []
    elapsed = time.time() - t0
    print(f"=== {cfg.name} done in {elapsed:.1f}s ===", file=sys.stderr)
    return {"config": serialize_cfg(cfg), "results": results, "elapsed_sec": elapsed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    configs = [get_config(args.config)] if args.config else CONFIGS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        data = run_config(cfg, with_llm=not args.no_llm)
        out = RESULTS_DIR / f"{cfg.name}.json"
        out.write_text(json.dumps(data, indent=2, default=str))
        print(f"  → wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
