"""Runs the full Resource Contest experiment suite.

For each config × method × seed:
  - runs the episode
  - dumps per-round rows + cumulative regret summary

Output: experiments/rc/results/<config_name>.json

Usage:
  uv run python -m experiments.rc.run                # all configs, with LLM
  uv run python -m experiments.rc.run --no-llm       # baselines only
  uv run python -m experiments.rc.run --config easy  # single config
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from experiments.rc.configs import CONFIGS, ExpConfig, get_config
from scenarios.resource_contest.baselines import (
    eps_greedy_policy, oracle_policy, random_policy, run_episode_with_policy,
    uniform_policy,
)
from scenarios.resource_contest.game import optimal_reward


RESULTS_DIR = Path(__file__).parent / "results"


BASELINE_METHODS = [
    ("oracle", lambda: oracle_policy),
    ("uniform", lambda: uniform_policy),
    ("random", lambda: random_policy),
    ("eps_greedy_0.1", lambda: eps_greedy_policy(epsilon=0.1, explore_rounds=1)),
]


def run_baselines(cfg: ExpConfig) -> dict[str, list[list[dict]]]:
    """Returns {method_name: [rows_per_seed]}."""
    out: dict[str, list[list[dict]]] = {}
    for name, mk_policy in BASELINE_METHODS:
        runs: list[list[dict]] = []
        for seed in cfg.seeds:
            policy = mk_policy()
            rows = run_episode_with_policy(cfg.params, T=cfg.T, policy=policy, seed=seed)
            runs.append(rows)
        out[name] = runs
        print(f"  [baseline] {name}: {len(runs)} seeds done", file=sys.stderr)
    return out


def run_llm(cfg: ExpConfig) -> list[list[dict]]:
    """Returns [rows_per_seed]. Single seed by default (LLM API cost)."""
    from scenarios.resource_contest.driver import run_episode as run_episode_llm

    runs: list[list[dict]] = []
    for seed in cfg.llm_seeds:
        workspace = Path(f"workspace/rc_exp/{cfg.name}/seed_{seed}").resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        rows = run_episode_llm(params=cfg.params, T=cfg.T, workspace=workspace)
        runs.append(rows)
        print(f"  [llm] seed={seed}: done", file=sys.stderr)
    return runs


def serialize_cfg(cfg: ExpConfig) -> dict:
    p = cfg.params
    return {
        "name": cfg.name,
        "T": cfg.T,
        "seeds": list(cfg.seeds),
        "llm_seeds": list(cfg.llm_seeds),
        "description": cfg.description,
        "params": {
            "n_agents": p.n_agents,
            "max_actions": list(p.max_actions),
            "budget": p.budget,
            "x_max": p.x_max,
            "cap_schedule": (
                [[t, list(caps)] for t, caps in p.cap_schedule]
                if p.cap_schedule else None
            ),
        },
        "R_star_t0": optimal_reward(p, 0),
    }


def run_config(cfg: ExpConfig, with_llm: bool) -> dict:
    t0 = time.time()
    print(f"\n=== config: {cfg.name} ({cfg.description}) ===", file=sys.stderr)
    results: dict[str, list[list[dict]]] = run_baselines(cfg)
    if with_llm:
        load_dotenv()
        try:
            results["bilevel_llm"] = run_llm(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  [llm] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results["bilevel_llm"] = []
    elapsed = time.time() - t0
    print(f"=== {cfg.name} done in {elapsed:.1f}s ===", file=sys.stderr)
    return {
        "config": serialize_cfg(cfg),
        "results": results,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the bilevel LLM run")
    ap.add_argument("--config", type=str, default=None,
                    help="run only this config (default: all)")
    args = ap.parse_args()

    if args.config:
        configs = [get_config(args.config)]
    else:
        configs = CONFIGS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        data = run_config(cfg, with_llm=not args.no_llm)
        out = RESULTS_DIR / f"{cfg.name}.json"
        out.write_text(json.dumps(data, indent=2, default=str))
        print(f"  → wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
