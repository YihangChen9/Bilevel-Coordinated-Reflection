"""OC gate ablation for the two paper figures.

For each config, runs the bilevel-LLM episode with the J-ascent write gate ON and
OFF (multiple seeds), recording score / deliveries / gate_stats / j_history. Also
runs oracle + random baselines for reference. Output:
experiments/oc/results_gate/<config>.json.

  fig 1 (gate vs no-gate): deliveries/score, gate ON vs OFF, per layout.
  fig 2 (beta regression):  convergence exponent fit on the per-turn J trajectory.

Usage:
  python -m experiments.oc.run_gate                    # all configs, both gates
  python -m experiments.oc.run_gate --config cramped_room
  python -m experiments.oc.run_gate --seeds 0 1 2      # more LLM seeds
  python -m experiments.oc.run_gate --no-baselines
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from experiments.oc.baselines import run_oracle_episode, run_random_episode
from experiments.oc.configs import CONFIGS, get_config

RESULTS_DIR = Path(__file__).parent / "results_gate"


def run_llm_gate(cfg, gate: bool, seeds) -> list[dict]:
    from scenarios.overcooked.driver import run_episode
    runs = []
    for seed in seeds:
        tag = "gate" if gate else "nogate"
        workspace = Path(f"workspace/oc_gate/{cfg.name}/{tag}/seed_{seed}").resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = run_episode(
                config=cfg.config, workspace=workspace,
                reset_orch_log=True, reset_scratches=True, gate=gate,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [llm {tag} seed={seed}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        r["elapsed_sec"] = time.time() - t0
        r["seed"] = seed
        print(f"  [llm {tag} seed={seed}] {time.time()-t0:.0f}s "
              f"deliveries={r.get('deliveries')} score={r.get('score')} "
              f"gate_stats={r.get('gate_stats')}", file=sys.stderr)
        runs.append(r)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    configs = [get_config(args.config)] if args.config else CONFIGS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        print(f"\n=== {cfg.name} ({cfg.description}) ===", file=sys.stderr)
        data = {"config": cfg.name, "layout": cfg.config.layout,
                "description": cfg.description, "seeds": args.seeds}
        if not args.no_baselines:
            try:
                data["oracle"] = run_oracle_episode(cfg.config)
            except Exception as e:  # noqa: BLE001
                print(f"  [oracle] FAILED: {e}", file=sys.stderr); data["oracle"] = None
            data["random"] = []
            for s in cfg.random_seeds:
                try:
                    data["random"].append(run_random_episode(cfg.config, seed=s))
                except Exception as e:  # noqa: BLE001
                    print(f"  [random seed={s}] FAILED: {e}", file=sys.stderr)
        data["gate_on"] = run_llm_gate(cfg, gate=True, seeds=args.seeds)
        data["gate_off"] = run_llm_gate(cfg, gate=False, seeds=args.seeds)
        out = RESULTS_DIR / f"{cfg.name}.json"
        out.write_text(json.dumps(data, indent=2, default=str))
        print(f"  → wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
