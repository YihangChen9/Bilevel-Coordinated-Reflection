"""Generate plots from Overcooked experiment results.

Per-config plots:
  - score bar chart (oracle / random / bilevel_llm)

Summary plot:
  - grouped bar chart of normalized score (% of oracle) across configs.

Usage:
  uv run python -m experiments.oc.plot               # all configs
  uv run python -m experiments.oc.plot --config cramped_room
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent / "figs"

METHOD_ORDER = ["oracle", "bilevel_llm", "random"]
METHOD_COLORS = {
    "oracle":      "#2ca02c",
    "bilevel_llm": "#1f77b4",
    "random":      "#d62728",
}


def _summarize(runs: list[dict], key: str) -> tuple[float, float]:
    if not runs:
        return 0.0, 0.0
    vals = np.array([r[key] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std() if len(vals) > 1 else 0.0)


def plot_score_bar(data: dict, out: Path) -> None:
    cfg = data["config"]
    results = data["results"]
    methods = [m for m in METHOD_ORDER if results.get(m)]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(methods))
    means = [_summarize(results[m], "score")[0] for m in methods]
    stds = [_summarize(results[m], "score")[1] for m in methods]
    bars = ax.bar(x, means, yerr=stds, color=[METHOD_COLORS[m] for m in methods],
                   capsize=4, edgecolor="black", linewidth=0.5)
    for bar, m, mean in zip(bars, methods, means):
        n_runs = len(results[m])
        ax.text(bar.get_x() + bar.get_width()/2, mean,
                f" {mean:.1f}\n n={n_runs}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("episode score (= 20 × deliveries)")
    ax.set_title(f"Overcooked — {cfg['name']}  (max_steps={cfg['config']['max_steps']})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def plot_deliveries_bar(data: dict, out: Path) -> None:
    cfg = data["config"]
    results = data["results"]
    methods = [m for m in METHOD_ORDER if results.get(m)]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(methods))
    means = [_summarize(results[m], "deliveries")[0] for m in methods]
    stds = [_summarize(results[m], "deliveries")[1] for m in methods]
    bars = ax.bar(x, means, yerr=stds, color=[METHOD_COLORS[m] for m in methods],
                   capsize=4, edgecolor="black", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean,
                f" {mean:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("# deliveries")
    ax.set_title(f"Overcooked — {cfg['name']}: deliveries")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def plot_summary_score(all_data: list[dict], out: Path) -> None:
    """Grouped bar chart: raw score per method × config (avoids divide-by-zero
    when oracle scores 0 on a layout)."""
    config_names = [d["config"]["name"] for d in all_data]
    methods_present = [m for m in METHOD_ORDER
                       if any(d["results"].get(m) for d in all_data)]

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(config_names)), 5))
    width = 0.8 / max(1, len(methods_present))
    x = np.arange(len(config_names))
    for i, m in enumerate(methods_present):
        means, stds = [], []
        for d in all_data:
            runs = d["results"].get(m, [])
            scores = [r["score"] for r in runs] if runs else [0]
            means.append(np.mean(scores))
            stds.append(np.std(scores) if len(scores) > 1 else 0)
        pos = x + (i - len(methods_present)/2 + 0.5) * width
        bars = ax.bar(pos, means, width=width, yerr=stds,
                      label=m, color=METHOD_COLORS[m], capsize=3,
                      edgecolor="black", linewidth=0.3)
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, mean,
                        f"{mean:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=10)
    ax.set_ylabel("episode score (= 20 × deliveries)")
    ax.set_title("Overcooked — score across layouts")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS_DIR.glob("*.json"))
    if args.config:
        files = [RESULTS_DIR / f"{args.config}.json"]
    if not files:
        print(f"no results in {RESULTS_DIR}. Run `python -m experiments.oc.run` first.",
              file=sys.stderr)
        sys.exit(1)

    all_data = []
    for f in files:
        data = json.loads(f.read_text())
        all_data.append(data)
        name = data["config"]["name"]
        plot_score_bar(data, FIGS_DIR / f"{name}_score.png")
        plot_deliveries_bar(data, FIGS_DIR / f"{name}_deliveries.png")

    if len(all_data) > 1:
        plot_summary_score(all_data, FIGS_DIR / "summary.png")


if __name__ == "__main__":
    main()
