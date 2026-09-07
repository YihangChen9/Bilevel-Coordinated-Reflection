"""Generate plots from experiment results.

Produces three figures per config:
  1. cumulative regret curves (one line per method, ± std band over seeds)
  2. per-round reward curves
  3. final τ allocation as a stacked bar across methods

Plus one summary figure across all configs:
  - bar chart of final cumulative regret per method × config

Usage:
  uv run python -m experiments.rc.plot                # plot all configs in results/
  uv run python -m experiments.rc.plot --config easy  # one config
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

METHOD_ORDER = ["oracle", "bilevel_llm", "eps_greedy_0.1", "uniform", "random"]
METHOD_COLORS = {
    "oracle":         "#2ca02c",
    "bilevel_llm":    "#1f77b4",
    "eps_greedy_0.1": "#ff7f0e",
    "uniform":        "#7f7f7f",
    "random":         "#d62728",
}
METHOD_STYLES = {
    "oracle":         {"linestyle": "--", "linewidth": 2},
    "bilevel_llm":    {"linestyle": "-",  "linewidth": 2.5},
    "eps_greedy_0.1": {"linestyle": "-",  "linewidth": 1.5},
    "uniform":        {"linestyle": ":",  "linewidth": 1.5},
    "random":         {"linestyle": ":",  "linewidth": 1.5},
}


def _cum(values: list[float]) -> np.ndarray:
    return np.cumsum(np.array(values, dtype=float))


def _stack(runs: list[list[dict]], key: str) -> np.ndarray:
    """Return (n_seeds, T) array of `key` values."""
    return np.array([[row[key] for row in run] for run in runs], dtype=float)


def _plot_curve(ax, runs: list[list[dict]], method: str, transform=lambda a: a):
    if not runs:
        return
    vals = _stack(runs, "regret")
    cum = np.cumsum(vals, axis=1)
    cum = transform(cum)
    t = np.arange(cum.shape[1])
    mean = cum.mean(axis=0)
    style = METHOD_STYLES.get(method, {})
    color = METHOD_COLORS.get(method, None)
    ax.plot(t, mean, label=method, color=color, **style)
    if cum.shape[0] > 1:
        std = cum.std(axis=0)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)


def plot_cumulative_regret(data: dict, out: Path) -> None:
    cfg = data["config"]
    results = data["results"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHOD_ORDER:
        runs = results.get(method, [])
        _plot_curve(ax, runs, method)
    if cfg["params"].get("cap_schedule"):
        for t_start, _ in cfg["params"]["cap_schedule"]:
            ax.axvline(t_start, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(t_start, ax.get_ylim()[1] * 0.95, f" cap shift @ t={t_start}",
                    fontsize=8, va="top")
    ax.set_xlabel("round t")
    ax.set_ylabel("cumulative regret  $G_T = \\sum_t (R^* - r_t)$")
    ax.set_title(f"Resource Contest — {cfg['name']}  (N={cfg['params']['n_agents']}, "
                 f"M={cfg['params']['max_actions']}, T={cfg['T']})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def plot_reward_curve(data: dict, out: Path) -> None:
    cfg = data["config"]
    results = data["results"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    R_star_t0 = cfg["R_star_t0"]
    ax.axhline(R_star_t0, color="green", linestyle=":", linewidth=1,
               label=f"$R^*$ at t=0 = {R_star_t0:.2f}", alpha=0.6)
    for method in METHOD_ORDER:
        runs = results.get(method, [])
        if not runs:
            continue
        vals = _stack(runs, "reward")
        mean = vals.mean(axis=0)
        t = np.arange(len(mean))
        style = METHOD_STYLES.get(method, {})
        color = METHOD_COLORS.get(method, None)
        ax.plot(t, mean, label=method, color=color, **style)
        if vals.shape[0] > 1:
            std = vals.std(axis=0)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)
    if cfg["params"].get("cap_schedule"):
        for t_start, _ in cfg["params"]["cap_schedule"]:
            ax.axvline(t_start, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("round t")
    ax.set_ylabel("per-round reward $r_t = \\sum \\tau_i x_i$")
    ax.set_title(f"Resource Contest — {cfg['name']}: per-round reward")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def plot_final_tau(data: dict, out: Path) -> None:
    cfg = data["config"]
    results = data["results"]
    n_agents = cfg["params"]["n_agents"]
    methods_present = [m for m in METHOD_ORDER if results.get(m)]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * len(methods_present)))
    y = np.arange(len(methods_present))
    bottoms = np.zeros(len(methods_present))
    cmap = plt.get_cmap("tab10")
    for i in range(n_agents):
        vals = []
        for m in methods_present:
            runs = results[m]
            final_taus = np.array([run[-1]["tau"][i] for run in runs])
            vals.append(final_taus.mean())
        vals_arr = np.array(vals)
        ax.barh(y, vals_arr, left=bottoms, color=cmap(i),
                label=f"agent_{i} (M={cfg['params']['max_actions'][i]})")
        for j, v in enumerate(vals_arr):
            if v > 0.05:
                ax.text(bottoms[j] + v / 2, y[j], f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.15 else "black")
        bottoms += vals_arr
    ax.set_yticks(y)
    ax.set_yticklabels(methods_present)
    ax.invert_yaxis()
    ax.set_xlim(0, cfg["params"]["budget"])
    ax.set_xlabel("τ allocation at t=T-1 (mean over seeds)")
    ax.set_title(f"Resource Contest — {cfg['name']}: final τ")
    ax.legend(loc="lower right", fontsize=8, ncol=n_agents)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}", file=sys.stderr)


def plot_summary_across_configs(all_data: list[dict], out: Path) -> None:
    """Bar chart: cumulative regret at end (G_T) per method × config."""
    config_names = [d["config"]["name"] for d in all_data]
    methods_present: list[str] = []
    for m in METHOD_ORDER:
        if any(d["results"].get(m) for d in all_data):
            methods_present.append(m)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(config_names)), 5))
    x = np.arange(len(config_names))
    width = 0.8 / max(1, len(methods_present))
    for i, m in enumerate(methods_present):
        means, stds = [], []
        for d in all_data:
            runs = d["results"].get(m, [])
            if not runs:
                means.append(0); stds.append(0); continue
            cum = np.array([sum(row["regret"] for row in run) for run in runs])
            means.append(cum.mean())
            stds.append(cum.std() if len(cum) > 1 else 0)
        pos = x + (i - len(methods_present) / 2 + 0.5) * width
        ax.bar(pos, means, width=width, yerr=stds, label=m,
               color=METHOD_COLORS.get(m), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names)
    ax.set_ylabel("cumulative regret $G_T$ (mean ± std over seeds)")
    ax.set_title("Resource Contest — cumulative regret across configs")
    ax.legend(loc="upper left", fontsize=9)
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
        print(f"no results in {RESULTS_DIR}. Run `python -m experiments.rc.run` first.",
              file=sys.stderr)
        sys.exit(1)

    all_data = []
    for f in files:
        data = json.loads(f.read_text())
        all_data.append(data)
        name = data["config"]["name"]
        plot_cumulative_regret(data, FIGS_DIR / f"{name}_cum_regret.png")
        plot_reward_curve(data, FIGS_DIR / f"{name}_reward.png")
        plot_final_tau(data, FIGS_DIR / f"{name}_final_tau.png")

    if len(all_data) > 1:
        plot_summary_across_configs(all_data, FIGS_DIR / "summary.png")


if __name__ == "__main__":
    main()
