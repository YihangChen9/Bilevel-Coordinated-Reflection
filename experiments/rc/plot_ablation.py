"""Plot RC ablation results.

Produces, per model:
  figs_ablation/<model>/regret_bar_<config>.png       — bar chart of final regret per variant
  figs_ablation/<model>/cum_regret_<config>.png       — cumulative regret per variant over time
  figs_ablation/<model>/tau_traj_<variant>_<config>.png — tau over time per agent
  figs_ablation/<model>/x_eff_<variant>_<config>.png    — x_effective over time per agent
  figs_ablation/<model>/gated_in_pct.png              — % rounds gated_in per variant × config
  figs_ablation/<model>/heatmap_regret.png            — variant × config regret matrix
  figs_ablation/<model>/RESULTS_ABLATION.md           — text summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_root: Path, model_slug: str) -> dict[tuple[str, str], dict]:
    """Returns {(variant, config): episode_dict}."""
    out: dict[tuple[str, str], dict] = {}
    model_dir = results_root / model_slug
    if not model_dir.exists():
        return out
    for variant_dir in sorted(model_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        for f in sorted(variant_dir.glob("*.json")):
            d = json.loads(f.read_text())
            out[(variant_dir.name, f.stem)] = d
    return out


def plot_regret_bar(data: dict, model_slug: str, out_dir: Path) -> None:
    """Per config: bar chart of final regret across variants."""
    configs = sorted({c for (_, c) in data.keys()})
    for cfg in configs:
        rows = sorted(((v, d['summary']['final_regret'])
                       for (v, c), d in data.items() if c == cfg),
                      key=lambda x: x[1])
        if not rows:
            continue
        variants = [r[0] for r in rows]
        regrets = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(variants, regrets, color="steelblue")
        # Highlight default
        for i, v in enumerate(variants):
            if v == "default":
                bars[i].set_color("orange")
        ax.set_xlabel("Final regret (lower is better)")
        ax.set_title(f"{model_slug} — {cfg}")
        for i, r in enumerate(regrets):
            ax.text(r, i, f" {r:.2f}", va="center", fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(out_dir / f"regret_bar_{cfg}.png", dpi=120)
        plt.close()


def plot_cum_regret(data: dict, model_slug: str, out_dir: Path) -> None:
    """Per config: cumulative regret curves, one line per variant."""
    configs = sorted({c for (_, c) in data.keys()})
    for cfg in configs:
        runs = [(v, d) for (v, c), d in data.items() if c == cfg]
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for v, d in sorted(runs):
            rows = d.get("rows", [])
            ts = [r["t"] for r in rows]
            cum_reg = np.cumsum([r["regret"] for r in rows])
            style = "-" if v == "default" else "--"
            lw = 2.5 if v == "default" else 1.5
            ax.plot(ts, cum_reg, style, lw=lw, label=v, alpha=0.85)
        ax.set_xlabel("round t")
        ax.set_ylabel("cumulative regret")
        ax.set_title(f"{model_slug} — {cfg} (cum regret over T)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"cum_regret_{cfg}.png", dpi=120)
        plt.close()


def plot_tau_traj(data: dict, model_slug: str, out_dir: Path) -> None:
    """For each (variant, config), plot tau over t (one line per agent)."""
    for (v, c), d in data.items():
        rows = d.get("rows", [])
        if not rows:
            continue
        tau_arr = np.array([r["tau"] for r in rows])  # T × N
        ts = [r["t"] for r in rows]
        n = tau_arr.shape[1] if tau_arr.size else 0
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for i in range(n):
            ax.plot(ts, tau_arr[:, i], marker="o", ms=4, label=f"agent_{i}")
        ax.set_xlabel("round t")
        ax.set_ylabel("τ (budget share)")
        ax.set_ylim(-0.05, 1.05)
        # Mark drift flip
        if d.get("config", {}).get("cap_schedule"):
            for t_flip, _ in d["config"]["cap_schedule"]:
                ax.axvline(t_flip, color="red", linestyle=":", alpha=0.6, label="flip" if t_flip == d["config"]["cap_schedule"][0][0] else None)
        ax.set_title(f"{model_slug} — {v} / {c}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"tau_traj_{v}_{c}.png", dpi=120)
        plt.close()


def plot_x_eff(data: dict, model_slug: str, out_dir: Path) -> None:
    """For each (variant, config), plot x_effective over t (one line per agent)."""
    for (v, c), d in data.items():
        rows = d.get("rows", [])
        if not rows:
            continue
        x_arr = np.array([r["x_effective"] for r in rows])
        ts = [r["t"] for r in rows]
        n = x_arr.shape[1] if x_arr.size else 0
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for i in range(n):
            ax.plot(ts, x_arr[:, i], marker="s", ms=4, label=f"agent_{i}")
        ax.set_xlabel("round t")
        ax.set_ylabel("x_effective (env clipped)")
        if d.get("config", {}).get("cap_schedule"):
            for t_flip, _ in d["config"]["cap_schedule"]:
                ax.axvline(t_flip, color="red", linestyle=":", alpha=0.6)
        ax.set_title(f"{model_slug} — {v} / {c} (x_eff)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"x_eff_{v}_{c}.png", dpi=120)
        plt.close()


def plot_gated_pct(data: dict, model_slug: str, out_dir: Path) -> None:
    """Stacked bar: per (variant, config) percentage of rounds gated_in to orch_log."""
    keys = sorted(data.keys())
    if not keys:
        return
    variants = sorted({v for v, _ in keys})
    configs = sorted({c for _, c in keys})
    mat = np.full((len(variants), len(configs)), np.nan)
    for (v, c), d in data.items():
        T = d["summary"]["n_rounds_recorded"]
        if T == 0:
            continue
        pct = 100.0 * d["summary"]["n_rounds_gated_in"] / T
        mat[variants.index(v), configs.index(c)] = pct
    fig, ax = plt.subplots(figsize=(2 + len(configs) * 1.3, 0.5 + len(variants) * 0.45))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    for i in range(len(variants)):
        for j in range(len(configs)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        color="white" if v < 60 else "black", fontsize=8)
    plt.colorbar(im, label="% rounds gated_in to orch_log")
    plt.title(f"{model_slug} — orch_log write rate")
    plt.tight_layout()
    plt.savefig(out_dir / "gated_in_pct.png", dpi=120)
    plt.close()


def plot_regret_heatmap(data: dict, model_slug: str, out_dir: Path) -> None:
    """variant × config heatmap of final regret."""
    keys = sorted(data.keys())
    if not keys:
        return
    variants = sorted({v for v, _ in keys})
    configs = sorted({c for _, c in keys})
    mat = np.full((len(variants), len(configs)), np.nan)
    for (v, c), d in data.items():
        mat[variants.index(v), configs.index(c)] = d["summary"]["final_regret"]
    fig, ax = plt.subplots(figsize=(2 + len(configs) * 1.3, 0.5 + len(variants) * 0.45))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    for i in range(len(variants)):
        for j in range(len(configs)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, label="final regret (lower = better)")
    plt.title(f"{model_slug} — regret heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_regret.png", dpi=120)
    plt.close()


def write_summary_md(data: dict, model_slug: str, out_dir: Path) -> None:
    keys = sorted(data.keys())
    variants = sorted({v for v, _ in keys})
    configs = sorted({c for _, c in keys})
    lines: list[str] = []
    lines.append(f"# RC Ablation Results — `{model_slug}`\n")
    lines.append(f"{len(data)} episodes total, 1 seed (=42) each.\n\n")
    lines.append("## Final Regret Matrix\n")
    lines.append(f"| variant | " + " | ".join(configs) + " |")
    lines.append("|" + "---|" * (len(configs) + 1))
    for v in variants:
        cells = []
        for c in configs:
            d = data.get((v, c))
            if d is None:
                cells.append("—")
            else:
                cells.append(f"{d['summary']['final_regret']:.2f}")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Total Reward Matrix (out of optimal)\n")
    lines.append(f"| variant | " + " | ".join(configs) + " |")
    lines.append("|" + "---|" * (len(configs) + 1))
    for v in variants:
        cells = []
        for c in configs:
            d = data.get((v, c))
            if d is None:
                cells.append("—")
            else:
                cells.append(f"{d['summary']['total_reward']:.1f} / {d['summary']['optimal_total']:.0f}")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## orch_log Write Rate (gated_in / T)\n")
    lines.append(f"| variant | " + " | ".join(configs) + " |")
    lines.append("|" + "---|" * (len(configs) + 1))
    for v in variants:
        cells = []
        for c in configs:
            d = data.get((v, c))
            if d is None:
                cells.append("—")
            else:
                T = d['summary']['n_rounds_recorded']
                g = d['summary']['n_rounds_gated_in']
                pct = 100.0 * g / T if T else 0.0
                cells.append(f"{g}/{T} ({pct:.0f}%)")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Wall-clock per episode\n")
    lines.append(f"| variant | " + " | ".join(configs) + " |")
    lines.append("|" + "---|" * (len(configs) + 1))
    for v in variants:
        cells = []
        for c in configs:
            d = data.get((v, c))
            if d is None:
                cells.append("—")
            else:
                cells.append(f"{d['elapsed_sec']:.0f}s")
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")
    (out_dir / "RESULTS_ABLATION.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "ablation_results")
    ap.add_argument("--figs-dir", type=Path,
                    default=Path(__file__).parent / "figs_ablation")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Model slugs to process (default: all present)")
    args = ap.parse_args()

    available = [p.name for p in args.results_dir.iterdir() if p.is_dir()]
    targets = args.models if args.models is not None else available
    for slug in targets:
        data = load_results(args.results_dir, slug)
        if not data:
            print(f"[skip] no data for {slug}")
            continue
        out = args.figs_dir / slug
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{slug}] {len(data)} episodes → {out}")
        plot_regret_bar(data, slug, out)
        plot_cum_regret(data, slug, out)
        plot_tau_traj(data, slug, out)
        plot_x_eff(data, slug, out)
        plot_gated_pct(data, slug, out)
        plot_regret_heatmap(data, slug, out)
        write_summary_md(data, slug, out)
        print(f"  → wrote regret_bar/cum_regret/tau_traj/x_eff/gated_in/heatmap/RESULTS_ABLATION.md")


if __name__ == "__main__":
    main()
