"""RC figures: (1) gated vs non-gated write comparison, (2) beta convergence-exponent
regression on the per-t regret curves. Reads experiments/rc/ablation_results/<model>/<variant>/<config>.json."""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "ablation_results"
OUT = Path(__file__).parent / "figs_ablation"
OUT.mkdir(exist_ok=True)
CONFIGS = ["easy", "hard", "many", "drift"]

def load(model, variant, cfg):
    p = ROOT / model / variant / f"{cfg}.json"
    return json.load(open(p)) if p.exists() else None

def total_reward(model, variant, cfg):
    d = load(model, variant, cfg)
    if not d: return None
    return d["summary"].get("total_reward") if isinstance(d.get("summary"), dict) else None

# ---- Figure 1: gated vs non-gated (MiniMax has the workboard-gate variants) ----
def fig_gate():
    model = "MiniMax-M2.7"
    # non-gated baseline vs the two workboard-gated variants
    series = {
        "no-gate (default)": "default",
        "gated workboard": "gated_workboard",
        "gate + strat-read (algorithm_complete)": "algorithm_complete",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CONFIGS)); w = 0.26
    for i, (label, var) in enumerate(series.items()):
        vals = [total_reward(model, var, c) or np.nan for c in CONFIGS]
        ax.bar(x + (i-1)*w, vals, w, label=label)
    # oracle reference per config
    orac = [total_reward(model, "default", c) for c in CONFIGS]  # placeholder if no oracle stored
    ax.set_xticks(x); ax.set_xticklabels(CONFIGS)
    ax.set_ylabel("Σ reward (higher better)"); ax.set_title("RC: gated vs non-gated write (MiniMax-M2.7)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "gate_vs_nogate.png", dpi=140)
    print("wrote", OUT / "gate_vs_nogate.png")
    # print the numbers too
    print("\n[gate comparison Σ-reward]")
    for c in CONFIGS:
        row = " | ".join(f"{lbl}={total_reward(model,var,c)}" for lbl,var in series.items())
        print(f"  {c}: {row}")

# ---- Figure 2: beta convergence-exponent regression ----
def cum_regret(model, variant, cfg):
    d = load(model, variant, cfg)
    if not d or not d.get("rows"): return None
    reg = np.array([r["regret"] for r in d["rows"]])
    return np.cumsum(reg)

def inst_regret(model, variant, cfg):
    d = load(model, variant, cfg)
    if not d or not d.get("rows"): return None
    return np.array([max(r["regret"], 1e-6) for r in d["rows"]])

def fig_beta():
    model = "MiniMax-M2.7"
    variant = "algorithm_complete"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    betas = {}
    for cfg in CONFIGS:
        r = inst_regret(model, variant, cfg)
        if r is None: continue
        t = np.arange(1, len(r)+1)
        # power-law fit: log r = log A - beta*log t  (regret ~ t^-beta)
        lr, lt = np.log(r), np.log(t)
        beta_pow = -np.polyfit(lt, lr, 1)[0]
        # exponential fit: log r = log A - beta*t   (regret ~ e^-beta t)
        beta_exp = -np.polyfit(t, lr, 1)[0]
        betas[cfg] = (beta_pow, beta_exp)
        axes[0].plot(t, r, marker="o", ms=3, label=f"{cfg} (β_pow={beta_pow:.2f})")
    axes[0].set_yscale("log"); axes[0].set_xlabel("round t"); axes[0].set_ylabel("instantaneous regret (log)")
    axes[0].set_title(f"RC regret decay + power-law β ({variant}, {model})")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # bar of fitted betas
    cfgs = list(betas.keys()); x = np.arange(len(cfgs)); w = 0.38
    axes[1].bar(x-w/2, [betas[c][0] for c in cfgs], w, label="power-law β (r~t^-β)")
    axes[1].bar(x+w/2, [betas[c][1] for c in cfgs], w, label="exponential β (r~e^-βt)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cfgs); axes[1].set_ylabel("fitted β (convergence exponent)")
    axes[1].set_title("RC: convergence exponent β by config"); axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "beta_regression.png", dpi=140)
    print("wrote", OUT / "beta_regression.png")
    print("\n[beta fits]  config: power-law β / exponential β")
    for c, (bp, be) in betas.items():
        print(f"  {c}: {bp:.3f} / {be:.3f}")

if __name__ == "__main__":
    fig_gate()
    fig_beta()
