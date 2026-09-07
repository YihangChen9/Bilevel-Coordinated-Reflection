"""OC figures mirroring the RC pair:
  (1) gate vs no-gate delivery comparison (+ oracle / random reference),
  (2) J-trajectory convergence + fitted beta exponent on steps-to-delivery.
Reads experiments/oc/results_gate/<layout>.json (produced by run_gate.py)."""
import json
import glob
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "results_gate"
OUT = Path(__file__).parent / "figs_gate"
OUT.mkdir(exist_ok=True)

# layout order: 3 standard (greedy oracle works) then 3 hard (greedy oracle = 0)
ORDER = ["cramped_room", "asymmetric_advantages", "centre_pots",
         "counter_circuit", "coordination_ring", "forced_coordination"]


def load(name):
    p = ROOT / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def mean_deliveries(runs):
    ds = [r.get("deliveries", 0) for r in (runs or [])]
    return mean(ds) if ds else 0.0


def fig_gate():
    data = {n: load(n) for n in ORDER if load(n)}
    names = list(data.keys())
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    oracle = [(data[n].get("oracle") or {}).get("deliveries", 0) for n in names]
    rand = [mean_deliveries(data[n].get("random")) for n in names]
    gate = [mean_deliveries(data[n].get("gate_on")) for n in names]
    nogate = [mean_deliveries(data[n].get("gate_off")) for n in names]
    ax.bar(x - 1.5*w, oracle, w, label="greedy oracle", color="#9aa0a6")
    ax.bar(x - 0.5*w, rand, w, label="random", color="#c7cdd4")
    ax.bar(x + 0.5*w, nogate, w, label="bilevel, no gate", color="#e8845b")
    ax.bar(x + 1.5*w, gate, w, label="bilevel, gated (J-ascent)", color="#4c8bf5")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.axvline(2.5, ls="--", color="k", alpha=0.3)
    ax.text(1.0, ax.get_ylim()[1]*0.95, "greedy-oracle solvable", fontsize=8, alpha=0.6)
    ax.text(3.6, ax.get_ylim()[1]*0.95, "hard (oracle=0)", fontsize=8, alpha=0.6)
    ax.set_ylabel("mean deliveries / episode")
    ax.set_title("OC: gated vs non-gated write (Qwen3-235B, 3 seeds)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "oc_gate_vs_nogate.png", dpi=140)
    print("wrote", OUT / "oc_gate_vs_nogate.png")
    print("\n[OC gate comparison mean deliveries]")
    for i, n in enumerate(names):
        print(f"  {n:<22} oracle={oracle[i]} random={rand[i]:.2f} "
              f"nogate={nogate[i]:.2f} gate={gate[i]:.2f}")


def j_series(runs):
    """Concatenate j_before trajectories ordered by step, per run -> list of arrays."""
    out = []
    for r in (runs or []):
        jh = sorted(r.get("j_history") or [], key=lambda e: e["step"])
        js = [e["j_before"] for e in jh if e.get("j_before") and e["j_before"] > 0]
        if len(js) >= 3:
            out.append(np.array(js, dtype=float))
    return out


def fig_beta():
    data = {n: load(n) for n in ORDER if load(n)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    betas = {}
    for n, d in data.items():
        series = j_series(d.get("gate_on"))
        if not series:
            continue
        # fit each run's J(turn) ~ turn^-beta, average beta; plot the longest run
        bs = []
        for js in series:
            t = np.arange(1, len(js) + 1)
            # J = steps-to-delivery (lower=better); shrinking J => positive beta
            b = -np.polyfit(np.log(t), np.log(js), 1)[0]
            bs.append(b)
        betas[n] = mean(bs)
        longest = max(series, key=len)
        axes[0].plot(np.arange(1, len(longest)+1), longest, marker="o", ms=3,
                     label=f"{n} (β={betas[n]:.2f})")
    axes[0].set_xlabel("gated-write turn"); axes[0].set_ylabel("J = steps-to-delivery (lower=closer)")
    axes[0].set_title("OC: J trajectory under gated writes (gate ON)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    if betas:
        ns = list(betas.keys()); x = np.arange(len(ns))
        axes[1].bar(x, [betas[n] for n in ns], color="#4c8bf5")
        axes[1].set_xticks(x); axes[1].set_xticklabels(ns, rotation=20, ha="right", fontsize=7)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_ylabel("fitted β (J ~ turn^-β; >0 = converging)")
        axes[1].set_title("OC: convergence exponent β by layout")
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "oc_beta_regression.png", dpi=140)
    print("wrote", OUT / "oc_beta_regression.png")
    print("\n[OC beta fits, J~turn^-beta (gate ON)]")
    for n, b in betas.items():
        print(f"  {n:<22} beta={b:.3f}")


if __name__ == "__main__":
    fig_gate()
    fig_beta()
