"""Drift calibration from OC gate logs (Assumptions: acceptance-law, decrement-law,
rate-form). V here is the GreedyHumanModel rollout surrogate (NOT the exact BFS table)
— report accordingly. RC 'many/drift' regret used as a secondary rate-form check."""
import json, glob
import numpy as np
import statsmodels.api as sm

trips = []   # (V_t, committed, dV)
traj = []    # per-run V sequence
for f in sorted(glob.glob('experiments/oc/results_gate/*.json')):
    d = json.load(open(f))
    for run in (d.get('gate_on') or []):
        seq = []
        for e in (run.get('j_history') or []):
            vb, va = e.get('j_before'), e.get('j_after')
            if not vb or vb <= 0:
                continue
            dv = (vb - va) if va is not None else 0
            trips.append((vb, 1 if e.get('committed') else 0, dv))
            seq.append(vb)
        if len(seq) >= 3:
            traj.append(seq)

V = np.array([t[0] for t in trips], float)
acc = np.array([t[1] for t in trips], float)
dV = np.array([t[2] for t in trips], float)
print(f"N gate events = {len(V)}  (accept {int(acc.sum())}, reject {int(len(V)-acc.sum())})")

# (1) Acceptance law: logit P(accept) = a + beta*log V  -> exponent beta on log V
X = sm.add_constant(np.log(V))
res = sm.Logit(acc, X).fit(disp=0)
beta_acc = res.params[1]; ci = res.conf_int()[1]
print(f"(1) acceptance-law  beta_hat(logV) = {beta_acc:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  p={res.pvalues[1]:.3f}")

# (2) Decrement law: among accepted, dV ~ c*V (through origin) vs constant
m = acc == 1
Va, dVa = V[m], dV[m]
if len(Va) >= 3:
    prop = sm.OLS(dVa, Va).fit()                      # dV = c*V (no intercept)
    const = sm.OLS(dVa, np.ones_like(Va)).fit()       # dV = c0
    daic = const.aic - prop.aic
    print(f"(2) decrement-law   E[dV|acc,V] = {prop.params[0]:.3f}*V   "
          f"(proportional vs constant  dAIC = {daic:.1f}; +ve favours proportional)")
else:
    print("(2) decrement-law   insufficient accepted events")

# (3) Rate-form: per trajectory, log V vs t (geometric) and log V vs log t (power)
geo_r2, pow_r2, geo_exp = [], [], []
for seq in traj:
    v = np.array(seq, float); t = np.arange(1, len(v) + 1)
    lv = np.log(v)
    # geometric: lv = a - b t
    bg = np.polyfit(t, lv, 1); pg = np.polyval(bg, t)
    r2g = 1 - np.sum((lv - pg) ** 2) / max(np.sum((lv - lv.mean()) ** 2), 1e-9)
    # power: lv = a - b log t
    bp = np.polyfit(np.log(t), lv, 1); pp = np.polyval(bp, np.log(t))
    r2p = 1 - np.sum((lv - pp) ** 2) / max(np.sum((lv - lv.mean()) ** 2), 1e-9)
    geo_r2.append(r2g); pow_r2.append(r2p); geo_exp.append(-bg[0])
geo_r2m, pow_r2m = np.mean(geo_r2), np.mean(pow_r2)
form = "geometric" if geo_r2m >= pow_r2m else "power-law"
beta_traj = np.mean([e for e in geo_exp])
print(f"(3) rate-form       geometric R2={geo_r2m:.3f}  power-law R2={pow_r2m:.3f}  -> {form} fits better")
print(f"    trajectory decay exponent (geometric, mean) = {beta_traj:.3f}")

# (4) Closed loop
print(f"(4) closed loop     beta_acceptance = {beta_acc:.3f}   vs   beta_trajectory = {beta_traj:.3f}")
