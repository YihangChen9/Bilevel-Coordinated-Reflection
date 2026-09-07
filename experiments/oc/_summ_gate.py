import json, glob, os
from statistics import mean
print(f"{'layout':<22} {'oracle':>6} {'rand':>6} {'gate_del':>9} {'nogate_del':>11} {'jhist(g/ng)':>12} {'discarded(g/ng)':>16}")
for f in sorted(glob.glob('experiments/oc/results_gate/*.json')):
    d = json.load(open(f))
    name = d['config']
    orc = d.get('oracle') or {}
    orc_d = orc.get('deliveries') if isinstance(orc, dict) else None
    rnd = d.get('random') or []
    rnd_d = mean([r.get('deliveries', 0) for r in rnd]) if rnd else None
    def agg(arm):
        runs = d.get(arm) or []
        dels = [r.get('deliveries', 0) for r in runs]
        jh = sum(len(r.get('j_history') or []) for r in runs)
        disc = sum((r.get('gate_stats') or {}).get('writes_discarded', 0) for r in runs)
        seen = sum((r.get('gate_stats') or {}).get('writes_seen', 0) for r in runs)
        return (mean(dels) if dels else None), jh, disc, seen
    g_d, g_jh, g_disc, g_seen = agg('gate_on')
    ng_d, ng_jh, ng_disc, ng_seen = agg('gate_off')
    print(f"{name:<22} {str(orc_d):>6} {str(round(rnd_d,1) if rnd_d is not None else None):>6} "
          f"{str(round(g_d,2) if g_d is not None else None):>9} {str(round(ng_d,2) if ng_d is not None else None):>11} "
          f"{str(g_jh)+'/'+str(ng_jh):>12} {str(g_disc)+'/'+str(ng_disc):>16}")
