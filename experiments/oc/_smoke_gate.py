"""Fast smoke: one short cramped_room episode with gate ON, verify j_history populates."""
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from dataclasses import replace
from experiments.oc.configs import get_config
from scenarios.overcooked.driver import run_episode

cfg = get_config("cramped_room")
short = replace(cfg.config, max_steps=80)
ws = Path("workspace/oc_smoke/cramped").resolve(); ws.mkdir(parents=True, exist_ok=True)
r = run_episode(config=short, workspace=ws, reset_orch_log=True, reset_scratches=True, gate=True)
print("SMOKE RESULT:", {k: v for k, v in r.items() if k != "j_history"}, file=sys.stderr)
print("j_history entries:", len(r.get("j_history") or []), file=sys.stderr)
print("gate_stats:", r.get("gate_stats"), file=sys.stderr)
