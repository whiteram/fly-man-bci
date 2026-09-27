"""bci/dstate16: the gain-axis transition map -- can the lock and the
latch family convert into each other?

dstate2/3 mapped formation from NAIVE (window 0.0016-0.0023 ->
latch above); dstate14 mapped dissolution from the LOCK (drops below
the window dissolve it).  The untested transitions: WITHIN an
established state, moving the gain ACROSS regions --

  from lock (formed at 0.002, dose 150):      up to 0.003 / 0.004
  from latch (formed at 0.004 full drive):    down to 0.002 / 0.001
  (from latch, does 0.004's latch survive a drop to 0.002 -- the
   lock's own window -- i.e. does it LAND in the lock or dissolve?)

Design (sz_l1 full-drive latch needs gain x2 = 0.004; the lock is
formed at the working point):
  lock_up_3   lock @30s -> CEN_C x1.5 (0.003)
  lock_up_4   lock @30s -> CEN_C x2.0 (0.004)
  latch_down2 latch @30s -> CEN_C x0.5 (0.002 = lock window center)
  latch_down1 latch @30s -> CEN_C x0.25 (0.001)
60 s trials, sz_l1, seed 62.

Usage (conda ffbm, repo root):
    python bci/dstate16/acquire.py
    python bci/dstate16/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 60000.0
SEED = 62
CHEM = "sz_l1"
# tag -> (chem entry, schedule, expected pre-state).  Lock arms form
# the lock with the odor pulse (da1_dc60) at the working point; latch
# arms run sz_l1 at x2.0 (0.004 full latch, dstate9 g20).  The gain
# schedule then moves ACROSS regions at t=30 s.
ARMS = {
    "lock_up3": ("da1_dc60", "30000:CEN_C=1.5", "lock"),
    "lock_up4": ("da1_dc60", "30000:CEN_C=2.0", "lock"),
    "latch_dn2": ("sz_l1", "30000:CEN_C=0.5", "latch"),
    "latch_dn1": ("sz_l1", "30000:CEN_C=0.25", "latch"),
}
PRE, POST = (10000, 29000), (40000, 55000)


def run(tag, spec):
    chem_l, sched, _pre = spec
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", CHEM, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if sched:
        cmd += ["--gain-schedule", sched]
    if chem_l != CHEM:
        cmd[cmd.index(CHEM)] = chem_l
    print(f"[acquire] {tag}: {chem_l} {sched}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: {"chem": v[0], "sched": v[1], "pre_expect": v[2]}
                  for k, v in ARMS.items()},
         "groups": GROUPS, "dur_ms": T_END, "seed": SEED,
         "regions": REGIONS, "chem": CHEM, "drop_ms": 30000,
         "windows_ms": {"pre": list(PRE), "post": list(POST)},
         "note": "lock arms: odor pulse (da1_dc60) forms the lock at "
                 "the working point; latch arms: sz_l1 at x2.0 (0.004, "
                 "dstate9 g20 full latch).  Gain schedule moves ACROSS "
                 "regions at t=30 s."}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
