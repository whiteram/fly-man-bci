"""bci/dstate14: does the ESTABLISHED quiet lock survive a gain drop
below its formation window?  (hysteresis / the parameter-route cure)

dstate5 found no dynamic exit (probe ladder absorbed); dstate2/7/8
mapped the formation edges (gain window 0.0016-0.0023, dose edge
100-150 pA, duration edge 100-300 ms) -- always from NAIVE.  The open
question: form the lock (30 s at the working point), then DROP CEN_C
gain mid-run below the window.  Dissolves to dark = a cure exists;
persists = the lock is hysteretic (its own basin, formation edge is
not the exit edge).

New runtime flag `--gain-schedule t_ms:GRP=f` (step-hook scalar;
GPU only).

Arms (da1_dc60, 60 s, seed 62; drop at t=30 s):
  hold    no schedule (control -- lock persists)
  d0015   CEN_C x0.75 -> 0.0015 (just below the window)
  d001    CEN_C x0.50 -> 0.0010
  d0005   CEN_C x0.25 -> 0.0005 (deep below)

Usage (conda ffbm, repo root):
    python bci/dstate14/acquire.py
    python bci/dstate14/analyze.py
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
CHEM = "da1_dc60"
ARMS = {"hold": None, "d0015": "30000:CEN_C=0.75",
        "d001": "30000:CEN_C=0.5", "d0005": "30000:CEN_C=0.25"}
PRE, POST = (10000, 29000), (35000, 55000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145)}


def classify(alpn, std):
    return min(REFS, key=lambda k: (abs(alpn - REFS[k][0]) / 20.0
                                    + abs(std - REFS[k][1]) / 0.05))


def run(tag, sched):
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
    print(f"[acquire] {tag}: sched={sched}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, sched in ARMS.items():
        run(tag, sched)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "groups": GROUPS, "dur_ms": T_END,
         "seed": SEED, "regions": REGIONS, "chem": CHEM,
         "drop_ms": 30000,
         "windows_ms": {"pre": list(PRE), "post": list(POST)},
         "refs": {k: {"alpn": v[0], "std": v[1]}
                  for k, v in REFS.items()}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
