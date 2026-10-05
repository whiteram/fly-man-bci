"""bci/condstate9: per-cell resolution -- DAN_err clamping profile and
micro-pool recruitment vs gain.

condstate8 closed the carrier chain but showed type-aggregated (K,V)
models cannot close the r account (lock: K +4.4%, V +10%, r -2.9 Hz) --
the residual lives in per-cell heterogeneity.  The existing
--pop-rate-neurons flag dumps per-neuron whole-trial spike counts,
which for DAN_err is dominated by the training window (baseline
0-0.33 Hz elsewhere).  Four arms (seed 62):

  usn9    cs6us   US-only naive     -- DAN_err fully driven (~110 Hz)
  usl9    cs6usl  US-only quiet lock -- who is clamped, and how?
  ctln9   cs2ctl  full training naive -- + micro-pool per-cell counts
  lockn9  cs2lock full training lock

Questions (analyze.py):
  1. Is the lock's DAN_err suppression a uniform down-scaling of all
     154 cells or a recruitment split (some clamped to baseline, some
     still driven)?  Distribution shapes answer it.
  2. Clamping depth vs per-cell GABA contact weight (from the exact
     349-row subset): the divisive kernel predicts cells with more
     GABA convergence clamp harder.
  3. MBON09/30/11/03/31 per-cell counts: is the lock's +25~78 Hz
     jump RECRUITMENT (silent cells firing) or GAIN (active cells
     firing more)?

Usage (conda ffbm, repo root):
    python bci/condstate9/acquire.py
    python bci/condstate9/analyze.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,MBON,DAN,Kenyon_Cell"
TYPES = ("MBON01,MBON02,MBON03,MBON04,MBON05,MBON06,MBON07,MBON09,"
         "MBON10,MBON11,MBON20,MBON25,MBON30,MBON31,MBON32,MBON34,"
         "MBON15-like,MBON25-like,DPM,APL,KCg-m")
ARMS = {
    "usn9":   {"chem": "cs6us",   "seed": 62, "t_end": 20000.0},
    "usl9":   {"chem": "cs6usl",  "seed": 62, "t_end": 20000.0},
    "ctln9":  {"chem": "cs2ctl",  "seed": 62, "t_end": 20000.0},
    "lockn9": {"chem": "cs2lock", "seed": 62, "t_end": 20000.0},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_neurons.npz"
    if dst.exists() and (out / f"{tag}_pop.npz").exists() \
            and (out / f"{tag}_dan.npy").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"],
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--pop-rate-neurons",
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(spec["t_end"]), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    if (trial / "_pop_rate_neurons.npz").exists():
        shutil.copy(trial / "_pop_rate_neurons.npz", dst)
    else:
        raise SystemExit(f"{tag}: no per-neuron dump")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta_p.write_text(json.dumps({"arms": ARMS}, indent=1))
    print("[condstate9] acquire done; run analyze.py")


if __name__ == "__main__":
    main()
