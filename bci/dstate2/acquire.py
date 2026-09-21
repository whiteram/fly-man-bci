"""dstate2: what sustains the ALPN plateau inside the quiescence lock?

dstate1 + the ORN probe: a single DA1@150 pulse pushes the network
into a deep-quiescence lock (mean -1.821, std 0.0145, all monitored
central populations silent) in which ALPN is the ONLY active
population, self-sustaining at ~20 Hz (dark: 0 Hz).  The ORN is
silent, so the plateau is not held by sensory drive.  Candidates:
(a) CEN_C-mediated synaptic loops (in the lock the only participants
left standing are the AL-local rows -- PN<->LN/PN within the AL --
since every other central population is silent), or (b) ALPN
intrinsic dynamics.

Lever: the static CEN_C gain scale (the lock forms via ORN_C, a
separate group, so formation is preserved at any CEN_C gain).
Ladder arms (60 s trials, chem da1_dc60, seed 62):
  g025  CEN_C x0.025 (5e-5 -- recurrence effectively off)
  g050  CEN_C x0.5   (0.001)
  g150  CEN_C x1.5   (0.003)
dstate1's s62 run is the x1.0 point.  Readout: ALPN plateau rate
(10-60 s) vs gain -- monotone dependence = synapse-driven (then the
AL-local split isolates the loop); gain-insensitive = intrinsic.

Usage (conda ffbm, repo root):
    python bci/dstate2/acquire.py
    python bci/dstate2/analyze.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ORN,MBON,ALPN,Kenyon_Cell,DAN,ALLN,ALIN,ALON"
SEED = 62
T_END = 60000.0
ARMS = {"g025": 0.025, "g050": 0.5, "g150": 1.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    arms = args.arms.split(",")
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        tag = f"da1_dc60_{arm}_s{SEED}"
        dst = out / f"{tag}_scalp.npy"
        if dst.exists():
            print(f"[acquire] {tag}: exists, skip", flush=True)
            continue
        trial = out / f"_trial_{tag}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", "da1_dc60", "--pop-rate", GROUPS,
               "--t-end", str(T_END), "--seed", str(SEED),
               "--gain-scale", f"CEN_C={ARMS[arm]:g}",
               "--out", str(trial), "--gpu"]
        print(f"[acquire] {tag}: CEN_C x{ARMS[arm]:g}", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {tag}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "seed": SEED, "dur_ms": T_END, "chem": "da1_dc60",
         "groups": GROUPS, "regions": REGIONS,
         "reference_x1": "dstate1 da1_dc60_s62 (ALPN plateau ~19.9 Hz)",
         "working_point": "chem CEN_C gain 0.002 x scale"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
