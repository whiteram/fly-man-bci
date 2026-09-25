"""bci/dstate8: does the quiescence-lock window survive AL population
subsampling?  (lock-window size sensitivity)

dstate2/3 mapped the lock window on the CEN_C gain axis at full
population (0.0016-0.0023 abs at the chem working point).  The open
robustness question: how does the window (and the lock itself) move
when the AL-local loop LOSES CELLS -- new surgery `--al-subsample F`
(a seeded, cache-deterministic subset of AL cells; only rows with
both ends in the subset remain).

Arms (seed 62, 60 s, single DA1@150 [300,600] unless laddered):
  lock formation at the neutral gain for F in {0.7, 0.5, 0.3}
  gain ladder {0.0015, 0.0025} abs at F=0.5 (window edges)

Refs (full population, prior arcs): lock at 150 pA / gain 0.002
(ALPN 20, std 0.0145); window edges 0.0016 / 0.0023.

Usage (conda ffbm, repo root):
    python bci/dstate8/acquire.py
    python bci/dstate8/analyze.py
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
GROUPS = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 60000.0
SEED = 62
CHEM = "da1_dc60"
# tag -> (subsample F, absolute ALL/CEN_C gain or None=auto 0.002)
ARMS = {
    "f070": (0.7, None),
    "f050": (0.5, None),
    "f030": (0.3, None),
    "f050_g0015": (0.5, 0.0015),
    "f050_g0025": (0.5, 0.0025),
    "f070_g004": (0.7, 0.004),
    "f050_g008": (0.5, 0.008),
}
WIN = (10000, 29000)


def run(tag, f, gain):
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
           "--al-local-gain", f"{gain if gain else 0.002:g}",
           "--al-subsample", f"{f:g}",
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if gain:
        # absolute gain = 0.002 * factor
        cmd += ["--gain-scale", f"CEN_C={gain / 0.002:g}"]
    print(f"[acquire] {tag}: F={f:g} gain={gain or 'auto(0.002)'}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    for a in args.arms.split(","):
        f, g = ARMS[a]
        run(a, f, g)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {a: {"F": ARMS[a][0], "gain": ARMS[a][1]}
                  for a in args.arms.split(",")},
         "groups": GROUPS, "dur_ms": T_END, "seed": SEED,
         "regions": REGIONS, "chem": CHEM,
         "window_ms": list(WIN),
         "refs_fullpop": {"lock": {"alpn": 19.8, "std": 0.0145},
                          "window_abs": [0.0016, 0.0023]}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
