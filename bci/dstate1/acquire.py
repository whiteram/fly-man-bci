"""dstate1: is the odor-DC switch the SAME state as the sz plateau?

Two independent discoveries point at one state:
  mmdev2: any effective odor pulse flips a sustained DC baseline
          (mean -1.389 -> -1.825 uV, std 0.100 -> 0.179; all-or-none,
          identity/amplitude blind, >= 8 s no recovery);
  sleep4-6: after seizure ignition the network sits in an actively
          maintained suppressed plateau (mean -1.825, std 0.18,
          MBON 3-5 Hz vs dark ~43 Hz baseline, resources recovered
          to 97.6% yet responses capped at 45%).
The target numbers agree to three decimals.  This study tests the
unification: a single DA1@150 pulse (chem da1_dc60, 60 s trial, 2
seeds) -- measure the DC step's persistence over 60 s and the
in-state population rate signature (ALPN/MBON/KC/DAN/Kenyon), to be
aligned against the sz_l1 plateau (MBON 3-5 Hz, ALPN ~38 Hz).

Usage (conda ffbm, repo root):
    python bci/dstate1/acquire.py
    python bci/dstate1/analyze.py
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
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
SEEDS = [62, 63]
T_END = 60000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        tag = f"da1_dc60_s{seed}"
        dst = out / f"{tag}_scalp.npy"
        if dst.exists():
            print(f"[acquire] {tag}: exists, skip", flush=True)
            continue
        trial = out / f"_trial_{tag}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", "da1_dc60", "--pop-rate", GROUPS,
               "--t-end", str(T_END), "--seed", str(seed),
               "--out", str(trial), "--gpu"]
        print(f"[acquire] {tag}", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {tag}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"chem": "da1_dc60", "seeds": seeds, "dur_ms": T_END,
         "groups": GROUPS, "regions": REGIONS,
         "pulse_ms": [300, 600], "fs": 1000.0,
         "sz_plateau_reference": {"mean_uv": -1.825, "std_uv": 0.18,
                                  "MBON_hz": [3, 5], "ALPN_hz": 38}},
        indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
