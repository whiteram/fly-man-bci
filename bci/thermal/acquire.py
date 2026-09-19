"""bci/thermal: temperature and humidity discrimination (new region).

The new `thermal` region (exp023: TH 25 TRN / HY 66 HRN, both with
~45k-syn CEN feedforward dipoles) read out at the scalp: 3-s slow
ramps (the natural envelope for tonic TRN/HRN firing).

  classes: warm (TRN_VP1m, hot pathway) vs cool (TRN_VP2+VP3*)
  blocks:  dry (HRN_VP4) vs moist (HRN_VP1*+VP5)
6 runs each class = 24 trials, 3.6 s per trial.  Decode: spatial
window-rms LOO over the ramp plateau (1500-3400 ms), permutation.

Usage (conda ffbm, repo root):
    python bci/thermal/acquire.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,thermal"
BLOCKS = {"temp": ["thermal_w", "thermal_c"],
          "hygro": ["hygro_d", "hygro_m"]}
T_END = 3600.0
SEED0 = 742


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    i = 0
    for classes in BLOCKS.values():
        for ci, chem in enumerate(classes):
            for r in range(args.runs):
                stem = f"{chem}_r{r}"
                dst = out / f"{stem}.npy"
                if dst.exists():
                    continue
                seed = SEED0 + i
                i += 1
                trial = out / f"_trial_{stem}"
                cmd = [sys.executable,
                       str(ROOT / "viz" / "export_data.py"),
                       "--regions", REGIONS, "--visual-input", "dark",
                       "--chem-input", chem,
                       "--t-end", str(T_END), "--seed", str(seed),
                       "--out", str(trial), "--gpu"]
                print("[acquire] " + " ".join(cmd[1:]), flush=True)
                r_ = subprocess.run(cmd, cwd=str(ROOT))
                if r_.returncode != 0:
                    raise SystemExit(f"export failed for {stem}")
                shutil.copy(trial / "_debug_phi_scalp.npy", dst)
                shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"blocks": {k: v for k, v in BLOCKS.items()},
         "runs": args.runs, "fs": 1000.0, "dur_ms": T_END,
         "regions": REGIONS, "window_ms": [1500, 3400],
         "note": "class 0 = first entry of each block (warm / dry)",
         "seed_map": "sequential from 742"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
