"""MRCP (motor readiness potential) acquisition: L/R classes.

Per trial 3.6 s (class = side):
  1300-3500 ms  LAL* staircase ramp on ONE side (300/600/900/1200 pA
                via stacked same-group channels; ~90-360 pA sustained
                after PhotoCascade adaptation) = the readiness buildup
  2800-3500 ms  same-side leg PRO reafference 400 pA = movement
                execution (the sensory copy)

Regions identical to exp020 MI (PRO->VNC reflex arcs included).

Usage (conda ffbm, repo root):
    python bci/mrcp/acquire.py               # 2 classes x 6
    python bci/mrcp/acquire.py --repeats 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = ["L", "R"]
REGIONS = "visual_bilateral,ol_rest,central_brain,vnc,proprioception"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, cls in enumerate(CLASSES):
        for r in range(args.repeats):
            stem = f"{cls}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", f"mrcp_{cls.lower()}",
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"classes": CLASSES, "repeats": args.repeats, "fs": 1000.0,
         "dur_ms": T_END, "regions": REGIONS,
         "ramp_ms": [1300, 3500], "movement_ms": [2800, 3500],
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
