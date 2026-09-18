"""bci/seizure acquisition: focal ictal onset + lateralization.

The exp019 calibration discovered that at the DEFAULT central
recurrence gain any suprathreshold chem input latches the whole CEN
into a self-sustained ~82 Hz state -- an ictal (seizure-like)
attractor.  This paradigm uses that property: a focal drive on ONE
side's LAL group at default gain ("no_cen_damp") seeds a lateralized
seizure; the same drive at the chem working point (sz_ctl, damped)
stays healthy.  Classes: sz_L / sz_R / ctl, 4 repeats each.

Usage (conda ffbm, repo root):
    python bci/seizure/acquire.py               # 3 classes x 4
    python bci/seizure/acquire.py --repeats 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = ["sz_l", "sz_r", "ctl"]
REGIONS = "visual_bilateral,ol_rest,central_brain"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
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
                   "--chem-input", cls if cls != "ctl" else "sz_ctl",
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
         "dur_ms": T_END, "drive_ms": [800, 2800],
         "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
