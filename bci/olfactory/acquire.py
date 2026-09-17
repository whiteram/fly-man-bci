"""bci/olfactory acquisition: 3-class odor-identity decoding.

One trial = one 900 ms pulse of a SINGLE odor (exp019-calibrated
minimal AL-recruitment protocol: ORN_DA1 = cVA, ORN_DM2 = vinegar
ester, ORN_VA1v = fatty acid; 150 pA @300 ms), classes = odor
identity, 6 repeats each (seed = 42 + i*repeats + r).  Per-trial clean
scalp EEG kept as outputs/<class>_r<r>.npy.

Usage (conda ffbm, repo root):
    python bci/olfactory/acquire.py               # 3 classes x 6 reps
    python bci/olfactory/acquire.py --repeats 1   # wiring smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = ["da1", "dm2", "va1v"]
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
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
            if (out / f"{stem}.npy").exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", f"odor_{cls}_short",
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", out / f"{stem}.npy")
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"classes": CLASSES, "repeats": args.repeats, "fs": 1000.0,
         "dur_ms": T_END, "drive_ms": [300, 1200],
         "regions": REGIONS,
         "odor_map": {"da1": "ORN_DA1 (cVA)", "dm2": "ORN_DM2 (ester)",
                      "va1v": "ORN_VA1v (fatty acid)"},
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
