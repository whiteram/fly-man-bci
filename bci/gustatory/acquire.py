"""Shared acquisition template for chem-class paradigms
(bci/gustatory: 3 organs; bci/touch: 2 sides).  One trial = one
export_data.py run with a single-pulse chem entry; per-trial clean
scalp EEG kept as outputs/<class>_r<r>.npy.

Usage (conda ffbm, repo root):
    python bci/<paradigm>/acquire.py               # full
    python bci/<paradigm>/<paradigm>.py --repeats 1  # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
T_END = 3600.0
SEED0 = 42

# per-paradigm configuration (this module is copied per directory and
# CONFIG is the only difference)
CONFIG = {
    "gustatory": {
        "classes": ["lglg", "wg", "claw"],
        "prefix": "taste_",
        "regions": "visual_bilateral,ol_rest,central_brain,gustatory,"
                   "touch",
    },
    "touch": {
        "classes": ["l", "r"],
        "prefix": "touch_",
        "regions": "visual_bilateral,ol_rest,central_brain,gustatory,"
                   "touch",
    },
}

if __name__ == "__main__":
    paradigm = HERE.name
    cfg = CONFIG[paradigm]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, cls in enumerate(cfg["classes"]):
        for r in range(args.repeats):
            stem = f"{cls}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", cfg["regions"], "--visual-input", "dark",
                   "--chem-input", cfg["prefix"] + cls,
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"classes": cfg["classes"], "repeats": args.repeats,
         "fs": 1000.0, "dur_ms": T_END, "regions": cfg["regions"],
         "drive_ms": [300, 1200],
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")
