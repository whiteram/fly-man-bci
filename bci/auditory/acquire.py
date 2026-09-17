"""bci/auditory acquisition: 4-class song decoding (side x song mode).

One trial = one export_data.py run: a 1.7 s courtship-song envelope on
ONE side's JO-A auditory population (pulse song = 25 Hz envelope
train, sine song = plateau), classes pulse_L / pulse_R / sine_L /
sine_R, 6 repeats each (seed = 42 + i*repeats + r).  Per-trial clean
scalp EEG harvested from _debug_phi_scalp.npy (x1.7 calibrated) and
kept as outputs/<class>_r<r>.npy; trial scratch dirs are removed.

Usage (conda ffbm, repo root):
    python bci/auditory/acquire.py               # 4 classes x 6 reps
    python bci/auditory/acquire.py --repeats 1   # wiring smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = ["pulse_L", "pulse_R", "sine_L", "sine_R"]
REGIONS = "visual_bilateral,ol_rest,central_brain,audition"
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
                   "--chem-input", f"aud_{cls}",
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
         "dur_ms": T_END, "drive_ms": [300, 2000],
         "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
