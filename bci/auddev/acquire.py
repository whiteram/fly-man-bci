"""bci/auddev: auditory deviance detection acquisition.

One trial = a bilateral song train (JO-A* groups, 600 pA): four
500 ms pulse-song bursts at 25 Hz envelope with the THIRD burst as a
50 Hz deviant.  The deviance question: does the response to the
deviant burst exceed the adaptation-matched standard bursts?  6 runs,
different OU seeds; segment responses are cut in analysis.

Usage (conda ffbm, repo root):
    python bci/auddev/acquire.py               # 6 runs
    python bci/auddev/acquire.py --runs 2      # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,audition"
T_END = 3600.0
SEED0 = 42
BURSTS_MS = [(300, 800), (900, 1400), (1500, 2000), (2100, 2600),
             (2700, 3200)]
DEV_IDX = 2                       # third burst is the 50 Hz deviant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for r in range(args.runs):
        dst = out / f"train_r{r}.npy"
        if dst.exists():
            continue
        trial = out / f"_trial_r{r}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "dark",
               "--chem-input", "aud_dev",
               "--t-end", str(T_END), "--seed", str(SEED0 + r),
               "--out", str(trial), "--gpu"]
        print(f"[acquire] run {r} seed {SEED0 + r}: started", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed run {r}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"runs": args.runs, "fs": 1000.0, "dur_ms": T_END,
         "regions": REGIONS, "bursts_ms": BURSTS_MS,
         "deviant_index": DEV_IDX,
         "seed_map": "seed = 42 + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
