"""bci/motor_imagery acquisition: 2-class imagery-hold decoding.

Reuses the exp020 MI-v2 protocol (its outputs are the same simulation
runs, no recompute): cue 300-1200 ms = reafference on ONE leg side;
imagery window 1300-3500 ms = same-side LAL* drive (300 pA sustained
after PhotoCascade adaptation) that lights DN through REAL LAL->DN
synapses -- class = side.  This acquire step copies the calibrated
runs from experiments/exp020_proprioception/outputs/mi2 into this
directory's outputs; pass --refetch to re-run export_data instead.

Usage (conda ffbm, repo root):
    python bci/motor_imagery/acquire.py            # copy exp020 runs
    python bci/motor_imagery/acquire.py --refetch  # fresh acquisition
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = {"L": "mi2_L_short", "R": "mi2_R_short"}
REGIONS = "visual_bilateral,ol_rest,central_brain,vnc,proprioception"
T_END = 3600.0
SEED0 = 42
SRC = ROOT / "experiments" / "exp020_proprioception" / "outputs" / "mi2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true",
                    help="re-run export_data instead of copying exp020")
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, (cls, chem) in enumerate(CLASSES.items()):
        for r in range(args.repeats):
            stem = f"{cls}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            if not args.refetch:
                src = SRC / f"{cls}{r}" / "_debug_phi_scalp.npy"
                if src.exists():
                    shutil.copy(src, dst)
                    continue
                raise SystemExit(f"missing exp020 run {src} "
                                 "(use --refetch)")
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", chem, "--t-end", str(T_END),
                   "--seed", str(seed), "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"classes": list(CLASSES), "repeats": args.repeats,
         "fs": 1000.0, "dur_ms": T_END,
         "cue_ms": [300, 1200], "imagery_ms": [1300, 3500],
         "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
