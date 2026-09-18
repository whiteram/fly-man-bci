"""bci/odormix: mixture nonlinearity + stimulus-order decoding.

Five conditions (single 900 ms or two 400 ms pulses, equalized
amplitudes DA1 150 / DM2 600 pA):
  a      DA1 alone        (reuses odor_da1_short)
  b      DM2 alone        (reuses odor_dm2_short)
  ab     DA1+DM2 together [300,1200]
  abseq  A then B         [300,700] / [800,1200]
  baseq  B then A         (order reversed)

Analyses:
  1. mixture nonlinearity: rms(ab) vs baseline + (rms(a)-base) +
     (rms(b)-base) -- linear summation would land on the prediction.
  2. 3-class {a, b, ab} spatial-rms LOO.
  3. order decoding: abseq vs baseq, temporal-template 2-class
     (the identity of the FIRST pulse lives in the early response).

Usage (conda ffbm, repo root):
    python bci/odormix/acquire.py               # 5 conditions x 5
    python bci/odormix/acquire.py --repeats 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CONDS = ["a", "b", "ab", "abseq", "baseq"]
CHEM = {"a": "odor_da1_short", "b": "odor_dm2_short",
        "ab": "odormix_ab", "abseq": "odormix_abseq",
        "baseq": "odormix_baseq"}
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, cond in enumerate(CONDS):
        for r in range(args.repeats):
            stem = f"{cond}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", CHEM[cond],
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"conditions": CONDS, "repeats": args.repeats, "fs": 1000.0,
         "dur_ms": T_END, "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
