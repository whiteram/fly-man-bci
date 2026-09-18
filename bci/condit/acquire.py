"""bci/condit: driven olfactory conditioning (KC representation + reward).

The model's first LEARNING experiment.  Trial 8 s:
  500-2500 ms   training: KC-cluster drive 1600 pA (the odor's KC
                representation INJECTED -- the PN->KC pathway lives in
                CEN_C and the chem working point suppresses it to
                0.002, odor identity cannot reach KC; documented
                degradation) + bilateral PAM DAN drive 800 pA (reward
                = dopamine); --plastic-window open -> active KC->MBON
                synapses are DEPRESSED (fly three-factor rule)
  4200-4700 ms  probe: the SAME KC drive (500 ms) -- the readout is
                the pathway-level depression of the probe response
Conditions: learn (lr 3e-6, calibrated to ~50% depression) vs nolr
(lr 0) -- IDENTICAL drives and dynamics; the only difference is
whether synapses learn.

Usage (conda ffbm, repo root):
    python bci/condit/acquire.py               # 2 conditions x 4
    python bci/condit/acquire.py --runs 1      # calibration
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CONDS = {"learn": "3e-6", "nolr": "0.0"}
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 8000.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, (cond, lr) in enumerate(CONDS.items()):
        for r in range(args.runs):
            stem = f"{cond}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.runs + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", "condt2",
                   "--plastic-mb", "--plastic-window", "500,2500",
                   "--plastic-lr", lr,
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"conditions": list(CONDS), "lr": CONDS, "runs": args.runs,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "train_ms": [500, 2500], "testA_ms": [4200, 5200],
         "testB_ms": [6200, 7200],
         "seed_map": "seed = 42 + i*runs + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
