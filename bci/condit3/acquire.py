"""bci/condit3: TRUE odor conditioning (AL-independence pathway).

With --al-gain 0.01 the AL->KC (PN->Kenyon) rows escape the chem
working point's 500x damping: a REAL odor (ORN entry -> antennal lobe
-> PN -> KC) drives Kenyon cells through the connectome -- no injected
KC representation.  Trial 5 s:

  500-2500 ms  training: odor DA1 via ORN (150 pA) + bilateral PAM
              reward (800 pA); --plastic-window open -> the KC subset
              ACTIVATED BY THE ODOR depresses its KC->MBON synapses
  4200-4700 ms  probe: condt3a probes the TRAINED odor (DA1), condt3b
              an UNTRAINED one (DM2, group-size-compensated amp) --
              the identity-specificity test driven conditioning could
              not do.

Conditions: learn (lr 5e-5) vs nolr (0.0) x probe {a, b}, identical
drives otherwise.  5 runs each = 20 trials.

Usage (conda ffbm, repo root):
    python bci/condit3/acquire.py            # 20 trials (~25 min)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
AL_GAIN = "0.01"
LR = {"learn": "5e-5", "nolr": "0.0"}
PROBES = ("condt3a", "condt3b")
T_END = 5000.0
SEED0 = 242


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    i = 0
    for lr_name, lr in LR.items():
        for probe in PROBES:
            for r in range(args.runs):
                stem = f"{lr_name}_{probe[-1]}_r{r}"
                dst = out / f"{stem}.npy"
                if dst.exists():
                    continue
                seed = SEED0 + i
                i += 1
                trial = out / f"_trial_{stem}"
                cmd = [sys.executable,
                       str(ROOT / "viz" / "export_data.py"),
                       "--regions", REGIONS, "--visual-input", "dark",
                       "--chem-input", probe,
                       "--plastic-mb", "--plastic-window", "500,2500",
                       "--plastic-lr", lr,
                       "--al-gain", AL_GAIN,
                       "--t-end", str(T_END), "--seed", str(seed),
                       "--out", str(trial), "--gpu"]
                print("[acquire] " + " ".join(cmd[1:]), flush=True)
                r_ = subprocess.run(cmd, cwd=str(ROOT))
                if r_.returncode != 0:
                    raise SystemExit(f"export failed for {stem}")
                shutil.copy(trial / "_debug_phi_scalp.npy", dst)
                shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"conditions": ["learn", "nolr"], "probes": list(PROBES),
         "lr": LR, "al_gain": float(AL_GAIN), "runs": args.runs,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "train_ms": [500, 2500], "probe_ms": [4300, 4700],
         "probe_odor": {"a": "DA1 (trained)", "b": "DM2 (untrained)"},
         "seed_map": "sequential from 242"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
