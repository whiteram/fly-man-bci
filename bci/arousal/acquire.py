"""bci/arousal acquisition: synthetic state manipulation.

--noise-scale scales every OU sigma: quiet state = 0.3x, active
state = 1.0x (documented as a SYNTHETIC arousal proxy, not a modeled
sleep/attention circuit).  Conditions {quiet, active} x {dark,
near-threshold odor (dose_075 = 75 pA DA1 pulse)} x 4 runs:
  - state decoding from spontaneous dark EEG
  - odor detectability per state (the exp018 premise made dynamic:
    signal-to-floor improves in the quiet state)

Usage (conda ffbm, repo root):
    python bci/arousal/acquire.py            # 2x2 x 4
    python bci/arousal/acquire.py --runs 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

STATES = {"quiet": "0.3", "active": "1.0"}
STIM = {"dark": None, "odor": "dose_075"}
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, (state, ns) in enumerate(STATES.items()):
        for j, (stim, chem) in enumerate(STIM.items()):
            for r in range(args.runs):
                stem = f"{state}_{stim}_r{r}"
                dst = out / f"{stem}.npy"
                if dst.exists():
                    continue
                seed = SEED0 + (i * len(STIM) + j) * args.runs + r
                trial = out / f"_trial_{stem}"
                cmd = [sys.executable,
                       str(ROOT / "viz" / "export_data.py"),
                       "--regions", REGIONS, "--visual-input", "dark",
                       "--noise-scale", ns]
                if chem:
                    cmd += ["--chem-input", chem]
                cmd += ["--t-end", str(T_END), "--seed", str(seed),
                        "--out", str(trial), "--gpu"]
                print("[acquire] " + " ".join(cmd[1:]), flush=True)
                r_ = subprocess.run(cmd, cwd=str(ROOT))
                if r_.returncode != 0:
                    raise SystemExit(f"export failed for {stem}")
                shutil.copy(trial / "_debug_phi_scalp.npy", dst)
                shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"states": list(STATES), "noise_scale": STATES,
         "stimuli": list(STIM), "runs": args.runs, "fs": 1000.0,
         "dur_ms": T_END, "regions": REGIONS,
         "seed_map": "seed = 42 + (i*2+j)*runs + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
