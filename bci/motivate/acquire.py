"""bci/motivate: motivation gating of learning (--plastic-mod-scale).

The reinforcement gate now has a STRENGTH axis (mod amplitude = the
dopamine proxy's gain) -- the model's motivation/reward-strength
control, isomorphic to hungry flies conditioning better to odor+
sugar.  Single-shot design (condit-style): ONE paired trial (KC train
+ PAM reward, window 0.5-2.5 s) at mod-scale {0.125, 0.25, 0.5, 1.0},
plus a zero-learning reference (lr=0).  Readouts per trial:
  mechanism -- KCM w_scale mean after the single trial (state file);
  scalp    -- probe net response vs the lr=0 reference.
4 runs per arm = 20 trials, 5 s each.

Usage (conda ffbm, repo root):
    python bci/motivate/acquire.py
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
LR = "3e-6"
MODS = [0.125, 0.25, 0.5, 1.0]
T_END = 5000.0
SEED0 = 542


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    i = 0
    for mod in MODS + [0.0]:
        for r in range(args.runs):
            stem = (f"m{mod:g}_r{r}" if mod else f"nolr_r{r}")
            dst = out / f"{stem}.npy"
            nxt = st / f"{stem}.npz"
            if dst.exists() and nxt.exists():
                continue
            seed = SEED0 + i
            i += 1
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", "condt2",
                   "--plastic-mb", "--plastic-window", "500,2500",
                   "--plastic-lr", LR,
                   "--plastic-mod-scale", f"{mod:g}",
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu",
                   "--plastic-state-out", str(nxt)]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"mods": MODS, "lr": LR, "runs": args.runs,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "train_ms": [500, 2500], "probe_ms": [4300, 4700],
         "chem": "condt2 (KC 1600 pA + PAM 800 pA)",
         "note": "mod 0.0 arm runs lr=0 (zero-learning reference)",
         "seed_map": "sequential from 542"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
