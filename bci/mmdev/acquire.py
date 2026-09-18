"""bci/mmdev: mismatch-negativity-style deviance with/without STD.

Trial = olfactory oddball train: five DA1 standard pulses (150 pA,
300 ms), one DM2 deviant (600 pA, identity deviation), one DA1
recovery pulse.  Condition std_on adds --std-gates "ORN_C:0.25,600"
(short-term depression on the first olfactory synapse); std_off runs
the identical train with cascade adaptation only.  4 runs each.

Analysis (analyze.py): per-event net response; repetition suppression
(A1..A5 slope), deviance ratio dev/A5, recovery.

Usage (conda ffbm, repo root):
    python bci/mmdev/acquire.py
    python bci/mmdev/analyze.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CONDS = {"std_on": ["--std-gates", "ORN_C:0.25,600"], "std_off": []}
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, (cond, extra) in enumerate(CONDS.items()):
        for r in range(args.runs):
            stem = f"{cond}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.runs + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", "mmn_train", "--t-end", str(T_END),
                   "--seed", str(seed), "--out", str(trial), "--gpu"]
            cmd = cmd[:2] + extra + cmd[2:]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"conditions": list(CONDS), "runs": args.runs, "fs": 1000.0,
         "dur_ms": T_END, "regions": REGIONS,
         "events_ms": [[300, 600], [700, 1000], [1100, 1400],
                       [1500, 1800], [1900, 2200], [2300, 2600],
                       [2700, 3000]],
         "labels": ["A1", "A2", "A3", "A4", "A5", "devB", "recA"],
         "seed_map": "seed = 42 + i*runs + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
