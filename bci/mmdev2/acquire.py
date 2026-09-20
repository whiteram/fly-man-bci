"""bci/mmdev2: matched-intensity oddball -- true identity deviance.

The mmdev re-audit (2026-09-20) found STD is an intensity-dependent
gain control: the old deviant (DM2 at 600 pA = 4x the DA1 standard)
self-depleted deeper through the ORN_C STP gate and INVERTED the
deviance ratio (0.78 at U=0.05, -0.62 at U=0.1).  Chem entry
`mmn_match` removes the confound: the deviant DM2 pulse runs at the
SAME 150 pA as the standards, so it deviates by identity only.

Conditions (4 runs each, identical train and seeds policy):
  std_off       cascade adaptation only
  std_on_u0.05  --std-gates ORN_C:0.05,600   (A1 preserved)
  std_on_u0.1   --std-gates ORN_C:0.1,600    (A1 preserved)

Pre-registered prediction: std_on devB/A5 lifts ABOVE the std_off
devB/A5 iff the repetition-specific depletion of the DA1-carrying
ORN->CEN edges (while the DM2 edges stay fresh) contributes a genuine
identity-deviance signal (MMN analog) on top of any fixed DA1/DM2
pathway asymmetry.

Usage (conda ffbm, repo root):
    python bci/mmdev2/acquire.py
    python bci/mmdev2/analyze.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CONDS = {
    "std_off": [],
    "std_on_u0.05": ["--std-gates", "ORN_C:0.05,600"],
    "std_on_u0.1": ["--std-gates", "ORN_C:0.1,600"],
}
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 3600.0
SEED0 = 62


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of conditions to run")
    args = ap.parse_args()
    conds = CONDS
    if args.only:
        keep = set(args.only.split(","))
        conds = {c: e for c, e in CONDS.items() if c in keep}
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for cond, extra in conds.items():
        i = list(CONDS).index(cond)  # seed index is stable under --only
        for r in range(args.runs):
            stem = f"{cond}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.runs + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", "mmn_match", "--t-end", str(T_END),
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
         "dur_ms": T_END, "regions": REGIONS, "chem": "mmn_match",
         "events_ms": [[300, 600], [700, 1000], [1100, 1400],
                       [1500, 1800], [1900, 2200], [2300, 2600],
                       [2700, 3000]],
         "labels": ["A1", "A2", "A3", "A4", "A5", "devB", "recA"],
         "seed_map": "seed = 62 + i*runs + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
