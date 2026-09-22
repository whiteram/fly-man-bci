"""bci/kcdrv2: is the DAN->KC channel causally driveable?  (kcdrv1
follow-up, tiny)

kcdrv1 localized the plateau's KC +6 Hz to the reward channel via lag
locking + wiring triangulation (DAN->KC: 5,608 excitatory rows).  This
arc closes the loop with a direct causal probe: inject a DAN pool
ALONE (800 pA per cell, no odor channels) into the NAIVE dark network
(KC baseline = 0) and watch Kenyon_Cell.

Wiring prediction (CEN_C table): PPL103*'s 2 cells broadcast onto
1,978 rows / 1,469 distinct KC (mean w 7.07) -- a genuine US fan-out;
PAM07*'s 14 cells touch only 147 rows / 138 KC (w 5.47).  So PPL103
US should lift KC broadly, PAM07 US barely -- a 13x fan-in asymmetry
between the two conditioning US channels.

Arms (seed 616, 5 s, default gain):
  dark        no chem at all -- KC baseline (0 Hz expected)
  us_pam07    PAM07* only, 800 pA [500,2500] ms
  us_ppl103   PPL103* only, 800 pA [500,2500] ms

Usage (conda ffbm, repo root):
    python bci/kcdrv2/acquire.py
    python bci/kcdrv2/analyze.py
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
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
T_END = 5000.0
SEED = 616
ARMS = {"dark": None, "us_pam07": "us_pam07", "us_ppl103": "us_ppl103"}
WIN = {"baseline": (0, 400), "us": (600, 2400), "after": (2600, 4900)}


def run(tag, chem, amp=1.0):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if chem:
        cmd += ["--chem-input", chem]
    if amp != 1.0:
        cmd += ["--chem-amp-scale", f"{amp:g}"]
    print(f"[acquire] {tag}: chem={chem} amp={amp:g}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="dark,us_pam07,us_ppl103",
                    help="comma list; us_ppl103_xN adds an amplitude "
                         "xN arm on the PPL103 channel")
    args = ap.parse_args()
    for a in args.arms.split(","):
        if a.startswith("us_ppl103_x"):
            run(a, "us_ppl103", float(a.split("_x")[1]))
        else:
            run(a, ARMS[a])
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": args.arms.split(","), "groups": GROUPS,
         "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
         "windows_ms": {k: list(v) for k, v in WIN.items()},
         "us_pa": 800.0}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
