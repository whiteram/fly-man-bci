"""bci/attend: attention as pathway-gain modulation (--gain-scale).

The hybrid experiment found the SSVEP line masked ~500x when motor
imagery runs concurrently.  The mechanistic attention model here is
the standard one -- attention = gain modulation of the attended
pathway -- realized by --gain-scale OLR_V=k (the visual cascade's
feedforward group into central brain; g_unit is a runtime scalar, no
kernel rebuild).  Questions:

  1. RECOVERY DOSE-RESPONSE: does the visual line come back at k=4
     (upper physiological) / k=16 (super-physiological)?
  2. COST: does boosting the visual pathway hurt the imagery decode
     (attention trade-off)?
  3. CONFOUND CHECK for hybrid: the hyb_n control arm was run WITHOUT
     --chem-input (default gains) although its chem entry intends the
     0.002 working point -- the 500x masking number mixes the working
     point with true dual-task interference.  Arm "nwp" re-runs the
     no-imagery control WITH hyb_n (0.002) to split the two.

6 imagery arms (2 freqs x 2 sides x {k0, k4, k16}) x 2 repeats + 2
nwp trials = 26 trials, 10.5 s each.

Usage (conda ffbm, repo root):
    python bci/attend/acquire.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

FREQS = [10.0, 14.0]
SIDES = ["l", "r"]
K_ARMS = {"k0": None, "k4": "OLR_V=4", "k16": "OLR_V=16"}
REGIONS = "visual_bilateral,ol_rest,central_brain"
T_END = 10500.0
SEED0 = 342


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / "bci" / "ssvep_benchmark"))
    import make_stimulus as ms0
    for f in FREQS:
        ms0.make(f, single=True)
    i = 0
    for f in FREQS:
        for side in SIDES:
            for kname, kspec in K_ARMS.items():
                for r in range(args.repeats):
                    stem = f"f{f:.0f}_{side}_{kname}_r{r}"
                    dst = out / f"{stem}.npy"
                    if dst.exists():
                        continue
                    seed = SEED0 + i
                    i += 1
                    trial = out / f"_trial_{stem}"
                    cmd = [sys.executable,
                           str(ROOT / "viz" / "export_data.py"),
                           "--regions", REGIONS,
                           "--visual-input", f"ssvep1_f{f:.0f}",
                           "--chem-input", f"hyb_{side}",
                           "--t-end", str(T_END), "--seed", str(seed),
                           "--out", str(trial), "--gpu"]
                    if kspec:
                        cmd += ["--gain-scale", kspec]
                    print("[acquire] " + " ".join(cmd[1:]), flush=True)
                    r_ = subprocess.run(cmd, cwd=str(ROOT))
                    if r_.returncode != 0:
                        raise SystemExit(f"export failed for {stem}")
                    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
                    shutil.rmtree(trial)
    for r in range(args.repeats):           # confound-check arm
        stem = f"f10_nwp_r{r}"
        dst = out / f"{stem}.npy"
        if dst.exists():
            continue
        trial = out / f"_trial_{stem}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "ssvep1_f10",
               "--chem-input", "hyb_n",
               "--t-end", str(T_END), "--seed", str(SEED0 + 90 + r),
               "--out", str(trial), "--gpu"]
        print("[acquire] " + " ".join(cmd[1:]), flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed for {stem}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"freqs": FREQS, "sides": SIDES, "k_arms": K_ARMS,
         "repeats": args.repeats, "fs": 1000.0, "dur_ms": T_END,
         "regions": REGIONS, "imagery_ms": [1500, 9800],
         "gain_group": "OLR_V (visual cascade -> central feedforward)",
         "nwp": "no-imagery control WITH hyb_n chem (0.002 working "
                "point) -- hybrid confound check"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
