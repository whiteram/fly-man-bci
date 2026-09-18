"""bci/hybrid: SSVEP flicker + motor imagery, dual-task acquisition.

Simultaneous streams in one recording (10.5 s trial):
  visual  full-screen single-slot flicker at 10 or 14 Hz
          (the SSVEP branch's single-target stimulus, both eyes)
  imagery same-side LAL*@side plateau drive 1200 pA [1500,9800]
          (hyb_l / hyb_r) or none (hyb_n -- pure visual at the same
          explicit CEN_C=0.002 working point, no gain confound)

6 conditions = 2 freqs x {none, L, R}, 3 repeats each.  Analyses
(analyze.py): (a) flicker-frequency classification by spectral peak
ratio; (b) imagery-side LOO decode; (c) dual-task interference =
SSVEP SNR in imagery runs vs none runs.

Usage (conda ffbm, repo root):
    python bci/hybrid/acquire.py               # 6 conditions x 3
    python bci/hybrid/acquire.py --repeats 1   # smoke
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
IMAGERY = ["n", "l", "r"]
REGIONS = "visual_bilateral,ol_rest,central_brain"
T_END = 10500.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / "bci" / "ssvep_benchmark"))
    import make_stimulus as ms0
    for f in FREQS:
        ms0.make(f, single=True)
    conds = [f"f{f:.0f}_{im}" for f in FREQS for im in IMAGERY]
    for i, cond in enumerate(conds):
        f, im = cond.split("_")
        for r in range(args.repeats):
            stem = f"{cond}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS,
                   "--visual-input", f"ssvep1_f{f}.0"]
            if im != "n":
                cmd += ["--chem-input", f"hyb_{im}"]
            cmd += ["--t-end", str(T_END), "--seed", str(seed),
                    "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"freqs": FREQS, "imagery": IMAGERY, "repeats": args.repeats,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "imagery_ms": [1500, 9800],
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
