"""bci/dose: odor concentration decoding (DA1 at 0.5x/1x/2x).

Classes: dose_075 (75 pA), odor_da1_short (150 pA), dose_300 (300 pA),
single 900 ms pulse, 6 repeats each.  Same region set as bci/olfactory
(cached).  Analysis: 3-class spatial LOO + the dose-response curve.

Usage (conda ffbm, repo root):
    python bci/dose/acquire.py               # 3 levels x 6
    python bci/dose/acquire.py --repeats 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

LEVELS = [("075", "dose_075"), ("150", "odor_da1_short"),
          ("300", "dose_300")]
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
T_END = 3600.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    for i, (tag, chem) in enumerate(LEVELS):
        for r in range(args.repeats):
            stem = f"d{tag}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", chem,
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"levels": [t for t, _ in LEVELS], "repeats": args.repeats,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "amps": [75.0, 150.0, 300.0],
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
