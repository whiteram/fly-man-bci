"""Looming paradigm acquisition: 3 classes x repeats, visual+OLR+CEN
regions (the escape path R -> cascade -> LC/LPLC(OLR) -> DN(CEN) is
inside these; DN is a scalp source via CEN_C).  Class stimulus via
--visual-input loom_<cls>; trial scalp EEG kept as outputs/<cls>_r<r>.npy.

Usage (conda ffbm, repo root):
    python bci/looming/acquire.py               # 3 x 6
    python bci/looming/acquire.py --repeats 1   # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = ["loom", "recede", "static"]
REGIONS = "visual_bilateral,ol_rest,central_brain"
T_END = 3000.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    import make_stimulus as ms

    for c in ms.CLASSES:
        ms.make(c)
    for i, cls in enumerate(CLASSES):
        for r in range(args.repeats):
            stem = f"{cls}_r{r}"
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = SEED0 + i * args.repeats + r
            trial = out / f"_trial_{stem}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", f"loom_{cls}",
                   "--t-end", str(T_END), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            print("[acquire] " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {stem}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"classes": CLASSES, "repeats": args.repeats, "fs": 1000.0,
         "dur_ms": T_END, "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
