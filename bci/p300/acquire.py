"""P300 oddball acquisition: 6 runs of the same train, different OU
seeds; event epochs are cut in analysis.  One run = one export_data.py
visual-only trial (12 s, ~4 min GPU).

Usage (conda ffbm, repo root):
    python bci/p300/acquire.py               # 6 runs
    python bci/p300/acquire.py --runs 2      # smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral"
T_END = 12000.0
SEED0 = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    import make_stimulus as ms
    ms.make()
    for r in range(args.runs):
        dst = out / f"train_r{r}.npy"
        if dst.exists():
            continue
        trial = out / f"_trial_r{r}"
        cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
               "--regions", REGIONS, "--visual-input", "p300_train",
               "--t-end", str(T_END), "--seed", str(SEED0 + r),
               "--out", str(trial), "--gpu"]
        print(f"[acquire] run {r} seed {SEED0 + r}: started", flush=True)
        r_ = subprocess.run(cmd, cwd=str(ROOT))
        if r_.returncode != 0:
            raise SystemExit(f"export failed run {r}")
        shutil.copy(trial / "_debug_phi_scalp.npy", dst)
        shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"runs": args.runs, "fs": 1000.0, "dur_ms": T_END,
         "regions": REGIONS, "soa_ms": ms.SOA, "flash_ms": ms.FLASH,
         "events": ms.EVENTS, "targets": list(ms.TARGETS),
         "seed_map": "seed = 42 + r"}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
