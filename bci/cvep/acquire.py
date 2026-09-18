"""c-VEP acquisition driver (BETA analog): one simulated recording =
one export_data.py run per target code (visual-only circuit, caches
make each trial ~1 min).  Per-trial clean scalp EEG harvested from
_debug_phi_scalp.npy (x1.7 calibrated, 17 leads x 1 kHz) and kept as
outputs/eeg_t<k>_r<r>.npy.

Usage (conda ffbm, repo root):
    python bci/cvep/acquire.py --targets 0 11 22 33 --repeats 2  # pilot
    python bci/cvep/acquire.py --all40 --repeats 2              # full
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY = sys.executable

T_END = 3000.0
REGIONS = "visual_bilateral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+")
    ap.add_argument("--all40", action="store_true")
    ap.add_argument("--repeats", type=int, default=2,
                    help="acquisitions per target, each with its own "
                         "RNG seed (42 + i*repeats + r)")
    a = ap.parse_args()
    if a.all40:
        a.targets = list(range(40))
    if not a.targets:
        ap.error("--targets or --all40 required")
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    import make_code as mc

    for i, k in enumerate(a.targets):
        mc.make(k)
        for r in range(a.repeats):
            stem = f"eeg_t{k:02d}" + ("" if a.repeats == 1
                                      else f"_r{r}")
            dst = out / f"{stem}.npy"
            if dst.exists():
                continue
            seed = 42 + i * a.repeats + r
            trial = out / f"_trial_t{k:02d}_r{r}"
            cmd = [PY, str(ROOT / "viz" / "export_data.py"),
                   "--visual-input", f"cvep_t{k:02d}",
                   "--regions", REGIONS,
                   "--t-end", str(T_END),
                   "--out", str(trial), "--seed", str(seed), "--gpu"]
            print(f"[acquire] t={k} rep={r} seed={seed}: started",
                  flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for t={k} rep={r}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.rmtree(trial)
    (out / "meta.json").write_text(json.dumps(
        {"targets": a.targets, "repeats": a.repeats, "fs": 1000.0,
         "dur_ms": T_END, "code": {"n_bits": mc.N_BITS,
                                   "shift_step": mc.SHIFT_STEP,
                                   "fps": mc.FPS},
         "epoch_ms": [mc.LEAD_FR / mc.FPS * 1000,
                      (mc.LEAD_FR + mc.N_BITS) / mc.FPS * 1000],
         "elec": "default 17-lead scalp",
         "regions": REGIONS,
         "seed_map": "seed = 42 + i*repeats + r (i=class idx, r=rep)"},
        indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
