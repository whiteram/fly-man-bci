"""SSVEP acquisition driver: one simulated recording session = one
export_data.py run per stimulus frequency (circuit + kernel caches make
each trial ~1 min at visual-only scale).  Per-trial clean scalp EEG is
harvested from the exported _debug_phi_scalp.npy (x1.7 calibrated,
45 leads x 1 kHz, no background -- see bci/README.md).

Usage (pilot with 6 frequencies):
  python bci/ssvep_benchmark/acquire.py --freqs 8.0 9.8 11.6 13.4 15.2 15.8
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=float, nargs="+")
    ap.add_argument("--all40", action="store_true",
                    help="run the full 40-frequency benchmark set")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--regions", type=str, default="visual_bilateral")
    a = ap.parse_args()
    if a.all40:
        sys.path.insert(0, str(HERE))
        import make_stimulus as ms0
        a.freqs = ms0.freqs()
    if not a.freqs:
        ap.error("--freqs or --all40 required")
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    import make_stimulus as ms

    meta = {"freqs": a.freqs, "fs": 1000.0, "dur_s": 10.5,
            "elec": "elec_layout_1010 (45 leads)",
            "scale": "thought-experiment x202 (nominal x400)",
            "mode": "single-target (attended-slot analog)"}
    for f in a.freqs:
        ms.make(f, single=True)
        trial_out = out / f"_trial_f{f:.1f}"
        cmd = [PY, str(ROOT / "viz" / "export_data.py"),
               "--visual-input", f"ssvep1_f{f:.1f}",
               "--regions", a.regions,
               "--elec-layout", str(ROOT / "viz" / "data"
                                    / "elec_layout_1010.json"),
               "--out", str(trial_out)]
        if a.gpu:
            cmd.append("--gpu")
        print(f"[acquire] f={f}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"export failed for f={f}")
        shutil.copy(trial_out / "_debug_phi_scalp.npy",
                    out / f"eeg_f{f:.1f}.npy")
        shutil.rmtree(trial_out)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
