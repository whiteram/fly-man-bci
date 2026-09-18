"""bci/seizure gain sweep: onset-vs-gain relationship.

Runs the focal-drive seizure trial (800-2800 ms LAL*@L, 600 pA) at
explicit CEN_C recurrence gains between the safe working point
(0.002) and the default (0.004, known to self-ignite from noise in
~0.55 s -- bci/seizure README).  Question: is there an intermediate
gain where the network stays healthy until the DRIVE and the onset
becomes drive-locked (i.e. a controllable seizure-onset detection
task)?

Usage (conda ffbm, repo root):
    python bci/seizure/sweep.py               # 4 gains x 2 repeats
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GAINS = [0.002, 0.0025, 0.003, 0.0035]
REPEATS = 2
SEED0 = 42
T_END = 3600.0


def onset_of(phi, fs=1000, win=100, ratio=1.5):
    sig = (phi ** 2).mean(axis=1)
    rms = np.sqrt(np.convolve(sig, np.ones(win) / win, "same"))
    thr = rms[0:700].mean() * ratio
    idx = np.flatnonzero(rms[fs // 2:] > thr)
    return int(idx[0]) if len(idx) else -1


def main():
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    table = {}
    for gi, g in enumerate(GAINS):
        tag = str(g).replace("0.", "")
        ons = []
        for r in range(REPEATS):
            stem = f"szg_{tag}_r{r}.npy"
            p = out / stem
            if not p.exists():
                trial = out / f"_trial_szg_{tag}_r{r}"
                cmd = [sys.executable,
                       str(ROOT / "viz" / "export_data.py"),
                       "--regions",
                       "visual_bilateral,ol_rest,central_brain",
                       "--visual-input", "dark",
                       "--chem-input", f"szg_{tag}",
                       "--t-end", str(T_END),
                       "--seed", str(SEED0 + gi * REPEATS + r),
                       "--out", str(trial), "--gpu"]
                print(f"[sweep] gain={g} rep={r}: started", flush=True)
                r_ = subprocess.run(cmd, cwd=str(ROOT))
                if r_.returncode != 0:
                    raise SystemExit(f"export failed gain={g} rep={r}")
                np.save(p, np.load(trial / "_debug_phi_scalp.npy"))
                shutil_rmtree(trial)
            ons.append(onset_of(np.load(p) * 1e6 * 1.7))
        table[g] = ons
        print(f"gain {g}: onsets {ons} ms")
    (out / "sweep.json").write_text(json.dumps(table, indent=1))
    print("sweep saved -> outputs/sweep.json")


def shutil_rmtree(p):
    import shutil
    shutil.rmtree(p)


if __name__ == "__main__":
    main()
