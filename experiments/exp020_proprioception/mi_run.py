"""exp020: motor-intention decoding from scalp EEG (fly MI-BCI analog).

Paradigm per trial (3.6 s, fast protocol):
  300-1200 ms   instruction window -- a 400 pA reafference pattern is
                injected into ONE leg-proprioceptor population
                (rootSide L = "left movement" class, R = "right")
  1200-3600 ms  delay window -- NO drive; tests whether any class
                information persists on the scalp (a central "motor
                plan" held without execution)

RESULTS (12 trials, 6/class, seeds 42-53; honest reading):
  cue    : L/R rms differ by 0.0023 uV on a 0.695 uV background
           (0.3%, perm p=0.002) -- nearest-centroid decodes 99-100%,
           BUT the signature rides entirely on the PRO_C terminal
           dipole (4,847 pairs; PRO fires, CEN stays at 0.005 Hz) and
           is ~30x below even the scaled hypothetical equipment noise
           of the k=0 premise (exp018) -- real, but it would need
           heavy averaging outside the clean-signal setting
  delay  : PRO and CEN are exactly 0.0 Hz after drive offset and the
           raw L/R rms difference is ~0.0001 uV (p=0.09-0.90) -- NO
           persistent plan information. (A first-pass band-envelope
           decode reported 78-89% in the delay window; that was the
           0.1-4 Hz filtfilt tail of the cue response -- the same
           long-impulse-response pitfall exp018 documented. The
           decoder below therefore uses memoryless raw-rms features
           and a direct label-permutation test per window.)
  takeaway: intention decoding during drive works; plan persistence
           needs central recruitment, which PRO_C (like GRN_C) does
           not achieve at the CEN_C=0.002 working point -- same
           structural lesson as exp019's taste path.

Acquisition: export_data.py per trial (region set includes the VNC so
PRO->VNC reflex arcs and vnc_motor populations are simulated; the scalp
projection itself only carries CEN-side currents by design).

Run from repository root (conda ffbm):
    python experiments/exp020_proprioception/mi_run.py            # all
    python experiments/exp020_proprioception/mi_run.py --decode   # only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "outputs" / "mi"
REGIONS = "visual_bilateral,ol_rest,central_brain,vnc,proprioception"
T_END = 3600.0
N_PER_CLASS = 6
SEED0 = 42
CLASSES = {"L": "mi_L_short", "R": "mi_R_short"}
FS = 1000.0
CUE = (350, 1150)
DELAY_SUBS = ((1300, 2000), (2000, 2700), (2700, 3500))


def acquire():
    OUT.mkdir(parents=True, exist_ok=True)
    for ci, (cls, chem) in enumerate(CLASSES.items()):
        for r in range(N_PER_CLASS):
            d = OUT / f"{cls}{r}"
            done = d / "viz_data.json"
            if done.exists():
                continue
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", chem, "--t-end", str(T_END),
                   "--seed", str(SEED0 + ci * N_PER_CLASS + r),
                   "--out", str(d)]
            print("[mi] " + " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True, cwd=str(ROOT))


def decode():
    """Memoryless per-window analysis: raw broadband rms per channel
    (no filter memory), class effect size + label-permutation p, plus
    the nearest-centroid split accuracy for reference."""
    vals = {c: [] for c in CLASSES}
    for cls in CLASSES:
        for r in range(N_PER_CLASS):
            p = np.load(OUT / f"{cls}{r}" / "_debug_phi_scalp.npy")
            p = p * 1e6 * 1.7
            wins = (CUE,) + DELAY_SUBS
            vals[cls].append([np.sqrt((p[a:b] ** 2).mean())
                              for a, b in wins])
    L = np.array(vals["L"])
    R = np.array(vals["R"])
    rng = np.random.default_rng(1)
    names = ["cue"] + [f"delay{int(a)}-{int(b)}" for a, b in DELAY_SUBS]
    print(f"trials: {[f'{c}{r}' for c in CLASSES for r in range(N_PER_CLASS)]}")
    for w, nm in enumerate(names):
        l, r = L[:, w], R[:, w]
        obs = l.mean() - r.mean()
        pool = np.concatenate([l, r])
        n = len(l)
        diff = []
        for _ in range(5000):
            pm = rng.permutation(pool)
            diff.append(pm[:n].mean() - pm[n:].mean())
        p = (np.sum(np.abs(diff) >= abs(obs)) + 1) / 5001
        print(f"{nm:12s}: L {l.mean():.4f}+-{l.std():.4f} | "
              f"R {r.mean():.4f}+-{r.std():.4f} uV | "
              f"diff {obs:+.4f} perm-p={p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", action="store_true")
    args = ap.parse_args()
    if not args.decode:
        acquire()
    decode()
