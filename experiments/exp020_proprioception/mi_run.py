"""exp020: motor-intention decoding from scalp EEG (fly MI-BCI analog).

Trial per class (3.6 s, fast protocol; class = side):
  300-1200 ms   instruction -- 400 pA reafference on ONE leg-
                proprioceptor population (rootSide L = "left", R =
                "right"); identical across all sets
  1300-3500 ms  imagery window -- set-dependent:
    v1   (outputs/mi)   NO drive: probes persistence after the cue
    v2   (outputs/mi2)  same-side LAL* group (317 cells, 300 pA) --
                        the connectome's dominant DN driver (LAL->
                        DN, part of the 61% cb_intrinsic share; the
                        CX->LAL->DN motor-command loop) lights the
                        motor-plan layer through REAL synapses --
                        "imagery = top-down hold without execution"
    ctrl (outputs/mi2c) same-side DN group (596-712 cells, 250 pA)
                        driven DIRECTLY -- the lumped-corollary-
                        discharge upper bound that bypasses routing

v1 RESULT (12 trials): cue decodable (0.3% effect, perm p=0.002, but
the signature is the PRO_C terminal dipole -- CEN stays at 0.005 Hz);
delay NO persistence (raw rms diff ~1e-4 uV, p=0.09-0.90). A first-pass
band-envelope decode "decoding" 78-89% in the delay was the 0.1-4 Hz
filtfilt tail of the cue response (exp018 pitfall) -- the decoder below
is memoryless raw rms + direct label permutation.

DN pathway evidence (dn_probe.py): DN input 3.92M syn = cb_intrinsic
61% + DN-DN 13% + ascending 13% + VPN 6.7%; PRO direct only 0.27% --
why v1 could not recruit the plan layer. DN output edges to the central
brain (58.7k pairs) match ORN_C (57.5k pairs), the structure that made
olfaction scalp-visible -- driving DN plugs into the central amplifier
at its output end.

Acquisition: export_data.py per trial (regions include the VNC so
PRO->VNC reflex arcs and vnc_motor are simulated; the scalp projection
carries CEN-side currents by design).

Run from repository root (conda ffbm):
    python experiments/exp020_proprioception/mi_run.py               # v2+ctrl
    python experiments/exp020_proprioception/mi_run.py --set v1      # v1 only
    python experiments/exp020_proprioception/mi_run.py --decode      # only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "outputs"
REGIONS = "visual_bilateral,ol_rest,central_brain,vnc,proprioception"
T_END = 3600.0
N_PER_CLASS = 6
SEED0 = 42
FS = 1000.0
CUE = (350, 1150)
DELAY_SUBS = ((1300, 2000), (2000, 2700), (2700, 3500))

SETS = {
    "v1": {"dir": "mi", "classes": {"L": "mi_L_short",
                                    "R": "mi_R_short"}},
    "v2": {"dir": "mi2", "classes": {"L": "mi2_L_short",
                                     "R": "mi2_R_short"}},
    "ctrl": {"dir": "mi2c", "classes": {"L": "mi2c_L_short",
                                        "R": "mi2c_R_short"}},
}


def acquire(which):
    for key in which:
        st = SETS[key]
        out = OUT / st["dir"]
        out.mkdir(parents=True, exist_ok=True)
        for ci, (cls, chem) in enumerate(st["classes"].items()):
            for r in range(N_PER_CLASS):
                d = out / f"{cls}{r}"
                if (d / "viz_data.json").exists():
                    continue
                cmd = [sys.executable,
                       str(ROOT / "viz" / "export_data.py"),
                       "--regions", REGIONS, "--visual-input", "dark",
                       "--chem-input", chem, "--t-end", str(T_END),
                       "--seed", str(SEED0 + ci * N_PER_CLASS + r),
                       "--out", str(d)]
                print(f"[{key}] " + " ".join(cmd[1:]), flush=True)
                subprocess.run(cmd, check=True, cwd=str(ROOT))


def decode(which):
    """Memoryless per-window analysis: raw broadband scalp rms per
    class (no filter memory), permutation p for the L-R difference,
    nearest-centroid split accuracy for reference."""
    rng = np.random.default_rng(1)
    names = ["cue"] + [f"delay{int(a)}-{int(b)}" for a, b in DELAY_SUBS]
    for key in which:
        st = SETS[key]
        out = OUT / st["dir"]
        vals = {c: [] for c in st["classes"]}
        for cls in st["classes"]:
            for r in range(N_PER_CLASS):
                p = np.load(out / f"{cls}{r}" / "_debug_phi_scalp.npy")
                p = p * 1e6 * 1.7          # v1 convention, comparable
                vals[cls].append([np.sqrt((p[a:b] ** 2).mean())
                                  for a, b in (CUE,) + DELAY_SUBS])
        L = np.array(vals["L"])
        R = np.array(vals["R"])
        print(f"\n== {key} ({st['dir']}) ==")
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
            acc = ((np.r_[l, r] > (l.mean() + r.mean()) / 2)
                   == np.r_[np.ones(n), np.zeros(n)]).mean()
            print(f"{nm:12s}: L {l.mean():.4f}+-{l.std():.4f} | "
                  f"R {r.mean():.4f}+-{r.std():.4f} uV | "
                  f"diff {obs:+.4f} perm-p={p:.3f} acc={acc:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v2,ctrl",
                    help="comma list from v1,v2,ctrl")
    ap.add_argument("--decode", action="store_true")
    args = ap.parse_args()
    which = [s.strip() for s in args.set.split(",") if s.strip()]
    if not args.decode:
        acquire(which)
    decode(which)
