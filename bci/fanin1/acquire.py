"""bci/fanin1: KC->MBON readout-leg fan-in density.

Audit premise (2026-09-21): the original queue item "APL fan-in via
a lower connectome threshold" is FALSIFIED -- APL->KC already covers
100% of KCs at the w>=5 cut (4,104 pairs / 4,064 KCs; the weak tail
adds only +13% synapses and zero coverage).  The weak-tail mass sits
on the LEARNING legs instead: KC->MBON +27,714 pairs (+83%,
33,496 -> 61,210 at w>=1), DAN->KC +123k, KC->DAN +136k (both
damped legs, inactive in the current loop).  This study perturbs the
active one: --kcm-fanin W pulls the weak KC->MBON rows into the
plastic pool.

Arms (dangate2 calibration, global gate, chem dangateA = A+ pairing
with US, 5-trial chained sessions, seeds 3042+):
  base  w>=5 pool (33,398 modeled pairs), lr -2e-7
  f2    + w in [2,5) rows, same lr
  f1    + w in [1,5) rows (61,210-pair pool), same lr
  f1r   f1 with lr rescaled by the pool ratio (33,398/61,210 =
        0.546 -> lr -1.094e-7): the "same total plasticity" control

Readout (analyze.py): per-trial w_all / w_trained (KCg-m) /
grown_frac / mod integral (in-circuit RW decrement) / A-probe rms;
baseline + max scalp for stability (the mb-mod DPM lesson: density
changes can latch).

Usage (conda ffbm, repo root):
    python bci/fanin1/acquire.py
    python bci/fanin1/analyze.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
CHEM = "dangateA"
W0 = "0.03"
KCM_GAIN = "0.015"
MBMD_GAIN = "178"
DAN_GATE = "150,0.02,400"
MBON_BIAS = -300.0
T_END = 5000.0
N_TRIALS = 5
SEED0 = 3042
ARMS = {
    "base": {"fanin": None, "lr": "-2e-7"},
    "f2": {"fanin": "2", "lr": "-2e-7"},
    "f1": {"fanin": "1", "lr": "-2e-7"},
    "f1r": {"fanin": "1", "lr": "-1.094e-7"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma list from base,f2,f1,f1r")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    arms = list(ARMS) if not args.arms else args.arms.split(",")
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    for arm in arms:
        a = ARMS[arm]
        state = None
        for k in range(args.trials):
            tag = f"{arm}_t{k}"
            dst = out / f"{tag}.npy"
            nxt = out / "states" / f"{tag}.npz"
            if dst.exists() and nxt.exists():
                state = nxt
                continue
            trial = out / f"_trial_{tag}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", CHEM,
                   "--plastic-mb", f"--plastic-lr={a['lr']}",
                   "--plastic-w0", W0,
                   "--plastic-kcm-gain", KCM_GAIN,
                   "--mbon-dan-gain", MBMD_GAIN,
                   "--plastic-dan-gate", DAN_GATE,
                   f"--mbon-bias={MBON_BIAS:g}",
                   "--t-end", str(T_END), "--seed", str(SEED0 + k),
                   "--out", str(trial), "--gpu",
                   "--plastic-state-out", str(nxt)]
            if a["fanin"]:
                cmd += ["--kcm-fanin", a["fanin"]]
            if state is not None:
                cmd += ["--plastic-state-in", str(state)]
            print(f"[acquire] {tag}: " + " ".join(cmd[1:]), flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {tag}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.copy(trial / "_dan_trace.npy",
                        out / f"{tag}_dan.npy")
            shutil.rmtree(trial)
            state = nxt
    (out / "meta.json").write_text(json.dumps(
        {"arms": {a: ARMS[a] for a in arms}, "chem": CHEM,
         "n_trials": args.trials, "dur_ms": T_END,
         "seed0": SEED0, "fs": 1000.0, "regions": REGIONS,
         "w0": W0, "kcm_gain": KCM_GAIN, "mbmd_gain": MBMD_GAIN,
         "dan_gate": DAN_GATE, "mbon_bias": MBON_BIAS,
         "train_ms": [500, 2500], "probe_ms": [4200, 4700]},
        indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
