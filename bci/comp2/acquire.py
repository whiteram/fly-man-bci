"""bci/comp2: two compartments, two rewards -- independent associations
and the over-blocking test.

comp1 showed blocking is an AGGREGATE-error phenomenon (compartment
gate 1.00x vs global 1.41x).  The architecture's real capability claim
is the flip side: with TWO compartment-private reward channels, the
circuit should form TWO independent associations, and the aggregate
gate should OVER-BLOCK the second association (A's learned V suppresses
the shared DAN pool, closing the gate on B's trials).

Protocol (dangate2 calibration, 3+4 chained trials per arm, seed 5042+):
  p1  A+ = KCg-m odor + reward, 3 trials
      comp arm: compA  (US -> PAM07*, private gate)
      glob arm: dangateA (US -> DAN_err, global gate)
      ctl p1: compC / dangateC (C odor + same US -- A absent)
  p2  B+ = KCab-m odor + reward, 4 trials
      comp arm: compBp  (US -> PPL103*, private gate)
      glob arm: dangateB (US -> DAN_err, global gate)

Pre-registered readout: B-learning ratio = dw_B(trained) / dw_B(control)
-- comp ~1 (private gates, no crosstalk), glob <1 (aggregate V from A
over-blocks B).  A-learning confinement (comp < glob) is the comp1
result reproduced in passing.

Usage (conda ffbm, repo root):
    python bci/comp2/acquire.py
    python bci/comp2/analyze.py
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
LR = "-2e-7"
W0 = "0.03"
KCM_GAIN = "0.015"
MBMD_GAIN = "178"
DAN_GATE = "150,0.02,400"
MBON_BIAS = -300.0
T_END = 5000.0
N_P1, N_P2 = 3, 4
SEED0 = 5042
COMP_GATE = "PAM07*:0.16,PPL103*:0.02"
ARMS = {
    "comp_tr": {"p1": "compA", "p2": "compBp", "comp": COMP_GATE},
    "comp_ctl": {"p1": "compC", "p2": "compBp", "comp": COMP_GATE},
    "glob_tr": {"p1": "dangateA", "p2": "dangateB", "comp": None},
    "glob_ctl": {"p1": "dangateC", "p2": "dangateB", "comp": None},
}


def run_trial(arm, chem, seed, state_in, state_out, dst, tag):
    out = HERE / "outputs"
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", f"--plastic-lr={LR}",
           "--plastic-w0", W0,
           "--plastic-kcm-gain", KCM_GAIN,
           "--mbon-dan-gain", MBMD_GAIN,
           "--plastic-dan-gate", DAN_GATE,
           f"--mbon-bias={MBON_BIAS:g}",
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    comp = ARMS[arm]["comp"]
    if comp:
        cmd += ["--plastic-comp-reward", comp]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print(f"[acquire] {tag}: " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma list from "
                         "comp_tr,comp_ctl,glob_tr,glob_ctl")
    args = ap.parse_args()
    arms = list(ARMS) if not args.arms else args.arms.split(",")
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    for arm in arms:
        a = ARMS[arm]
        state = None
        plan = [("p1", k, a["p1"]) for k in range(N_P1)] \
            + [("p2", k, a["p2"]) for k in range(N_P2)]
        for i, (phase, k, chem) in enumerate(plan):
            tag = f"{arm}_{phase}{k}"
            dst = out / f"{tag}.npy"
            nxt = out / "states" / f"{tag}.npz"
            if dst.exists() and nxt.exists():
                state = nxt
                continue
            run_trial(arm, chem, SEED0 + i, state, nxt, dst, tag)
            state = nxt
    (out / "meta.json").write_text(json.dumps(
        {"arms": {a: ARMS[a] for a in arms}, "n_p1": N_P1,
         "n_p2": N_P2, "dur_ms": T_END, "seed0": SEED0, "fs": 1000.0,
         "lr": LR, "w0": W0, "kcm_gain": KCM_GAIN,
         "mbmd_gain": MBMD_GAIN, "dan_gate": DAN_GATE,
         "mbon_bias": MBON_BIAS, "regions": REGIONS,
         "train_ms": [500, 2500], "probe_ms": [4200, 4700]}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
