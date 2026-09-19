"""bci/dangate: the error gate moved INTO the circuit (mechanism 15).

Successor to bci/blocking2 (experiment-layer Rescorla-Wagner).  The
reinforcement gate is now the DAN population firing rate itself:

  reward   PAM*@L/R chem injection (blockA/AB/C entries, 800 pA)
  V        potentiated KC->MBON readout (negative lr, w0 baseline)
  error    MBON->DAN (MBMD, use_sign) subtracts V from the reward drive
  mod      clip((r_DAN - r0) * gain, 0, 1)  -- per step, causal readback

Prediction: blocked arm (A+ then AB+) shows a SHRINKING DAN error
response and poor B learning; control arm (C+ then AB+) keeps the error
and B learns.  Readouts: per-trial w by presynaptic KC subtype (state
files) + per-trial DAN traces (_dan_trace.npy: t, r, mod).

Usage (conda ffbm, repo root):
    python bci/dangate/acquire.py --pilot
    python bci/dangate/acquire.py            # 2 arms x 2 sessions
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
LR = "-1e-7"            # negative = potentiation (pilot-tuned)
W0 = 0.1                # baseline weight / recovery target
KCM_GAIN = 30.0         # KC->MBON absolute conductance gain
MBMD_GAIN = 0.1         # MBON->DAN absolute conductance gain
DAN_GATE = "150,0.05,400"   # tau_ms, gain, t_ref_ms
T_END = 5000.0
N_P1, N_P2 = 4, 6
SEED0 = {"blocked": 942, "control": 992}
PRESENT = {"dangateA": ["A"], "dangateAB": ["A", "B"],
           "dangateC": ["C"]}


def run_trial(chem, seed, state_in, state_out, dst, tag):
    trial = HERE / "outputs" / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", f"--plastic-lr={LR}",   # '=' form: negative
           "--plastic-w0", str(W0),                 # values would eat as
           "--plastic-kcm-gain", f"{KCM_GAIN:g}",   # option names
           "--mbon-dan-gain", f"{MBMD_GAIN:g}",
           "--plastic-dan-gate", DAN_GATE,
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print(f"[acquire] {tag}: " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", HERE / "outputs" / f"{tag}_dan.npy")
    shutil.rmtree(trial)


def session(cond, r, n_p1, n_p2, prefix=""):
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    kfile = out / f"{prefix}{cond}_r{r}.json"
    if kfile.exists():
        done = json.loads(kfile.read_text(encoding="utf-8"))["next"]
    else:
        done = 0
    state = None
    plan = ([("p1", k, "dangateA" if cond == "blocked" else "dangateC")
             for k in range(n_p1)]
            + [("p2", k, "dangateAB") for k in range(n_p2)])
    for i, (phase, k, chem) in enumerate(plan):
        tag = f"{prefix}{cond}_r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if i < done:
            state = nxt
            continue
        run_trial(chem, SEED0[cond] + 10 * r + k, state, nxt, dst, tag)
        kfile.write_text(json.dumps({"next": i + 1}), encoding="utf-8")
        state = nxt


def main():
    global LR, KCM_GAIN, MBMD_GAIN, DAN_GATE, W0
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--lr", default=LR,
                    help="plasticity rate (negative = potentiation)")
    ap.add_argument("--w0", default=W0, type=float,
                    help="baseline weight (control-arm V level)")
    ap.add_argument("--kcm-gain", default=KCM_GAIN, type=float)
    ap.add_argument("--mbmd-gain", default=MBMD_GAIN, type=float)
    ap.add_argument("--dan-gate", default=DAN_GATE)
    args = ap.parse_args()
    LR, KCM_GAIN, MBMD_GAIN, DAN_GATE = (
        args.lr, args.kcm_gain, args.mbmd_gain, args.dan_gate)
    W0 = args.w0
    n_p1, n_p2 = (2, 3) if args.pilot else (N_P1, N_P2)
    prefix = "pilot_" if args.pilot else ""
    arms = ["blocked"] if args.pilot else ["blocked", "control"]
    for r in range(1 if args.pilot else 2):
        for cond in arms:
            session(cond, r, n_p1, n_p2, prefix)
    (HERE / "outputs" / f"{prefix}meta.json").write_text(json.dumps(
        {"conditions": arms, "n_p1": n_p1, "n_p2": n_p2,
         "sessions": 1 if args.pilot else 2, "fs": 1000.0,
         "dur_ms": T_END, "lr": LR, "w0": W0, "regions": REGIONS,
         "kcm_gain": KCM_GAIN, "mbmd_gain": MBMD_GAIN,
         "dan_gate": DAN_GATE,
         "gate": "circuit: mod=clip((r_DAN-r0)*g,0,1); V via MBMD "
                 "(use_sign) subtracts from PAM reward drive",
         "odors": {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"},
         "train_ms": [500, 2500], "probe_ms": [4200, 4700],
         "seed_map": "blocked 942+, control 992+ (+10r+k)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
