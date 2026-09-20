"""bci/dangate2: the circuit error gate WITH MBON headroom restored.

Successor to bci/dangate (1.11x blocking; bottleneck: the untrained
MBON baseline already saturates MBMD suppression, so V growth cannot
differentiate the gate -- blocked/control DAN traces were nearly
identical at 8.5/8.8 Hz, mod floor 0.42).  Fix: a tonic hyperpolarizing
bias (--mbon-bias -300 pA ~= -30 mV on all 96 CEN MBONs, a chem
channel spanning the trial) moves the untrained point BELOW the
w-knee, so the knee itself becomes the trained/untrained discriminator.

Calibration (bci/dangate2/outputs/scan, static w, seed 1011):
    bias 0:    w 0.03->0.2 gives DAN r 15.5->9.4 Hz  (1.6x, saturated)
    bias -300: w 0.03->0.2 gives DAN r 51.2->18.4 Hz (2.8x, headroom)
               w>=0.3 (trained) -> r <=11.5 Hz
Gate gain 0.05 -> 0.02 so untrained r~51 saturates mod at 1.0 while
trained r~10 gives mod ~0.2 (5:1 learning-rate contrast vs 1.05:1).

Usage (conda ffbm, repo root):
    python bci/dangate2/acquire.py --pilot
    python bci/dangate2/acquire.py            # 2 arms x 2 sessions
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
LR = "-2e-7"            # negative = potentiation (full-run tuned)
W0 = 0.03               # untrained baseline weight
KCM_GAIN = 0.015        # KC->MBON absolute conductance gain
MBMD_GAIN = 178.0       # MBON->DAN absolute conductance gain
DAN_GATE = "150,0.02,400"   # tau_ms, gain, t_ref_ms (gain retuned)
MBON_BIAS = -300.0      # pA, tonic MBON hyperpolarization
T_END = 5000.0
N_P1, N_P2 = 4, 6
SEED0 = {"blocked": 1042, "control": 1092}
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
           f"--mbon-bias={MBON_BIAS:g}",
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
    global LR, KCM_GAIN, MBMD_GAIN, DAN_GATE, W0, MBON_BIAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--lr", default=LR,
                    help="plasticity rate (negative = potentiation)")
    ap.add_argument("--w0", default=W0, type=float,
                    help="baseline weight (control-arm V level)")
    ap.add_argument("--kcm-gain", default=KCM_GAIN, type=float)
    ap.add_argument("--mbmd-gain", default=MBMD_GAIN, type=float)
    ap.add_argument("--dan-gate", default=DAN_GATE)
    ap.add_argument("--mbon-bias", default=MBON_BIAS, type=float,
                    help="tonic MBON hyperpolarization in pA (negative)")
    args = ap.parse_args()
    LR, KCM_GAIN, MBMD_GAIN, DAN_GATE = (
        args.lr, args.kcm_gain, args.mbmd_gain, args.dan_gate)
    W0 = args.w0
    MBON_BIAS = args.mbon_bias
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
         "dan_gate": DAN_GATE, "mbon_bias": MBON_BIAS,
         "gate": "circuit: mod=clip((r_DAN-r0)*g,0,1); V via MBMD "
                 "(use_sign) subtracts from PAM reward drive; MBON "
                 "bias restores suppression headroom",
         "odors": {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"},
         "train_ms": [500, 2500], "probe_ms": [4200, 4700],
         "seed_map": "blocked 1042+, control 1092+ (+10r+k)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
