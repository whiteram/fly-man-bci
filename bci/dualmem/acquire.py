"""bci/dualmem: short- vs long-term memory dissociation (dual tau_w).

The fly's memory compartments: gamma Kenyon cells mediate short-lived
memory, alpha/beta the persistent one.  Here KCM is split by
presynaptic KC subtype into TWO plastic groups with separate recovery
constants (--plastic-dual-tauw 30000,300000):

  KCMg   pre type KCg*  (gamma, 1,548 cells) -- tau_w 30 s
  KCMab  the rest       (alpha/beta-like)    -- tau_w 300 s

Trial 5.2 s: train ALL KC 0.5-2.5 s + PAM reward (both groups learn);
TWO probes per trial: gamma window 4.2-4.45 s (KCg*), alpha/beta
window 4.6-4.85 s (KCab*).  Session: 8 acquisition (dual1) + 8
extinction (dualx, no reward -- recovery only).  2 seeds = 32 trials.

Prediction: acquisition rises on both probes; extinction decays FAST
on the gamma probe (tau_w 30 s ~ 16%/trial) and SLOW on the alpha/
beta probe (tau_w 300 s ~ 1.7%/trial) -- the compartment
dissociation, read out on one and the same network.

Usage (conda ffbm, repo root):
    python bci/dualmem/acquire.py --pilot    # 1+1 chain check
    python bci/dualmem/acquire.py            # 32 trials (~35 min)
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
LR = "3e-6"
DUAL = "30000,300000"
T_END = 5200.0
N_ACQ, N_EXT = 8, 8
SEED0 = 642


def run_trial(chem, seed, state_in, state_out, dst, tag, cmd_extra=()):
    trial = HERE / "outputs" / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", "--plastic-dual-tauw", DUAL,
           "--plastic-lr", LR, *cmd_extra,
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print("[acquire] " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.rmtree(trial)


def session(r, n_acq, n_ext, prefix=""):
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    state = None
    plan = ([("a", k, "dual1") for k in range(n_acq)]
            + [("x", k, "dualx") for k in range(n_ext)])
    for phase, k, chem in plan:
        tag = f"{prefix}r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if dst.exists() and nxt.exists():
            state = nxt
            continue
        cmd_extra = (["--plastic-window", "500,2500"]
                     if phase == "a" else [])
        run_trial(chem, SEED0 + 10 * r + k, state, nxt, dst, tag,
                  cmd_extra)
        state = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sessions", type=int, default=2)
    args = ap.parse_args()
    n_acq, n_ext, n_sess = N_ACQ, N_EXT, args.sessions
    prefix = ""
    if args.pilot:
        n_acq, n_ext, n_sess, prefix = 1, 1, 1, "pilot_"
    for r in range(n_sess):
        session(r, n_acq, n_ext, prefix)
    (HERE / "outputs" / f"{prefix}meta.json").write_text(json.dumps(
        {"sessions": n_sess, "n_acq": n_acq, "n_ext": n_ext,
         "fs": 1000.0, "dur_ms": T_END, "lr": LR,
         "dual_tauw": DUAL, "regions": REGIONS,
         "train_ms": [500, 2500],
         "probe_g_ms": [4300, 4400], "probe_ab_ms": [4700, 4800],
         "groups": {"KCMg": "KCg* tau_w 30 s",
                    "KCMab": "rest tau_w 300 s"},
         "seed_map": "642 + 10r + k"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
