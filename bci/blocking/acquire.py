"""bci/blocking: Kamin blocking test of the three-factor rule.

The DAN-gated plasticity rule is PURE Hebbian (elig x mod, NO error
term).  Learning theory makes a falsifiable prediction: pre-training
A+ should NOT block B from learning during AB+ (Rescorla-Wagner-style
error correction would).  Synthetic odors are KC SUBTYPES:

  A = KCg-m (1,342 cells)   B = KCab-m (536)   C = KCab-s (657)

Sessions (state-chained, weights FROZEN between trials -- no tau_w):
  blocked: 4 x blockA  (A+ pre-training)  -> 6 x blockAB (AB+)
  control: 4 x blockC  (C+ pre-training)  -> 6 x blockAB (AB+)
2 seeds each = 40 trials, 5 s per trial.  Readouts:
  scalp -- probe net response per trial (phase-1 target probe, then
          the B-probe acquisition curve across phase 2);
  mechanism -- final-state w_scale decomposed by pre-synaptic KC
          subtype (state files carry per-edge pre ids).

Prediction (no error term): B learns equally in both conditions.

Usage (conda ffbm, repo root):
    python bci/blocking/acquire.py --pilot    # 1+1 trial chain check
    python bci/blocking/acquire.py            # 40 trials (~45 min)
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
T_END = 5000.0
N_P1, N_P2 = 4, 6
SEED0 = {"blocked": 442, "control": 492}


def run_trial(chem, seed, state_in, state_out, dst, tag):
    trial = HERE / "outputs" / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", "--plastic-window", "500,2500",
           "--plastic-lr", LR,
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


def session(cond, r, n_p1, n_p2, prefix=""):
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    state = None
    for phase, k, chem in (
            [("p1", k, "blockA" if cond == "blocked" else "blockC")
             for k in range(n_p1)]
            + [("p2", k, "blockAB") for k in range(n_p2)]):
        tag = f"{prefix}{cond}_r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if dst.exists() and nxt.exists():
            state = nxt
            continue
        run_trial(chem, SEED0[cond] + 10 * r + k, state, nxt, dst, tag)
        state = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    n_p1, n_p2 = (1, 1) if args.pilot else (N_P1, N_P2)
    prefix = "pilot_" if args.pilot else ""
    for r in range(1 if args.pilot else 2):
        session("blocked", r, n_p1, n_p2, prefix)
        session("control", r, n_p1, n_p2, prefix)
    (HERE / "outputs" / f"{prefix}meta.json").write_text(json.dumps(
        {"conditions": ["blocked", "control"], "n_p1": n_p1,
         "n_p2": n_p2, "sessions": 1 if args.pilot else 2,
         "fs": 1000.0, "dur_ms": T_END, "lr": LR, "regions": REGIONS,
         "odors": {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"},
         "train_ms": [500, 2500], "probe_ms": [4300, 4700],
         "seed_map": "blocked 442+, control 492+ (+10r+k)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
