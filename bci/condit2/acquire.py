"""bci/condit2: acquisition-extinction CURVES (cross-trial memory).

The follow-up to bci/condit: KCM plastic weights now PERSIST across
trials (--plastic-state-in/out chaining) and recover slowly
(--plastic-tau-w 30 s), so a session traces the classic conditioning
arc instead of one-shot snapshots.  Trial 5 s, one probe per trial:

  acquisition (a):  10 x condt2  -- KC train 0.5-2.5 s + PAM reward,
                    --plastic-window open -> KC->MBON depression
                    ACCUMULATES trial over trial; probe 4.2-4.7 s each
                    trial reads the growing pathway change
  extinction (x):   8 x condtx   -- same KC train, reward OMITTED (the
                    mod gate stays closed) -> only the slow recovery
                    acts: w crawls back toward 1, probe response decays
  control (c):      8 x condtx   -- never-rewarded session: a flat
                    probe curve rules out drift/order artifacts

Conditions: paired (acquisition then extinction, one chained session)
vs ctrl (flat).  2 sessions (seeds) each.

Usage (conda ffbm, repo root):
    python bci/condit2/acquire.py --pilot    # 2+1 trial chain check
    python bci/condit2/acquire.py            # full 52-trial study
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
TAU_W = "30000"
T_END = 5000.0
N_ACQ, N_EXT, N_CTRL = 10, 8, 8
SEED0 = 142


def run_trial(chem, window, seed, state_in, state_out, dst, tag):
    trial = HERE / "outputs" / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", "--plastic-lr", LR,
           "--plastic-tau-w", TAU_W,
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    if window:
        cmd += ["--plastic-window", window]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print("[acquire] " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.rmtree(trial)


def session(cond, r, n_acq, n_ext, seed0, prefix=""):
    """One chained session; returns [(phase, k, npy, state), ...]."""
    out = HERE / "outputs"
    st = HERE / "outputs" / "states"
    st.mkdir(parents=True, exist_ok=True)
    state = None
    if cond == "paired":
        plan = ([("a", k) for k in range(n_acq)]
                + [("x", k) for k in range(n_ext)])
    else:
        plan = [("c", k) for k in range(n_acq)]
    done = []
    for phase, k in plan:
        tag = f"{prefix}{cond}_r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if dst.exists() and nxt.exists():
            state = nxt
            done.append((phase, k, dst, nxt))
            continue
        chem = "condt2" if phase == "a" else "condtx"
        window = "500,2500" if phase == "a" else None
        run_trial(chem, window, seed0 + 10 * r + k, state, nxt, dst, tag)
        state = nxt
        done.append((phase, k, dst, nxt))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--pilot", action="store_true",
                    help="2 acquisition + 1 extinction trial, 1 session")
    args = ap.parse_args()
    n_acq, n_ext, n_ctrl = N_ACQ, N_EXT, N_CTRL
    prefix, runs = "", args.runs
    if args.pilot:
        n_acq, n_ext, n_ctrl, runs, prefix = 2, 1, 0, 1, "pilot_"
    for r in range(runs):
        session("paired", r, n_acq, n_ext, SEED0, prefix)
        session("ctrl", r, n_ctrl, 0, SEED0 + 50, prefix)
    (HERE / "outputs" / f"{prefix}meta.json").write_text(json.dumps(
        {"sessions": runs if not args.pilot else 1,
         "n_acq": n_acq, "n_ext": n_ext, "n_ctrl": n_ctrl,
         "fs": 1000.0, "dur_ms": T_END, "regions": REGIONS,
         "lr": LR, "tau_w_ms": float(TAU_W),
         "train_ms": [500, 2500], "probe_ms": [4300, 4700],
         "chem": {"a": "condt2", "x": "condtx", "c": "condtx"},
         "state_chaining": True,
         "seed_map": "paired: 142+10r+k; ctrl: 192+10r+k",
         "pilot": bool(args.pilot)}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
