"""bci/blocking2: Kamin blocking WITH a reward-prediction-error gate.

The companion to bci/blocking (which showed the plain window gate
produces NO blocking).  Here the reinforcement gate implements the
Rescorla-Wagner rule at the EXPERIMENT layer: per-trial

  error = R - sum(V_x for x present),   mod = clip(error, 0, 1)
  V_x  += alpha * error   (after the trial; V persisted per session)

with R = 1 on rewarded trials, alpha = 0.3, V start 0.  The gate
strength feeds the existing --plastic-mod-scale machinery, so the
synaptic rule itself is unchanged -- ONLY the gate becomes
error-driven.  Prediction (the classic blocking mechanism):

  blocked (A+ then AB+):  V_A ~ 1 after phase 1 -> AB+ error ~ 0 ->
                          mod ~ 0 -> B does NOT learn
  control (C+ then AB+):  V_C irrelevant in AB+ -> error ~ 1 ->
                          mod ~ 1 -> B learns

If this reproduces blocking where bci/blocking showed its absence,
the two studies together cleanly demonstrate that blocking in this
model is a property of the GATE, not the synapse.

Honest framing: the error computation lives in the acquisition script
(a per-odor V table); a CIRCUIT implementation -- MBON->DAN feedback
making dopamine itself error-coding -- is the mechanism-grade
follow-up.

Usage (conda ffbm, repo root):
    python bci/blocking2/acquire.py --pilot
    python bci/blocking2/acquire.py            # 40 trials (~40 min)
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
ALPHA = 0.3
T_END = 5000.0
N_P1, N_P2 = 4, 6
SEED0 = {"blocked": 842, "control": 892}
PRESENT = {"blockA": ["A"], "blockAB": ["A", "B"], "blockC": ["C"]}


def run_trial(chem, seed, mod, state_in, state_out, dst, tag):
    trial = HERE / "outputs" / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem,
           "--plastic-mb", "--plastic-window", "500,2500",
           "--plastic-lr", LR,
           "--plastic-mod-scale", f"{mod:g}",
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print(f"[acquire] {tag}: mod={mod:.3f} " + " ".join(cmd[1:]),
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.rmtree(trial)


def session(cond, r, n_p1, n_p2, prefix=""):
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    vpath = out / f"{prefix}V_{cond}_r{r}.json"
    if vpath.exists():
        vstate = json.loads(vpath.read_text(encoding="utf-8"))
    else:
        vstate = {"V": {"A": 0.0, "B": 0.0, "C": 0.0}, "next": 0}
    state = None
    for phase, k, chem in (
            [("p1", k, "blockA" if cond == "blocked" else "blockC")
             for k in range(n_p1)]
            + [("p2", k, "blockAB") for k in range(n_p2)]):
        tag = f"{prefix}{cond}_r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if vstate["next"] > k + (0 if phase == "p1" else n_p1):
            state = nxt
            continue
        present = PRESENT[chem]
        error = max(0.0, 1.0 - sum(vstate["V"][x] for x in present))
        run_trial(chem, SEED0[cond] + 10 * r + k, error,
                  state, nxt, dst, tag)
        for x in present:
            vstate["V"][x] += ALPHA * error
        vstate["next"] = k + 1 + (0 if phase == "p1" else n_p1)
        vpath.write_text(json.dumps(vstate, indent=1), encoding="utf-8")
        print(f"[V] {tag}: error={error:.3f} V={vstate['V']}", flush=True)
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
         "gate": "rescorla-wagner (alpha=0.3, R=1, mod=clip(R-sum V,"
                 "0,1); V persisted per session)",
         "odors": {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"},
         "train_ms": [500, 2500], "probe_ms": [4300, 4700],
         "seed_map": "blocked 842+, control 892+ (+10r+k)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
