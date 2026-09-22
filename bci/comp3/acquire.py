"""bci/comp3: co-presence DUAL reward -- the missing cell of the 2x2.

comp2 measured the sequential cell (B rewarded, A absent: both gates
~1.0, no crosstalk -- value expression is spike-driven/state-dependent).
comp1 measured the co-present cells but B was never rewarded in its own
compartment (glob 1.41x blocking; comp w_B flat).  The open question:
with A AND B co-present and BOTH compartments rewarded simultaneously
(compABd: AB + PAM07* US + PPL103* US), does A's co-expressed V block
B under the aggregate gate while the private gate stays blind to it?

Protocol (comp1's 4+6 dangate2 form, 2 sessions):
  p1  x4   comp arm: compA (A + PAM07 US)  / ctl: compC (C + PAM07 US)
           glob reference: protocol-bitwise-identical to comp1's global
           arm (dangateA/dangateC -> dangateAB, seeds 1042+/1092+), so
           those states are REUSED verbatim -- zero acquisition.
  p2  x6   compABd (AB + dual compartment US, B probe), gate
           "PAM07*:0.16,PPL103*:0.02" (comp2 calibration)

Headline: B-learning ratio = dw_B(trained)/dw_B(control) at p2 end.
Prediction: comp ~1 (PPL103's private gate cannot see A's V -- the
compartments' gated edge sets are disjoint), glob ~1.4 (A's V
co-expressed during AB suppresses the aggregate gate; comp1 cell).

Usage (conda ffbm, repo root):
    python bci/comp3/acquire.py                  # reuse glob + comp arms
    python bci/comp3/acquire.py --arms comp_tr   # single chain
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C1 = ROOT / "bci" / "comp1" / "outputs"

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
LR = "-2e-7"
W0 = "0.03"
KCM_GAIN = "0.015"
MBMD_GAIN = "178"
DAN_GATE = "150,0.02,400"
MBON_BIAS = -300.0
T_END = 5000.0
N_P1, N_P2 = 4, 6
SESSIONS = 2
COMP_GATE = "PAM07*:0.16,PPL103*:0.02"
SEED0 = {"comp_tr": 6042, "comp_ctl": 6542}
ARMS = {
    "comp_tr": {"p1": "compA", "p2": "compABd", "comp": COMP_GATE},
    "comp_ctl": {"p1": "compC", "p2": "compABd", "comp": COMP_GATE},
    "glob_tr": {"p1": "dangateA", "p2": "dangateAB", "comp": None},
    "glob_ctl": {"p1": "dangateC", "p2": "dangateAB", "comp": None},
}
# glob arms == comp1 global arm (same chems/flags/seeds): copy states,
# scalp and dan traces verbatim, tag-for-tag.
REUSE_SRC = {"glob_tr": "global_blocked", "glob_ctl": "global_control"}


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


def reuse_glob(arm, sessions):
    src = REUSE_SRC[arm]
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    for r in range(sessions):
        for i in range(N_P1 + N_P2):
            ph = "p1" if i < N_P1 else "p2"
            k = i if ph == "p1" else i - N_P1
            tag = f"{arm}_r{r}_{ph}{k}"
            s = f"{src}_r{r}_{ph}{k}"
            if not (C1 / "states" / f"{s}.npz").exists():
                raise SystemExit(f"reuse source missing: {C1}/{s}")
            shutil.copy(C1 / "states" / f"{s}.npz",
                        out / "states" / f"{tag}.npz")
            shutil.copy(C1 / f"{s}.npy", out / f"{tag}.npy")
            if (C1 / f"{s}_dan.npy").exists():
                shutil.copy(C1 / f"{s}_dan.npy", out / f"{tag}_dan.npy")
    print(f"[acquire] {arm}: reused comp1/{src} x{sessions} sessions",
          flush=True)


def session(arm, r):
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    state = None
    for i in range(N_P1 + N_P2):
        ph = "p1" if i < N_P1 else "p2"
        k = i if ph == "p1" else i - N_P1
        tag = f"{arm}_r{r}_{ph}{k}"
        dst = out / f"{tag}.npy"
        nxt = out / "states" / f"{tag}.npz"
        if dst.exists() and nxt.exists():
            state = nxt
            continue
        run_trial(arm, ARMS[arm][ph], SEED0[arm] + 10 * r + i,
                  state, nxt, dst, tag)
        state = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma list from "
                         "comp_tr,comp_ctl,glob_tr,glob_ctl "
                         "(default: reuse glob + run comp)")
    ap.add_argument("--sessions", type=int, default=SESSIONS)
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if args.arms:
        arms = args.arms.split(",")
    else:
        arms = ["glob_tr", "glob_ctl", "comp_tr", "comp_ctl"]
    for arm in arms:
        if arm.startswith("glob"):
            reuse_glob(arm, args.sessions)
        else:
            for r in range(args.sessions):
                session(arm, r)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{a: ARMS[a] for a in arms}},
                 "n_p1": N_P1, "n_p2": N_P2, "dur_ms": T_END,
                 "sessions": args.sessions, "fs": 1000.0,
                 "lr": LR, "w0": W0, "kcm_gain": KCM_GAIN,
                 "mbmd_gain": MBMD_GAIN, "dan_gate": DAN_GATE,
                 "mbon_bias": MBON_BIAS, "regions": REGIONS,
                 "comp_gate": COMP_GATE,
                 "seed0": SEED0, "train_ms": [500, 2500],
                 "probe_ms": [4200, 4700],
                 "reuse": {"glob_tr": "comp1/global_blocked",
                           "glob_ctl": "comp1/global_control",
                           "note": "protocol-bitwise-identical "
                                   "(dangateA/C->dangateAB, "
                                   "seeds 1042+/1092+ +10r+k)"}})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
