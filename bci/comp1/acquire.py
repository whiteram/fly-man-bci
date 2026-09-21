"""bci/comp1: per-compartment error gates (the fly's MB compartments).

dangate2's circuit error loop uses ONE global scalar mod for all
33,398 plastic KC->MBON edges; the real fly's MBON->DAN feedback is
compartment-specific (the code comment in export_data has said so
since bci/dangate).  This study adds --plastic-comp-reward: a
compartment is the MBON set feeding back (MBMD GABA rows) to ONE DAN
type; the reward channel and the gate monitor see only those cells
and mod becomes a per-edge vector, zero outside the compartment.

Pilot compartment: PAM07 (14 DAN cells <- 14 MBONs, 13.7% of the
plastic edges).  Arms, identical dangate2 protocol and calibration
(lr -2e-7 potentiation, w0 0.03, kcm 0.015, mbmd 178, gate
150,0.02,400, bias -300, seeds 1042+/1092+):
  global  dangateA/AB/C chems (US on DAN_err, 154 cells) -- the
          dangate2 reproduction baseline (kernel now takes a per-edge
          mod vector; the scalar path broadcasts, arithmetic unchanged)
  comp    compA/AB/C chems (US on PAM07* only, per-cell matched)
          + --plastic-comp-reward PAM07*

Readouts (analyze.py): per-arm blocking contrast (w_B final blocked
vs control; mod_ms suppression), plus the architectural restriction:
fraction of edges grown per trial (comp should cap at ~13.7%, global
spreads over every eligible edge).

Usage (conda ffbm, repo root):
    python bci/comp1/acquire.py --pilot     # comp arm, 1 session
    python bci/comp1/acquire.py             # 2 arms x 2 conds x 2
    python bci/comp1/analyze.py
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
N_P1, N_P2 = 4, 6
SEED0 = {"blocked": 1042, "control": 1092}
COMP_TYPE = "PAM07*"
ARMS = {
    "global": {"chems": {"A": "dangateA", "AB": "dangateAB",
                         "C": "dangateC"},
               "comp": None,
               "gate": DAN_GATE},
    # PAM07 cells sit at r0 ~ 0 and reach only ~6.3 Hz under the
    # per-cell-matched US (the mixed DAN_err pool reaches ~51 Hz), so
    # the gate gain is retuned 0.02 -> 0.16 to give the same mod
    # dynamic range (untrained ~1.0); measured on the pilot trace
    "comp": {"chems": {"A": "compA", "AB": "compAB", "C": "compC"},
             "comp": COMP_TYPE,
             "gate": "150,0.16,400"},
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
           "--plastic-dan-gate", ARMS[arm]["gate"],
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


def session(arm, cond, r, n_p1, n_p2, prefix=""):
    out = HERE / "outputs"
    st = out / "states"
    st.mkdir(parents=True, exist_ok=True)
    chems = ARMS[arm]["chems"]
    kfile = out / f"{prefix}{arm}_{cond}_r{r}.json"
    done = json.loads(kfile.read_text())["next"] if kfile.exists() else 0
    state = None
    plan = ([("p1", k, chems["A"] if cond == "blocked" else chems["C"])
             for k in range(n_p1)]
            + [("p2", k, chems["AB"]) for k in range(n_p2)])
    for i, (phase, k, chem) in enumerate(plan):
        tag = f"{prefix}{arm}_{cond}_r{r}_{phase}{k}"
        dst = out / f"{tag}.npy"
        nxt = st / f"{tag}.npz"
        if i < done:
            state = nxt
            continue
        run_trial(arm, chem, SEED0[cond] + 10 * r + k, state, nxt,
                  dst, tag)
        kfile.write_text(json.dumps({"next": i + 1}), encoding="utf-8")
        state = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--arms", default=None,
                    help="comma list from global,comp (default both)")
    args = ap.parse_args()
    arms = list(ARMS) if not args.arms else args.arms.split(",")
    n_p1, n_p2 = (2, 3) if args.pilot else (N_P1, N_P2)
    prefix = "pilot_" if args.pilot else ""
    for r in range(1 if args.pilot else 2):
        for arm in arms:
            for cond in (["blocked"] if args.pilot
                         else ["blocked", "control"]):
                session(arm, cond, r, n_p1, n_p2, prefix)
    (HERE / "outputs" / f"{prefix}meta.json").write_text(json.dumps(
        {"arms": arms, "conditions": ["blocked", "control"],
         "n_p1": n_p1, "n_p2": n_p2,
         "sessions": 1 if args.pilot else 2, "fs": 1000.0,
         "dur_ms": T_END, "lr": LR, "w0": W0,
         "kcm_gain": KCM_GAIN, "mbmd_gain": MBMD_GAIN,
         "dan_gate": {a: ARMS[a]["gate"] for a in arms},
         "mbon_bias": MBON_BIAS,
         "comp_type": COMP_TYPE, "regions": REGIONS,
         "chems": {a: ARMS[a]["chems"] for a in arms},
         "seed_map": "blocked 1042+, control 1092+ (+10r+k)"}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
