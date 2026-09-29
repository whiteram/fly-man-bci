"""bci/condstate1: does conditioning LEARN inside the quiet lock?

dstate1-18 closed the lock's mechanism; the application-side question:
the lock silences KC/MBON (0 Hz baselines) while leaving the AL engine
self-sustaining -- conditioning drives KC directly (chem 1600 pA) and
gates via MBON->DAN feedback, so does an A+ trial INSIDE the lock form
the association as well as in the naive network?  (BCI semantics: does
one consolidating odor exposure also impair learning?)

Arms (dangate2 calibration, single trial, seed 1042, t_end 6 s):
  ctl     dgA_late   A+ at [2,4] s, probe [5,5.5] s (no lock)
  locked  lockdgA    DA1@150 [0.3,0.6] s forms the lock first, then
          the SAME A+ and probe
Readouts: w_A (KCg-m mean, from --plastic-state-out), mod_ms (gate
integral in the training window), lock check (ALPN [1,1.9] s and the
pre-training [1.2,1.9] s window), probe scalp.

Usage (conda ffbm, repo root):
    python bci/condstate1/acquire.py
    python bci/condstate1/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN"
T_END = 6000.0
SEED = 62  # dstate7's proven lock-forming seed
ARMS = {"ctl": "dgA_late", "locked": "lockdgA"}
TRAIN = (2000, 4000)
PROBE = (5000, 5500)


def run(tag, chem):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / "states" / f"{tag}.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem, "--pop-rate", GROUPS,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(out / "states" / f"{tag}.npz")]
    print(f"[acquire] {tag}: {chem}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    (HERE / "outputs" / "states").mkdir(parents=True, exist_ok=True)
    for tag, chem in ARMS.items():
        run(tag, chem)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "groups": GROUPS, "dur_ms": T_END,
         "seed": SEED, "regions": REGIONS,
         "train_ms": list(TRAIN), "probe_ms": list(PROBE),
         "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                  "mbmd 178, gate 150,0.02,400)"}, indent=1))
    # readouts
    print("== condstate1: conditioning inside the quiet lock ==")
    res = {}
    for tag in ARMS:
        st = HERE / "outputs" / "states" / f"{tag}.npz"
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        if not st.exists():
            continue
        d = np.load(st)
        w = d["w_KCM"]
        pt = d["pre_type"]
        wa = float(w[pt == "KCg-m"].mean())
        row = {"w_A": round(wa, 5),
               "w_A_growth": round(wa - 0.03, 5)}
        if pop.exists():
            z = np.load(pop)
            row["lock_check_alpn_1_1.9s"] = round(
                float(z["ALPN"][1000:1900].mean()), 2)
            row["pre_train_alpn_1.2_1.9s"] = round(
                float(z["ALPN"][1200:1900].mean()), 2)
            row["kc_train"] = round(
                float(z["Kenyon_Cell"][TRAIN[0]:TRAIN[1]].mean()), 2)
            row["mbon_train"] = round(
                float(z["MBON"][TRAIN[0]:TRAIN[1]].mean()), 2)
            row["mbon_probe"] = round(
                float(z["MBON"][PROBE[0]:PROBE[1]].mean()), 2)
        dt_ = HERE / "outputs" / f"{tag}_dan.npy"
        if dt_.exists():
            tr = np.load(dt_)
            inw = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
            row["mod_ms"] = round(float(tr[inw, 2].sum() * 0.5), 1)
        res[tag] = row
        print(f"   {tag:>7}: {row}")
    if "ctl" in res and "locked" in res:
        r = (res["locked"]["w_A_growth"]
             / res["ctl"]["w_A_growth"]
             if res["ctl"]["w_A_growth"] else float("nan"))
        res["locked_learning_ratio"] = round(r, 3)
        print(f"[condstate1] locked/ctl w_A growth ratio: {r:.3f}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate1] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
