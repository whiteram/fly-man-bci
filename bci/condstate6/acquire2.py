"""bci/condstate6 acquire2: decomposition runs after the US-only
discrimination (acquire.py):

  us-only verdict: quiet-lock ratio 0.996/1.001 (NO direct DAN-side
  suppression; usl63's lock failed to form -- knife edge -- making it a
  naive replicate) while the high lock keeps -17% with MBON at only
  7.5 Hz (r 110->40, near-multiplicative).  Anatomy: after the MBMD
  split, PAM/PPL1 cells have essentially NO inhibitory afferents left
  in CEN, so the only suppressor channel is MBON GABA feedback -- the
  deep-state component must be either MBON-baseline V or a conductance
  shunt from background afferents.

Arms here (seed 62, US window [15,17] s):
  usmb100/200/300  naive + --mbon-bias (calibration trio): match the
                   high lock's MBON baseline (~7.5 Hz) and compare
                   mod_ms to ush62's 1613 -- MBON-V vs shunt.
  ushinst          high lock + US-only with the full afferent-type
                   recording -- who actually fires onto the DAN cells?
  lockinst/ctlinst quiet-lock / naive FULL training with extended
                   groups -- identifies the state->MBON carrier pools
                   (DPM/CRE067/LAL155 excitatory; APL/LHMB1 inhibitory).

Usage (conda ffbm, repo root):
    python bci/condstate6/acquire2.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS_X = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN,CX"
TRAIN = (15000, 17000)
ARMS = {
    "usmb100": {"chem": "cs6us",  "bias": 100, "groups": "MBON,DAN"},
    "usmb200": {"chem": "cs6us",  "bias": 200, "groups": "MBON,DAN"},
    "usmb300": {"chem": "cs6us",  "bias": 300, "groups": "MBON,DAN"},
    "usmb115": {"chem": "cs6us",  "bias": 115, "groups": "MBON,DAN"},
    "usmb130": {"chem": "cs6us",  "bias": 130, "groups": "MBON,DAN"},
    "ushinst": {"chem": "cs6ush", "bias": None, "groups": GROUPS_X},
    "lockinst": {"chem": "cs2lock", "bias": None, "groups": GROUPS_X},
    "ctlinst":  {"chem": "cs2ctl",  "bias": None, "groups": GROUPS_X},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_pop.npz"
    if dst.exists() and (out / f"{tag}_dan.npy").exists():
        print(f"[acquire2] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"], "--pop-rate", spec["groups"],
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", "20000.0", "--seed", "62",
           "--out", str(trial), "--gpu"]
    if spec["bias"] is not None:
        cmd += [f"--mbon-bias={spec['bias']:g}"]
    print(f"[acquire2] {tag}: {spec['chem']} bias {spec['bias']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", dst)
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def readout():
    print("== condstate6/2: decomposition ==")
    res = {}
    for tag, spec in ARMS.items():
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        dt_ = HERE / "outputs" / f"{tag}_dan.npy"
        if not pop.exists():
            continue
        z = np.load(pop)
        row = {"mbon_train": round(
            float(z["MBON"][TRAIN[0]:TRAIN[1]].mean()), 2)}
        if dt_.exists():
            tr = np.load(dt_)
            inw = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
            row["mod_ms"] = round(float(tr[inw, 2].sum() * 0.5), 1)
            row["r_mean"] = round(float(tr[inw, 1].mean()), 2)
        for g in ("ALPN", "CX"):
            if g in z:
                row[f"{g}_pre"] = round(
                    float(z[g][12000:16500].mean()), 2)
        res[tag] = row
        print(f"   {tag:>8}: {row}")
    (HERE / "outputs" / "summary2.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate6/2] summary -> "
          f"{HERE / 'outputs' / 'summary2.json'}")


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta2.json"
    meta_p.write_text(json.dumps({"arms": {t: {k: (str(v) if k ==
                          "groups" else v) for k, v in s.items()}
                          for t, s in ARMS.items()}}, indent=1))
    readout()


if __name__ == "__main__":
    main()
