"""bci/condstate6: US-only pathway discrimination -- where does the
gate suppression live?

condstate4 localized the rent's driver to the AL self-sustained level;
the scatter across condstate1-5 arms REFUTES mean-MBON-rate mediation
(same mbon_train, wildly different mod_ms: naive2 59 Hz->550 vs
locked3 64 Hz->402 vs res 49->372).  Two candidate pathways remain:

  H-DAN   a lock-active pool X feeds DAN_err directly (X->DAN), and
          the suppression survives with the KC->MBON leg removed
  H-MBON  the suppression needs the KC->MBON leg (structured MBON
          recruitment / V feedback)

Discriminator: US-only arms -- DAN_err 800 pA alone in the training
window, NO KC drive.  In the QUIET lock MBON is fully silent (pre=0,
no KC drive -> stays 0), so V=0 in both arms; any mod_ms suppression
there is a direct DAN-side pathway.  The high-lock arm (MBON_pre ~6)
bounds the deep-state direct component.

Arms (t_end 20 s, US window [15,17] s, plastic flags on for the gate,
seed 62/63):
  us62    cs6us   naive baseline
  usl62   cs6usl  quiet lock + US-only
  usl63   cs6usl  replication
  ush62   cs6ush  high lock + US-only
  ush63   cs6ush  replication

Usage (conda ffbm, repo root):
    python bci/condstate6/acquire.py
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
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN"
T_END = 20000.0
W0 = 0.03
TRAIN = (15000, 17000)
ARMS = {
    "us62":  {"chem": "cs6us",  "seed": 62},
    "usl62": {"chem": "cs6usl", "seed": 62},
    "usl63": {"chem": "cs6usl", "seed": 63},
    "ush62": {"chem": "cs6ush", "seed": 62},
    "ush63": {"chem": "cs6ush", "seed": 63},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / f"{tag}_dan.npy").exists() \
            and (out / f"{tag}_pop.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"], "--pop-rate", GROUPS,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", str(W0),
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(T_END), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def readout():
    print("== condstate6: US-only pathway discrimination ==")
    res = {}
    for tag in ARMS:
        dt_ = HERE / "outputs" / f"{tag}_dan.npy"
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        if not dt_.exists():
            continue
        tr = np.load(dt_)
        inw = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
        r = tr[inw, 1]
        row = {"mod_ms": round(float(tr[inw, 2].sum() * 0.5), 1),
               "r_mean": round(float(r.mean()), 2),
               "r_peak": round(float(r.max()), 2),
               "r_onset": round(float(r[(tr[inw, 0] >= TRAIN[0])
                                        & (tr[inw, 0] < TRAIN[0] + 400)]
                                     .mean()), 2)}
        if pop.exists():
            z = np.load(pop)
            row["mbon_train"] = round(
                float(z["MBON"][TRAIN[0]:TRAIN[1]].mean()), 2)
            row["lock_alpn"] = round(
                float(z["ALPN"][1000:1900].mean()), 2)
        res[tag] = row
        print(f"   {tag:>5}: {row}")
    if "us62" in res:
        base = res["us62"]["mod_ms"]
        for tag in ("usl62", "usl63", "ush62", "ush63"):
            if tag in res:
                res[f"{tag}_ratio"] = round(res[tag]["mod_ms"] / base, 3)
                print(f"   [ratio] {tag}/us62 = "
                      f"{res[tag]['mod_ms'] / base:.3f}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate6] summary -> {HERE / 'outputs' / 'summary.json'}")


def main():
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": ARMS, "dur_ms": T_END, "train_ms": list(TRAIN),
                 "question": "US-only: does lock-state gate suppression "
                             "survive without the KC->MBON leg?"})
    meta_p.write_text(json.dumps(meta, indent=1))
    readout()


if __name__ == "__main__":
    main()
