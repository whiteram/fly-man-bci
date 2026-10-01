"""bci/condstate7: GABA-subset direct measurement -- is V the weighted
rate of the MBON GABA subset?

condstate6 established: the gate is a steep MBON-V clamp and V is
governed by GABA-subset firing, not the population mean (uniform 32 Hz
kills the DAN while training means of 101 Hz leave r~9).  The new
--pop-rate-type flag (type-level recording, class-NaN pools like
DPM/APL now addressable) makes the subset directly measurable.

Arms (seed 62, full training KC+US, type recording):
  ctl62t   cs2ctl   naive, train [15,17] s
  lock62t  cs2lock  quiet lock + full training (the +5 Hz MBON
                    training-response carrier hunt)
  res62t   cs3res   latch residual, train [17,19] s
  hl62t    cs4hl    high lock, train [17,19] s
  naive2r  cs5t + --plastic-state-in(condstate5 states/naive1.npz)
                    the condstate5 naive2 anomaly (MBON mean 59 Hz but
                    r only 13.8) re-run under the subset microscope

Analysis (analyze.py): reconstruct the EXACT MBMD GABA row set
(class-MBON -> class-DAN, 349 rows), predict V(t) = sum w_ij r_i(t)
from the recorded per-type rates, compare across arms; per-type
lock-vs-naive diff identifies the state->MBON carrier.

Usage (conda ffbm, repo root):
    python bci/condstate7/acquire.py
    python bci/condstate7/analyze.py
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
GROUPS = "ALPN,MBON,DAN,Kenyon_Cell"
TYPES = ("MBON01,MBON02,MBON03,MBON04,MBON05,MBON06,MBON07,MBON09,"
         "MBON10,MBON11,MBON20,MBON25,MBON30,MBON31,MBON32,MBON34,"
         "MBON15-like,MBON25-like,DPM,APL,LHMB1,CRE067,LAL155,"
         "PPL101,PPL103,PPL106,PAM05,PAM06,PAM07,PAM08,PAM09,PAM10,"
         "PAM11,PAM12,KCg-m")
ARMS = {
    "ctl62t":  {"chem": "cs2ctl",  "seed": 62, "t_end": 20000.0,
                "state_in": None},
    "lock62t": {"chem": "cs2lock", "seed": 62, "t_end": 20000.0,
                "state_in": None},
    "res62t":  {"chem": "cs3res",  "seed": 62, "t_end": 22000.0,
                "state_in": None, "sched": "6000:CEN_C=0.5"},
    "hl62t":   {"chem": "cs4hl",   "seed": 62, "t_end": 22000.0,
                "state_in": None},
    "naive2r": {"chem": "cs5t",    "seed": 63, "t_end": 4500.0,
                "state_in": ROOT / "bci" / "condstate5" / "outputs"
                            / "states" / "naive1.npz"},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_pop.npz"
    if dst.exists() and (out / f"{tag}_dan.npy").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"],
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(spec["t_end"]), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu"]
    if spec.get("sched"):
        cmd += ["--gain-schedule", spec["sched"]]
    if spec["state_in"] is not None:
        cmd += ["--plastic-state-in", str(spec["state_in"])]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", dst)
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {t: {k: str(v) for k, v in s.items()}
                          for t, s in ARMS.items()},
                 "types": TYPES, "groups": GROUPS})
    meta_p.write_text(json.dumps(meta, indent=1))
    print("[condstate7] acquire done; run analyze.py")


if __name__ == "__main__":
    main()
