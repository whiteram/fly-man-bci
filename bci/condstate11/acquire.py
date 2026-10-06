"""bci/condstate11: robustness spot-check -- new seed families for the
rent ratio and the micro-pool rearrangement.

Two headline results were single-family (seed 62 dominant):
  a. the quiet-lock rent ratio (~0.78; condstate1/2 used 62/63 only)
  b. the MBON09 cell-level rearrangement (L2 yields, L1/R1 light;
     condstate9, seed 62 only)
This arc runs the SAME cs2ctl/cs2lock pair under two NEW seed
families (2042, 2092 -- the seedchk1 convention) with both w-state
output and per-neuron recording (recording-only, no sim effect):
  ctl2042/lock2042, ctl2092/lock2092
Rent ratio: w_A growth lock/ctl per family.
Rearrangement: MBON09/30/11 per-cell counts across seeds 62/2042/2092.

Usage (conda ffbm, repo root):
    python bci/condstate11/acquire.py            # idempotent + readout
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
TYPES = "MBON03,MBON09,MBON11,MBON30,MBON31,DPM,APL"
TRAIN = (15000, 17000)
W0 = 0.03
ARMS = {
    "ctl2042":  {"chem": "cs2ctl",  "seed": 2042},
    "lock2042": {"chem": "cs2lock", "seed": 2042},
    "ctl2092":  {"chem": "cs2ctl",  "seed": 2092},
    "lock2092": {"chem": "cs2lock", "seed": 2092},
}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_neurons.npz"
    if dst.exists() and (out / "states" / f"{tag}.npz").exists() \
            and (out / f"{tag}_pop.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"],
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--pop-rate-neurons",
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", str(W0),
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", "20000.0", "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(out / "states" / f"{tag}.npz")]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}",
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.copy(trial / "_pop_rate_neurons.npz", dst)
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def main():
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = out / "meta.json"
    meta_p.write_text(json.dumps({"arms": ARMS}, indent=1))
    # readout
    print("== condstate11: robustness spot-check ==")
    res = {}
    for fam in ("2042", "2092"):
        row = {}
        for arm in (f"ctl{fam}", f"lock{fam}"):
            st = out / "states" / f"{arm}.npz"
            if not st.exists():
                continue
            d = np.load(st)
            w = d["w_KCM"]
            pt = d["pre_type"]
            row[arm] = round(float(w[pt == "KCg-m"].mean()), 5)
            pop = out / f"{arm}_pop.npz"
            if pop.exists():
                z = np.load(pop)
                row[f"{arm}_alpn"] = round(
                    float(z["ALPN"][1000:1900].mean()), 2)
        if f"ctl{fam}" in row and f"lock{fam}" in row:
            g_c = row[f"ctl{fam}"] - W0
            g_l = row[f"lock{fam}"] - W0
            row["ratio"] = round(g_l / g_c, 3) if g_c else float("nan")
        res[fam] = row
        print(f"   seed {fam}: {row}")
    # rearrangement across families (incl condstate9's seed 62)
    print("-- MBON09/11 per-cell counts across seed families --")
    res["micro"] = {}
    c9 = ROOT / "bci" / "condstate9" / "outputs"
    sources = {"62": (c9 / "ctln9_neurons.npz", c9 / "lockn9_neurons.npz")}
    for fam in ("2042", "2092"):
        sources[fam] = (out / f"ctl{fam}_neurons.npz",
                        out / f"lock{fam}_neurons.npz")
    for m in ("MBON09", "MBON11", "MBON30"):
        line = []
        res["micro"][m] = {}
        for fam, (pc, pl) in sources.items():
            if not (pc.exists() and pl.exists()):
                continue
            cc = np.asarray(np.load(pc)[m], dtype=float)
            cl = np.asarray(np.load(pl)[m], dtype=float)
            res["micro"][m][fam] = {"ctl": cc.tolist(),
                                    "lock": cl.tolist()}
            line.append(f"s{fam} {np.round(cc).astype(int).tolist()}"
                        f"->{np.round(cl).astype(int).tolist()}")
        print(f"   {m:8s} " + "  |  ".join(line))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[condstate11] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
