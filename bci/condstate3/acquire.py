"""bci/condstate3: conditioning inside the LATCH RESIDUAL state --
state rent vs structural debt.

condstate2 showed the quiet lock's 22% learning impairment is rent on
the state (bitwise recovery after the gain cure).  But the latch's
deep-hysteresis residual (dstate16/17: latch dropped to 0.002 keeps
ALPN ~98 / KC ~17 / MBON ~45) has KC ACTIVE -- the state itself may
write plastic changes.  This arc splits the residual's learning cost
into context (gate) and debt (w drift during the state):

  cs3ctl   naive, training [17,19] s            (time-matched baseline)
  cs3res   ignite latch -> x0.5 at 6 s -> residual -> train [17,19] s
  cs3resnt ignite latch -> x0.5 at 6 s -> residual, NO training
           (pure w-drift probe: structural debt of the state itself)

Recipe (dstate16 latch_dn2, same era): LAL*@L 600 pA [0.8,2.8] s,
no_cen_damp (default gain 0.004 full latch), then
--gain-schedule 6000:CEN_C=0.5 (0.004 -> 0.002 residual).  State
signature check is built into the readout (era-drift rule): the
pre-train window must match dstate17's residual profile before any
ratio is trusted.

Training-growth decomposition (pre-registered):
  debt    = w_A(resnt) - w0                     (state-written drift)
  trained = w_A(res) - w_A(resnt)               (learning in residual)
  ratio   = trained / (w_A(ctl) - w0)           (context cost)

Usage (conda ffbm, repo root):
    python bci/condstate3/acquire.py --arms res62    # pilot
    python bci/condstate3/acquire.py                 # all (idempotent)
"""
import argparse
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
T_END = 22000.0
W0 = 0.03
SCHED = "6000:CEN_C=0.5"
TRAIN = (17000, 19000)
PROBE = (20000, 20500)
PRE = (12000, 16500)          # residual signature window
ARMS = {
    "ctl62":   {"chem": "cs3ctl",   "seed": 62, "sched": None},
    "res62":   {"chem": "cs3res",   "seed": 62, "sched": SCHED},
    "resnt62": {"chem": "cs3resnt", "seed": 62, "sched": SCHED},
    "ctl63":   {"chem": "cs3ctl",   "seed": 63, "sched": None},
    "res63":   {"chem": "cs3res",   "seed": 63, "sched": SCHED},
    "resnt63": {"chem": "cs3resnt", "seed": 63, "sched": SCHED},
}
# dstate17 residual profile (ALPN/KC/MBON/DAN Hz) for the signature check
SIG = {"ALPN": 98.2, "Kenyon_Cell": 17.4, "MBON": 44.6, "DAN": 2.0}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / "states" / f"{tag}.npz").exists() \
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
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(out / "states" / f"{tag}.npz")]
    if spec["sched"]:
        cmd += ["--gain-schedule", spec["sched"]]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']}"
          + (f" sched {spec['sched']}" if spec["sched"] else ""),
          flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def readout(tags):
    print("== condstate3: residual-state conditioning (rent vs debt) ==")
    res = {}
    for tag in tags:
        st = HERE / "outputs" / "states" / f"{tag}.npz"
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        if not st.exists():
            continue
        d = np.load(st)
        w = d["w_KCM"]
        pt = d["pre_type"]
        wa = float(w[pt == "KCg-m"].mean())
        wall = float(w.mean())
        grown = float((np.abs(w - W0) > 0.005).mean())
        row = {"w_A": round(wa, 5), "w_A_growth": round(wa - W0, 5),
               "w_all": round(wall, 5), "grown_frac": round(grown, 3)}
        if pop.exists():
            z = np.load(pop)
            for g in GROUPS.split(","):
                row[f"{g}_pre"] = round(
                    float(z[g][PRE[0]:PRE[1]].mean()), 2)
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
    # signature check before ratios are trusted (era-drift rule)
    for tag in [t for t in tags if t.startswith("res")]:
        r = res.get(tag)
        if not r or "ALPN_pre" not in r:
            continue
        ok = (50 < r["ALPN_pre"] < 150 and 5 < r["Kenyon_Cell_pre"] < 40
              and 20 < r["MBON_pre"] < 80)
        res[f"{tag}_sig_ok"] = ok
        print(f"   [sig] {tag}: ALPN {r['ALPN_pre']} KC "
              f"{r['Kenyon_Cell_pre']} MBON {r['MBON_pre']} -> "
              f"{'residual OK' if ok else 'SIGNATURE MISMATCH'}"
              f" (want ~{SIG['ALPN']}/{SIG['Kenyon_Cell']}/{SIG['MBON']})")
    for sset in ("62", "63"):
        ctl, rs, nt = res.get(f"ctl{sset}"), res.get(f"res{sset}"), \
            res.get(f"resnt{sset}")
        if not (ctl and rs and nt):
            continue
        base = ctl["w_A_growth"]
        if base:
            debt = nt["w_A_growth"]
            trained = rs["w_A"] - nt["w_A"]
            res[f"debt{sset}"] = round(debt, 5)
            res[f"trained{sset}"] = round(trained, 5)
            res[f"trained_ratio{sset}"] = round(trained / base, 3)
            res[f"total_ratio{sset}"] = round(rs["w_A_growth"] / base, 3)
            print(f"   [seed {sset}] debt {debt:+.5f}  trained "
                  f"{trained:.5f} (ratio {trained / base:.3f})  total "
                  f"ratio {rs['w_A_growth'] / base:.3f}  mod_ms "
                  f"{ctl.get('mod_ms')}/{rs.get('mod_ms')}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate3] summary -> {HERE / 'outputs' / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma list from " + ",".join(ARMS))
    args = ap.parse_args()
    tags = list(ARMS) if not args.arms else args.arms.split(",")
    (HERE / "outputs" / "states").mkdir(parents=True, exist_ok=True)
    for tag in tags:
        run(tag, ARMS[tag])
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{t: ARMS[t] for t in tags}},
                 "groups": GROUPS, "dur_ms": T_END,
                 "regions": REGIONS, "train_ms": list(TRAIN),
                 "probe_ms": list(PROBE), "pre_ms": list(PRE),
                 "sched": SCHED, "sig_ref": "dstate17 residual profile",
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)"})
    meta_p.write_text(json.dumps(meta, indent=1))
    readout(tags)


if __name__ == "__main__":
    main()
