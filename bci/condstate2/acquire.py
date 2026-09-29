"""bci/condstate2: 受损可逆性直测 -- cure the lock, then train.

condstate1 showed conditioning INSIDE the quiet lock impairs learning
~22% (w-growth 0.78-0.80, gate-mediated), and the dstate14/15 map says
the impairment should ride on the state, not persist after it.  This
arc converts that in-map inference into a measurement: form the lock,
DISSOLVE it via the gain parameter axis (--gain-schedule x0.5), RESTORE
the gain (x2.0 compensating factor -- dstate15 lesson: factors compound
on the current value), then run the A+ training window AFTER the cure.

Timeline (t_end 20 s, dangate2 calibration, all arms time-matched):
  0.3-0.6 s   DA1 lock pulse (cs2lock / cs2lock300; ctl omits)
  8 s         CEN_C x0.5  -> g_unit 0.002 -> 0.001 (lock dissolves)
  12 s        CEN_C x2.0  -> back to 0.002 (dark, naive, no re-lock:
              dstate15 restore_only: no spontaneous re-lock w/o pulse)
  15-17 s     A+ training (KCg-m 1600 pA + DAN_err 800 pA)
  18-18.5 s   probe

Arms (6 trials, serial):
  ctl62  cs2ctl     seed 62   naive, time-matched baseline
  lock62 cs2lock    seed 62   lock persists through training (impairment ref)
  unlk62 cs2lock    seed 62   + gain schedule: cured BEFORE training
  ctl63  cs2ctl     seed 63   } replication set (300 pA lock dose,
  lock63 cs2lock300 seed 63   } condstate1's seed-63 working point)
  unlk63 cs2lock300 seed 63   }

Pre-registered readout: w_A growth ratio unlk/ctl vs lock/ctl.
Prediction from the dstate map: lock/ctl ~0.78 (replicates condstate1
at matched time), unlk/ctl ~1.0 (impairment reverses with the state).
Gate mechanism check: mod_ms (DAN-gate integral in the training window)
should recover with the cure.

Usage (conda ffbm, repo root):
    python bci/condstate2/acquire.py --arms unlk62   # pilot
    python bci/condstate2/acquire.py                 # all (idempotent)
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
T_END = 20000.0
W0 = 0.03
SCHED = "8000:CEN_C=0.5,12000:CEN_C=2.0"
TRAIN = (15000, 17000)
PROBE = (18000, 18500)
ARMS = {
    "ctl62":  {"chem": "cs2ctl",     "seed": 62, "sched": None},
    "lock62": {"chem": "cs2lock",    "seed": 62, "sched": None},
    "unlk62": {"chem": "cs2lock",    "seed": 62, "sched": SCHED},
    "ctl63":  {"chem": "cs2ctl",     "seed": 63, "sched": None},
    "lock63": {"chem": "cs2lock300", "seed": 63, "sched": None},
    "unlk63": {"chem": "cs2lock300", "seed": 63, "sched": SCHED},
}


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
    time.sleep(2.0)          # windows: cupy handle release before rmtree
    shutil.rmtree(trial, ignore_errors=True)


def readout(tags):
    print("== condstate2: cure-then-train (reversibility direct test) ==")
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
        row = {"w_A": round(wa, 5),
               "w_A_growth": round(wa - W0, 5)}
        if pop.exists():
            z = np.load(pop)
            def alpn(a, b):
                return round(float(z["ALPN"][a:b].mean()), 2)
            row["lock_1_1.9s"] = alpn(1000, 1900)        # formed?
            row["alpn_postdrop_8.5_11.5s"] = alpn(8500, 11500)
            row["alpn_prstrain_12.2_14.9s"] = alpn(12200, 14900)
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
        print(f"   {tag:>6}: {row}")
    for sset in ("62", "63"):
        ctl, lk, uk = res.get(f"ctl{sset}"), res.get(f"lock{sset}"), \
            res.get(f"unlk{sset}")
        if not (ctl and ctl["w_A_growth"]):
            continue
        for nm, r in (("lock", lk), ("unlk", uk)):
            if r and r["w_A_growth"]:
                res[f"{nm}{sset}_ratio"] = round(
                    r["w_A_growth"] / ctl["w_A_growth"], 3)
        if lk and uk:
            print(f"   [seed {sset}] lock/ctl "
                  f"{res.get(f'lock{sset}_ratio')}  unlk/ctl "
                  f"{res.get(f'unlk{sset}_ratio')}  "
                  f"mod_ms ctl/lock/unlk "
                  f"{ctl.get('mod_ms')}/{lk.get('mod_ms')}/"
                  f"{uk.get('mod_ms')}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate2] summary -> {HERE / 'outputs' / 'summary.json'}")


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
                 "probe_ms": list(PROBE), "sched": SCHED,
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)"})
    meta_p.write_text(json.dumps(meta, indent=1))
    readout(tags)


if __name__ == "__main__":
    main()
