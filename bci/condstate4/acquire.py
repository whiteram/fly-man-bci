"""bci/condstate4: conditioning inside the HIGH LOCK -- the third point
on the rent-vs-depth curve, and the driver discrimination.

Two points so far: quiet lock (ALPN 20 / MBON 0 / KC 0) -> rent -20%;
latch residual (ALPN 99 / MBON 24 / KC 17) -> rent -32%.  The high
lock (da1_d1600 direct route, dstate7: ALPN ~100 / MBON ~12 / KC ~4,
dstate6/17 profile) splits the confound: AL level matched to the
residual (~100), central afterburn much smaller (MBON 12 vs 24, KC 4
vs 17).

  rent(hl) ~ -30%  -> AL self-sustained level drives gate suppression
  rent(hl) ~ -20%  -> central afterburn (MBON/KC) drives it

Arms (t_end 22 s, dangate2 calibration; ctl reuses condstate3's
cs3ctl, same [17,19] s training window):
  ctl62/63  cs3ctl   naive time-matched baseline
  hl62/63   cs4hl    DA1@1600 [0.3,0.6] s -> high lock -> train
  hlnt62/63 cs4hlnt  high lock, NO training (pure drift probe; the
                     high lock's KC is at 4 Hz, marginally active)

State signature check built into the readout (era rule): pre-train
window must match the high-lock profile before ratios are trusted.

Usage (conda ffbm, repo root):
    python bci/condstate4/acquire.py --arms hl62     # pilot
    python bci/condstate4/acquire.py                 # all (idempotent)
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
TRAIN = (17000, 19000)
PROBE = (20000, 20500)
PRE = (12000, 16500)
ARMS = {
    "ctl62":   {"chem": "cs3ctl",   "seed": 62},
    "hl62":    {"chem": "cs4hl",    "seed": 62},
    "hlnt62":  {"chem": "cs4hlnt",  "seed": 62},
    "ctl63":   {"chem": "cs3ctl",   "seed": 63},
    "hl63":    {"chem": "cs4hl",    "seed": 63},
    "hlnt63":  {"chem": "cs4hlnt",  "seed": 63},
}
# high-lock profile bands (dstate6/17: ALPN 100.6 / KC 4.0 / MBON 11.8)
BANDS = {"ALPN": (60, 140), "Kenyon_Cell": (0.5, 10), "MBON": (5, 25)}


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


def readout(tags):
    print("== condstate4: high-lock conditioning (rent curve 3rd point) ==")
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
        grown = float((np.abs(w - W0) > 0.005).mean())
        row = {"w_A": round(wa, 5), "w_A_growth": round(wa - W0, 5),
               "grown_frac": round(grown, 3)}
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
    for tag in [t for t in tags if t.startswith("hl")]:
        r = res.get(tag)
        if not r or "ALPN_pre" not in r:
            continue
        ok = all(BANDS[g][0] < r[f"{g}_pre"] < BANDS[g][1]
                 for g in BANDS)
        res[f"{tag}_sig_ok"] = ok
        print(f"   [sig] {tag}: ALPN {r['ALPN_pre']} KC "
              f"{r['Kenyon_Cell_pre']} MBON {r['MBON_pre']} -> "
              f"{'high-lock OK' if ok else 'SIGNATURE MISMATCH'}"
              f" (want ~100.6/4.0/11.8)")
    for sset in ("62", "63"):
        ctl, hl, nt = res.get(f"ctl{sset}"), res.get(f"hl{sset}"), \
            res.get(f"hlnt{sset}")
        if not (ctl and hl and nt and ctl["w_A_growth"]):
            continue
        debt = nt["w_A_growth"]
        trained = hl["w_A"] - nt["w_A"]
        res[f"debt{sset}"] = round(debt, 5)
        res[f"trained{sset}"] = round(trained, 5)
        res[f"trained_ratio{sset}"] = round(trained / ctl["w_A_growth"], 3)
        res[f"total_ratio{sset}"] = round(
            hl["w_A_growth"] / ctl["w_A_growth"], 3)
        print(f"   [seed {sset}] debt {debt:+.5f}  trained {trained:.5f} "
              f"(ratio {trained / ctl['w_A_growth']:.3f})  mod_ms "
              f"{ctl.get('mod_ms')}/{hl.get('mod_ms')}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[condstate4] summary -> {HERE / 'outputs' / 'summary.json'}")


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
                 "sig_ref": "dstate6/17 high-lock profile",
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)"})
    meta_p.write_text(json.dumps(meta, indent=1))
    readout(tags)


if __name__ == "__main__":
    main()
