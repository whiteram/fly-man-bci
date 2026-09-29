"""bci/condstate5: multi-trial conditioning chains -- is the rent
per-trial constant, and is the memory readable from naive dynamics?

condstate1-4 established the single-trial picture (quiet lock rent
-21%, gate-mediated, zero debt).  Two remaining dynamics questions:

  1. COMPOUNDING: chain 5 A+ links (each re-entering the lock -- the
     plastic state carries across links, the dynamics reset, so each
     link re-forms the lock with one pulse) -- does each link's w
     increment stay at ~0.79x of naive (uniform scaling), or drift
     (earlier saturation / history dependence)?
  2. EXPRESSION: after the chains, one naive-dynamics probe link
     (odor, no lock, no US) -- is the trained-inside-the-lock memory
     read out as strongly as the trained-naive one?

Chains (dangate2 calibration, link t_end 4.5 s, states chained via
--plastic-state-in/out):
  naive  5 x cs5t (A+ [2,4] s)      -> probe cs5p
  locked 5 x cs5lt (lock + A+)      -> probe cs5p
Seeds: naive 62..66, locked 72..76 (decorrelated); probe 82/83.

Lock formation is verified PER LINK (ALPN [1,1.9] s) before any
increment is trusted (condstate1's knife-edge lesson).

Usage (conda ffbm, repo root):
    python bci/condstate5/acquire.py            # idempotent + readout
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
LINK_END = 4500.0
PROBE_END = 3000.0
W0 = 0.03
TRAIN = (2000, 4000)
PROBE = (2000, 2500)
N = 5
CHAINS = {
    "naive":  {"links": [("cs5t", 62 + i, LINK_END) for i in range(N)],
               "probe": ("cs5p", 82)},
    "locked": {"links": [("cs5lt", 72 + i, LINK_END) for i in range(N)],
               "probe": ("cs5p", 83)},
}


def run(tag, chem, seed, t_end, state_in, state_out):
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
           "--chem-input", chem, "--pop-rate", GROUPS,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", str(W0),
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(t_end), "--seed", str(seed),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(state_out)]
    if state_in is not None:
        cmd += ["--plastic-state-in", str(state_in)]
    print(f"[acquire] {tag}: {chem} seed {seed}"
          + (" +state_in" if state_in is not None else ""), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)


def main():
    out = HERE / "outputs"
    (out / "states").mkdir(parents=True, exist_ok=True)
    for chain, spec in CHAINS.items():
        state = None
        for i, (chem, seed, t_end) in enumerate(spec["links"], 1):
            tag = f"{chain}{i}"
            nxt = out / "states" / f"{tag}.npz"
            run(tag, chem, seed, t_end, state, nxt)
            state = nxt
        pchem, pseed = spec["probe"]
        run(f"{chain}_probe", pchem, pseed, PROBE_END, state,
            out / "states" / f"{chain}_probe.npz")
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"chains": CHAINS, "groups": GROUPS,
                 "link_ms": LINK_END, "probe_ms": PROBE_END,
                 "train_ms": list(TRAIN), "n_links": N,
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)"})
    meta_p.write_text(json.dumps(meta, indent=1))
    # readout
    print("== condstate5: chained conditioning inside/outside the lock ==")
    res = {"curves": {}, "increments": {}, "probe": {}}
    for chain in CHAINS:
        ws = [W0]
        rows = []
        for i in range(1, N + 1):
            tag = f"{chain}{i}"
            st = out / "states" / f"{tag}.npz"
            pop = out / f"{tag}_pop.npz"
            if not st.exists():
                break
            d = np.load(st)
            w = d["w_KCM"]
            pt = d["pre_type"]
            wa = float(w[pt == "KCg-m"].mean())
            row = {"w_A": round(wa, 5),
                   "dw": round(wa - ws[-1], 5)}
            if pop.exists():
                z = np.load(pop)
                row["lock_alpn"] = round(
                    float(z["ALPN"][1000:1900].mean()), 2)
                row["kc_train"] = round(
                    float(z["Kenyon_Cell"][TRAIN[0]:TRAIN[1]].mean()), 2)
            dt_ = out / f"{tag}_dan.npy"
            if dt_.exists():
                tr = np.load(dt_)
                inw = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
                row["mod_ms"] = round(float(tr[inw, 2].sum() * 0.5), 1)
            ws.append(wa)
            rows.append(row)
        res["curves"][chain] = rows
        print(f"-- {chain} chain --")
        for i, r in enumerate(rows, 1):
            print(f"   link {i}: {r}")
        res["increments"][chain] = [r["dw"] for r in rows]
    if res["increments"].get("naive") and res["increments"].get("locked") \
            and len(res["increments"]["naive"]) == len(
                res["increments"]["locked"]):
        print("-- per-link ratio locked/naive --")
        rats = []
        for i, (a, b) in enumerate(zip(res["increments"]["naive"],
                                       res["increments"]["locked"]), 1):
            r = b / a if a else float("nan")
            rats.append(round(r, 3))
            print(f"   link {i}: {b:+.5f}/{a:+.5f} -> {r:.3f}")
        res["per_link_ratio"] = rats
    for chain in CHAINS:
        pop = out / f"{chain}_probe_pop.npz"
        if pop.exists():
            z = np.load(pop)
            res["probe"][chain] = {
                "kc": round(float(z["Kenyon_Cell"][
                    PROBE[0]:PROBE[1]].mean()), 2),
                "mbon": round(float(z["MBON"][
                    PROBE[0]:PROBE[1]].mean()), 2)}
            print(f"   [probe {chain}] {res['probe'][chain]}")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[condstate5] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
