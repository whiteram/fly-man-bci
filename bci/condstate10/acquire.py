"""bci/condstate10: long-chain asymptote -- does the locked chain's w
converge to the naive chain's?

condstate5 (5 links) found the rent is front-loaded, not compounding:
per-link ratio 0.78/0.77/0.98/1.0/1.0, the locked chain lags ~1 link
with equal asymptotic increments.  The extrapolation beyond 5 links
was flagged as untested.  This arc doubles the chains to 10 links
(same protocol: cs5t / cs5lt, w carried via --plastic-state-in/out,
one seed per link, decorrelated families 62-71 / 72-81, probe at the
end) and asks:

  1. Do the per-link ratios stay at ~1.0 for links 6-10?
  2. Does the w_A gap between the chains converge toward 0 (pure
     delay) or settle at a constant offset (a persistent lag)?
  3. Final probe MBON response: does the gap close in expression too?

Usage (conda ffbm, repo root):
    python bci/condstate10/acquire.py            # idempotent + readout
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
N = 10
CHAINS = {
    "naive":  {"links": [("cs5t", 62 + i, LINK_END) for i in range(N)],
               "probe": ("cs5p", 92)},
    "locked": {"links": [("cs5lt", 72 + i, LINK_END) for i in range(N)],
               "probe": ("cs5p", 93)},
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
    meta.update({"chains": CHAINS, "link_ms": LINK_END, "n_links": N,
                 "train_ms": list(TRAIN), "probe_ms": list(PROBE)})
    meta_p.write_text(json.dumps(meta, indent=1))
    # readout
    print("== condstate10: long-chain asymptote ==")
    res = {"curves": {}, "gap": {}}
    ws = {}
    for chain in CHAINS:
        ws[chain] = [W0]
        rows = []
        for i in range(1, N + 1):
            st = out / "states" / f"{chain}{i}.npz"
            if not st.exists():
                break
            d = np.load(st)
            w = d["w_KCM"]
            pt = d["pre_type"]
            wa = float(w[pt == "KCg-m"].mean())
            row = {"w_A": round(wa, 5), "dw": round(wa - ws[chain][-1], 5)}
            pop = out / f"{chain}{i}_pop.npz"
            if pop.exists():
                z = np.load(pop)
                row["lock_alpn"] = round(
                    float(z["ALPN"][1000:1900].mean()), 2)
                row["mod_ms"] = None
            dt_ = out / f"{chain}{i}_dan.npy"
            if dt_.exists():
                tr = np.load(dt_)
                inw = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
                row["mod_ms"] = round(float(tr[inw, 2].sum() * 0.5), 1)
            ws[chain].append(wa)
            rows.append(row)
        res["curves"][chain] = rows
        print(f"-- {chain} chain --")
        for i, r in enumerate(rows, 1):
            print(f"   link {i:2d}: {r}")
    if len(ws["naive"]) == len(ws["locked"]):
        gaps = [round(a - b, 5) for a, b in zip(ws["naive"],
                                                ws["locked"])]
        res["gap"] = {"w_A_gap_by_link": gaps,
                      "gap_first3_mean": round(float(np.mean(gaps[1:4])), 5),
                      "gap_last3_mean": round(float(np.mean(gaps[-3:])), 5)}
        print(f"-- w_A gap (naive-locked) by link: {gaps}")
        print(f"   gap mean links 1-3: {res['gap']['gap_first3_mean']}  "
              f"links 8-10: {res['gap']['gap_last3_mean']}")
        ninc = res["curves"]["naive"]
        linc = res["curves"]["locked"]
        rats = [round(l["dw"] / n["dw"], 3) for n, l in zip(ninc, linc)
                if n["dw"] and l["dw"]]
        res["per_link_ratio"] = rats
        print(f"-- per-link ratio locked/naive: {rats}")
    for chain in CHAINS:
        pop = out / f"{chain}_probe_pop.npz"
        if pop.exists():
            z = np.load(pop)
            mb = round(float(z["MBON"][PROBE[0]:PROBE[1]].mean()), 2)
            kc = round(float(z["Kenyon_Cell"][PROBE[0]:PROBE[1]].mean()), 2)
            res.setdefault("probe", {})[chain] = {"mbon": mb, "kc": kc}
            print(f"   [probe {chain}] mbon {mb} kc {kc}")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[condstate10] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
