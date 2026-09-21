"""bci/comp2 analysis: independent associations + over-blocking.

Per arm: A-phase w_A (KCg-m) growth, B-phase w_B (KCab-m) growth,
grown_frac, per-trial mod integral (the effective gate), probe rms.
Headline: B-learning ratio = dw_B(trained) / dw_B(control) per gate
type -- comp ~1 (private gates, no crosstalk) vs glob <1 (aggregate V
over-blocks B).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
W0 = 0.03
TRAIN = (500, 2500)
PROBE = (4200, 4700)


def state_stats(npz):
    d = np.load(npz)
    w = d["w_KCM"]
    pt = d["pre_type"]
    return (float(w[pt == "KCg-m"].mean()), float(w[pt == "KCab-m"].mean()),
            float(w.mean()), float((np.abs(w - W0) > 0.005).mean()))


def mod_ms(tag):
    p = OUT / f"{tag}_dan.npy"
    if not p.exists():
        return np.nan
    tr = np.load(p)
    inwin = (tr[:, 0] >= TRAIN[0]) & (tr[:, 0] < TRAIN[1])
    return float(tr[inwin, 2].sum() * 0.5)


def probe_uv(tag):
    p = OUT / f"{tag}.npy"
    if not p.exists():
        return np.nan
    e = np.load(p)
    e = e.mean(axis=1) if e.ndim == 2 else e
    base = e[3000:4000]
    return float(np.sqrt(((e[PROBE[0]:PROBE[1]] - base.mean()) ** 2)
                         .mean()) * 1e6 * 1.7)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for arm in meta["arms"]:
        rows = []
        for i in range(meta["n_p1"] + meta["n_p2"]):
            ph = "p1" if i < meta["n_p1"] else "p2"
            tag = f"{arm}_{ph}{i if ph == 'p1' else i - meta['n_p1']}"
            st = OUT / "states" / f"{tag}.npz"
            if not st.exists():
                continue
            wa, wb, wall, gf = state_stats(st)
            rows.append({"tag": tag, "w_A": wa, "w_B": wb,
                         "w_all": wall, "grown_frac": gf,
                         "mod_ms": mod_ms(tag), "probe_uv": probe_uv(tag)})
        res[arm] = rows
        print(f"\n-- {arm} --  trial  w_A  w_B  w_all  grown%  "
              f"mod_ms  probe_uv")
        for r in rows:
            print(f"   {r['tag']:>16}  {r['w_A']:.4f} {r['w_B']:.4f} "
                  f"{r['w_all']:.4f} {r['grown_frac']:6.1%} "
                  f"{r['mod_ms']:8.1f} {r['probe_uv']:7.3f}")
    print("\n== over-blocking contrast (dw_B trained/ctl) ==")
    summary = {}
    for gate in ("comp", "glob"):
        tr, ct = res.get(f"{gate}_tr"), res.get(f"{gate}_ctl")
        if not tr or not ct:
            continue
        p1w_tr = tr[meta["n_p1"] - 1]["w_A"]
        dwB_tr = tr[-1]["w_B"] - W0
        dwB_ct = ct[-1]["w_B"] - W0
        ratio = dwB_tr / dwB_ct if dwB_ct > 0 else np.nan
        dA = p1w_tr - W0
        summary[gate] = {"dw_A_p1": dA, "dw_B_trained": dwB_tr,
                         "dw_B_control": dwB_ct, "B_ratio": ratio}
        print(f"  {gate}: dw_A(p1) {dA:.4f}  dw_B tr/ctl "
              f"{dwB_tr:.4f}/{dwB_ct:.4f} -> B ratio {ratio:.2f} "
              f"({'over-blocked' if ratio < 0.8 else 'independent'})")
    (OUT / "summary.json").write_text(json.dumps(
        {"per_arm": res, "contrast": summary}, indent=1))
    print(f"\n[comp2] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
