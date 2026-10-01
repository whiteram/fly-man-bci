"""bci/condstate7 analysis: subset-V forward model + carrier table.

1. Reconstruct the EXACT MBMD GABA row set the way export_data does
   (class-MBON -> class-DAN, sign<0) and aggregate per pre-TYPE:
   W_type = sum of GABA row weights.  V_pred(t) = sum_type W_type *
   r_type(t) (cells within a type share the type mean rate).
2. Per arm: V_pred integral over the training window vs the observed
   DAN r (from the dan trace) and mod_ms.  If the subset model is
   right, V_pred ranks the arms' suppression correctly -- including
   the condstate5 naive2 anomaly (MBON mean 59 Hz, r only 13.8).
3. Carrier table: per-type training-window rate, lock/naive diff --
   who carries the quiet lock's +5 Hz MBON training response?
   (candidates from condstate6 anatomy: DPM, APL, MBON06/30/25-like/11/05.)

Usage (conda ffbm, repo root):
    python bci/condstate7/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

OUT = HERE / "outputs"
TRAIN = {"ctl62t": (15000, 17000), "lock62t": (15000, 17000),
         "res62t": (17000, 19000), "hl62t": (17000, 19000),
         "naive2r": (2000, 4000)}


def exact_gaba_weights():
    """Per-pre-TYPE aggregate of the exact MBMD GABA rows."""
    from ffbm import data as fdata
    from ffbm import regions as freg
    ann = fdata.load_annotations()
    cls = ann.set_index("bodyId")["class"]
    circuit, _ = freg.build_circuit(
        {"visual_bilateral": True, "ol_rest": True, "central_brain": True,
         "olfactory": True})
    mask = (ann["superclass"].fillna("").str.startswith("cb_")
            | (ann["superclass"] == "descending_neuron"))
    soma = fdata.neuron_positions(ann)
    raw = np.array(sorted(
        ann.loc[mask & ann["bodyId"].isin(soma), "bodyId"].tolist()),
        dtype=np.int64)
    cen_ids = np.asarray(circuit["extra_pops"]["CEN"]["ids"])
    comp2raw = pd.Series(raw, index=cen_ids)
    tab = circuit["extra_edges"]["CEN_C"]["table"]
    cp = tab["body_pre"].map(comp2raw)
    cq = tab["body_post"].map(comp2raw)
    m = (cp.map(cls).to_numpy() == "MBON") \
        & (cq.map(cls).to_numpy() == "DAN") & (tab["sign"] < 0)
    g = tab[m].copy()
    g["pre_t"] = cp[m].map(ann.set_index("bodyId")["type"]).to_numpy()
    print(f"exact MBMD GABA rows: {len(g)} (sim splits 349)")
    w = g.groupby("pre_t")["weight"].sum().sort_values(ascending=False)
    return w


def main():
    w_type = exact_gaba_weights()
    print("\nsubset W by type (top 12):")
    print(w_type.head(12).to_string())

    res = {}
    print("\n== per-arm forward check ==")
    print(f"{'arm':8s} {'r_mean':>7s} {'mod_ms':>7s} "
          f"{'Vsub_int':>9s} {'Vmean_int':>9s} {'mbon_mean':>9s}")
    rates = {}
    for tag, (t0, t1) in TRAIN.items():
        pop = OUT / f"{tag}_pop.npz"
        dt_ = OUT / f"{tag}_dan.npy"
        if not pop.exists() or not dt_.exists():
            print(f"  {tag}: missing, skip")
            continue
        z = np.load(pop)
        tr = np.load(dt_)
        inw = (tr[:, 0] >= t0) & (tr[:, 0] < t1)
        r_mean = float(tr[inw, 1].mean())
        mod_ms = float(tr[inw, 2].sum() * 0.5)
        vsub = vmean = 0.0
        for gname in z.files:
            r_g = z[gname][t0:t1]        # pop-rate bins are 0.5 ms? ->
            # arrays are recorded at DT_MS; integral uses the mean rate
            m_g = float(r_g.mean())
            if gname in w_type.index:
                vsub += float(w_type[gname]) * m_g
            if gname == "MBON":
                vmean = m_g
        res[tag] = {"r_mean": round(r_mean, 2), "mod_ms": round(mod_ms, 1),
                    "vsub_int": round(vsub, 1), "mbon_mean": round(vmean, 2)}
        rates[tag] = z
        print(f"{tag:8s} {r_mean:7.2f} {mod_ms:7.1f} "
              f"{vsub:9.1f} {'':9s} {vmean:9.2f}")

    if "ctl62t" in res and "lock62t" in res:
        print("\n== carrier table: training-window rate lock - naive "
              "(Hz/neuron) ==")
        zc, zl = rates["ctl62t"], rates["lock62t"]
        t0, t1 = TRAIN["ctl62t"]
        rows = []
        for g in zl.files:
            if g not in zc:
                continue
            d = float(zl[g][t0:t1].mean()) - float(zc[g][t0:t1].mean())
            rows.append((g, d, float(zl[g][t0:t1].mean())))
        for g, d, lvl in sorted(rows, key=lambda x: -abs(x[1]))[:15]:
            wmark = " *subset*" if g in w_type.index else ""
            print(f"  {g:12s} {d:+7.2f}  (lock {lvl:6.2f}){wmark}")

    if "naive2r" in res:
        z = rates["naive2r"]
        t0, t1 = TRAIN["naive2r"]
        print("\n== naive2r subset anatomy (MBON mean "
              f"{res['naive2r']['mbon_mean']:.1f} Hz) ==")
        for g in w_type.index:
            if g in z.files:
                r_g = float(z[g][t0:t1].mean())
                if r_g > 0.5:
                    print(f"  {g:12s} {r_g:6.2f} Hz  "
                          f"W {w_type[g]:6.0f}")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[condstate7] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
