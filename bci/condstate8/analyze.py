"""bci/condstate8 analysis: micro-pool drive decomposition + closed
forward model of the gate.

1. Micro-pool drive: for each GABA-subset micro-pool M, the lock's
   extra synaptic drive is dI_M = sum_T sign*W_{T->M}*(r_T^lock -
   r_T^ctl) (training-window type means).  Ranked contributions show
   who pushes MBON09 from 157 to 235 Hz.
2. Closed forward model on the dan-trace timeline (10-ms bins):
   r(t) ~ a0 + b*K(t) - c*V(t), where K(t) = sum over KC->DAN rows
   (type-aggregated weights x smoothed KC-type rates) and V(t) = sum
   over the exact 349 GABA rows (weights x smoothed MBON-type rates).
   Fit (a0, b, c) on the condstate8 pair, test on the condstate7 arms
   (res/hl/naive2r recorded under the same flag).

Usage (conda ffbm, repo root):
    python bci/condstate8/analyze.py
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
C7 = ROOT / "bci" / "condstate7" / "outputs"
MICROS = ["MBON09", "MBON30", "MBON11", "MBON03", "MBON31"]
TRAIN = (15000, 17000)
BIN_MS = 1.0          # _pop_rate.npz bins are 1 ms (help text is right)


def load_arm(pop_path, dan_path, t0, t1, bin_ms=1.0):
    """1-ms-binned type rates + dan r over the training window.

    _pop_rate.npz arrays are 1-ms bins (index == ms); the dan trace
    runs at 0.5 ms and is averaged onto the same grid (pairs of steps).
    """
    z = np.load(pop_path)
    tr = np.load(dan_path)
    b0, b1 = int(t0 / bin_ms), int(t1 / bin_ms)
    rates = {g: z[g][b0:b1] for g in z.files}
    inw = (tr[:, 0] >= t0) & (tr[:, 0] < t1)
    r1 = tr[inw, 1].reshape(-1, 2).mean(axis=1)
    return rates, r1


def smooth(x, tau_ms=150.0):
    a = 1.0 - np.exp(-BIN_MS / tau_ms)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += (v - acc) * a
        y[i] = acc
    return y


def main():
    from ffbm import data as fdata
    from ffbm import regions as freg
    ann = fdata.load_annotations()
    cls = ann.set_index("bodyId")["class"]
    tb = ann.set_index("bodyId")["type"]
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
    cp_cls = tab["body_pre"].map(comp2raw).map(cls).to_numpy()
    cq_cls = tab["body_post"].map(comp2raw).map(cls).to_numpy()
    cp_typ = tab["body_pre"].map(comp2raw).map(tb).to_numpy()
    cq_typ = tab["body_post"].map(comp2raw).map(tb).to_numpy()

    # exact 349 GABA rows (class-MBON -> class-DAN, sign<0)
    gaba = tab[(cp_cls == "MBON") & (cq_cls == "DAN") & (tab["sign"] < 0)]
    w_sub = gaba.assign(pre_t=cp_typ[(cp_cls == "MBON") & (cq_cls == "DAN")
                                     & (tab["sign"] < 0)]) \
        .groupby("pre_t")["weight"].sum()
    # KC->DAN rows (sign>0) aggregated per KC TYPE (class is
    # "Kenyon_Cell" for every KC -- the type column carries the family)
    kcm = (pd.Series(cp_typ).str.startswith("KC", na=False)) \
        & (cq_cls == "DAN")
    w_kc = tab[kcm.to_numpy()].assign(
        pre_t=pd.Series(cp_typ)[kcm.to_numpy()]) \
        .groupby("pre_t")["weight"].sum()
    print(f"GABA subset rows {len(gaba)}, W_total {w_sub.sum():.0f}; "
          f"KC->DAN rows {int(kcm.sum())}, W_total {w_kc.sum():.0f}")

    # ---- 1. micro-pool drive decomposition (lock8 - ctl8) ----
    rates_c, r_c = load_arm(OUT / "ctl8_pop.npz", OUT / "ctl8_dan.npy",
                            *TRAIN)
    rates_l, r_l = load_arm(OUT / "lock8_pop.npz", OUT / "lock8_dan.npy",
                            *TRAIN)
    cen = pd.read_parquet(ROOT / "bci" / "condstate6" / "outputs"
                          / "cen_typed.parquet")
    print("\n== micro-pool drive decomposition (training window) ==")
    decomp = {}
    for m in MICROS:
        rows = cen[cen["post_t"] == m]
        agg = rows.groupby(["pre_t", "sign"])["weight"].sum()
        parts = []
        for (pt, sg), w in agg.items():
            if pt in rates_l and pt in rates_c:
                d = float(rates_l[pt].mean() - rates_c[pt].mean())
                parts.append((pt, float(sg) * float(w) * d))
        parts.sort(key=lambda x: -abs(x[1]))
        decomp[m] = parts
        print(f"-- {m}: top drivers of dI (sign*W*dRate) --")
        for pt, di in parts[:6]:
            print(f"   {pt:12s} {di:+12.0f}")

    # ---- 2. closed forward model, 10-ms timeline ----
    def kv(rates):
        k = np.zeros_like(next(iter(rates.values())))
        for g, w in w_kc.items():
            if g in rates:
                k += float(w) * smooth(rates[g])
        v = np.zeros_like(k)
        for g, w in w_sub.items():
            if g in rates:
                v += float(w) * smooth(rates[g])
        return k, v

    kc_c, v_c = kv(rates_c)
    kc_l, v_l = kv(rates_l)
    X = np.column_stack([np.ones_like(kc_c), kc_c, -v_c])
    y = np.concatenate([r_c, r_l])
    A = np.vstack([X, np.column_stack([np.ones_like(kc_l), kc_l, -v_l])])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a0, b, c = coef
    pred_c = X @ coef
    pred_l = np.column_stack([np.ones_like(kc_l), kc_l, -v_l]) @ coef
    print(f"\n== forward fit on condstate8 pair ==\n"
          f"  a0 {a0:.2f}  b(KC) {b:.3e}  c(V) {c:.3e}")
    print(f"  ctl8: r {r_c.mean():.2f} pred {pred_c.mean():.2f} | "
          f"lock8: r {r_l.mean():.2f} pred {pred_l.mean():.2f}")

    print("\n== predictions on condstate7 arms ==")
    res = {"coef": {"a0": float(a0), "b": float(b), "c": float(c)},
           "fit": {"ctl8": float(r_c.mean()), "lock8": float(r_l.mean()),
                   "pred_ctl8": float(pred_c.mean()),
                   "pred_lock8": float(pred_l.mean())},
           "decomp": {m: [(p, round(d)) for p, d in v[:8]]
                      for m, v in decomp.items()}}
    for tag, t0, t1 in (("ctl62t", 15000, 17000), ("lock62t", 15000, 17000),
                        ("res62t", 17000, 19000), ("hl62t", 17000, 19000),
                        ("naive2r", 2000, 4000)):
        pp, dd = C7 / f"{tag}_pop.npz", C7 / f"{tag}_dan.npy"
        if not pp.exists():
            continue
        rates, r_b = load_arm(pp, dd, t0, t1)
        k, v = kv(rates)
        pred = a0 + b * k - c * v
        res[f"pred_{tag}"] = round(float(pred.mean()), 2)
        print(f"  {tag:8s} r {r_b.mean():6.2f}  pred {pred.mean():6.2f}")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[condstate8] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
