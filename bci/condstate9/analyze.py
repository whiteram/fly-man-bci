"""bci/condstate9 analysis: per-cell DAN_err clamping profile,
GABA-contact correlation, and micro-pool recruitment vs gain.

Per-cell counts come from _pop_rate_neurons.npz (whole-trial spike
counts; the training window dominates for DAN_err since baseline is
0-0.33 Hz).  DAN_err identities are reconstructed the same way
export_data builds them: class-DAN CEN cells (sorted raw ids mapped
to compact ids); the 154-cell subset = unique posts of the exact 349
MBMD GABA rows.

Usage (conda ffbm, repo root):
    python bci/condstate9/analyze.py
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
MICROS = ["MBON09", "MBON30", "MBON11", "MBON03", "MBON31"]


def rebuild_ids():
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
    cq_cls = tab["body_post"].map(comp2raw).map(cls).to_numpy()
    cp_cls = tab["body_pre"].map(comp2raw).map(cls).to_numpy()
    gaba = tab[(cp_cls == "MBON") & (cq_cls == "DAN") & (tab["sign"] < 0)]
    # per-post GABA contact weight over the 349 rows
    gw = gaba.groupby("body_post")["weight"].sum()
    # DAN group membership: sorted compact ids of class-DAN cells
    dan_raw = np.array(sorted(
        ann.loc[(ann["class"] == "DAN") & mask
                & ann["bodyId"].isin(soma), "bodyId"].tolist()),
        dtype=np.int64)
    rmap = dict(zip(raw.tolist(), cen_ids.tolist()))
    dan_compact = np.array(sorted(rmap[int(b)] for b in dan_raw))
    err_ids = np.array(sorted(int(b) for b in gaba["body_post"].unique()))
    pos = np.searchsorted(dan_compact, err_ids)
    assert (dan_compact[pos] == err_ids).all()
    return dan_compact, pos, gw.reindex(err_ids).fillna(0.0).to_numpy(), \
        len(gaba)


def counts(arm):
    z = np.load(OUT / f"{arm}_neurons.npz")
    return {g: np.asarray(z[g]) for g in z.files}


def qstats(x, label):
    qs = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
    print(f"  {label:22s} n={len(x):3d}  "
          + " ".join(f"p{p:g}={v:7.1f}" for p, v in
                     zip([0, 10, 25, 50, 75, 90, 100], qs)))
    return qs


def main():
    dan_compact, err_pos, gaba_w, n_gaba = rebuild_ids()
    print(f"exact GABA rows {n_gaba}; DAN_err cells {len(err_pos)}; "
          f"GABA contact w: "
          f"min {gaba_w.min():.0f} med {np.median(gaba_w):.0f} "
          f"max {gaba_w.max():.0f}")

    arms = {a: counts(a) for a in ("usn9", "usl9", "ctln9", "lockn9")}
    res = {}
    print("\n== DAN_err per-cell training-window counts ==")
    for a in arms:
        c = arms[a]["DAN"][err_pos]
        res[f"{a}_dan_err"] = {"mean": float(c.mean()),
                               "std": float(c.std()),
                               "cv": float(c.std() / max(c.mean(), 1e-9)),
                               "zero_frac": float((c < 2).mean())}
        qstats(c, a)

    # 1. uniform scaling vs recruitment split: lock/naive ratio spread
    un, ul = arms["usn9"]["DAN"][err_pos], arms["usl9"]["DAN"][err_pos]
    ratio = ul / np.maximum(un, 1.0)
    res["us_ratio"] = {"median": float(np.median(ratio)),
                       "p10": float(np.percentile(ratio, 10)),
                       "p90": float(np.percentile(ratio, 90)),
                       "clamped_below_half": float((ratio < 0.5).mean())}
    print(f"\n  usl9/usn9 per-cell ratio: median "
          f"{np.median(ratio):.3f}  p10 {np.percentile(ratio, 10):.3f}  "
          f"p90 {np.percentile(ratio, 90):.3f}  "
          f"frac<0.5 {(ratio < 0.5).mean():.2f}")

    # 2. clamping depth vs GABA contact weight (divisive prediction)
    depth = 1.0 - ul / np.maximum(un, 1.0)
    from scipy.stats import spearmanr
    rho, pv = spearmanr(gaba_w, depth)
    res["gaba_contact_spearman"] = {"rho": float(rho), "p": float(pv)}
    print(f"  clamp depth vs GABA contact w: Spearman rho {rho:+.3f} "
          f"(p {pv:.1e})  [divisive predicts rho>0]")

    # 3. training arms
    cn, cl = arms["ctln9"]["DAN"][err_pos], arms["lockn9"]["DAN"][err_pos]
    ratio_t = cl / np.maximum(cn, 1.0)
    print(f"  lockn9/ctln9 per-cell ratio: median "
          f"{np.median(ratio_t):.3f}  p10 {np.percentile(ratio_t, 10):.3f}"
          f"  p90 {np.percentile(ratio_t, 90):.3f}  "
          f"frac<0.5 {(ratio_t < 0.5).mean():.2f}")
    res["train_ratio"] = {"median": float(np.median(ratio_t)),
                          "clamped_below_half":
                          float((ratio_t < 0.5).mean())}

    # 4. micro-pools: recruitment vs gain
    print("\n== micro-pool per-cell counts (naive -> lock) ==")
    res["micro"] = {}
    for m in MICROS:
        if m not in arms["ctln9"]:
            continue
        cn_m = arms["ctln9"][m] * 20.0      # counts/sec approx (counts
        cl_m = arms["lockn9"][m] * 20.0     # are whole-trial /20 s)
        row = {"ctl": [float(x) for x in cn_m],
               "lock": [float(x) for x in cl_m]}
        rec = float((cn_m < 5).sum() and ((cn_m < 5) & (cl_m >= 5)).sum())
        row["recruited"] = rec
        res["micro"][m] = row
        print(f"  {m:8s} ctl {np.round(cn_m, 1)} -> lock "
              f"{np.round(cl_m, 1)}  recruited {rec:.0f}")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[condstate9] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
