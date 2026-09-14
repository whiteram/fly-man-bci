"""exp016 phase 1 -- VPN -> central-brain inventory (pure data query).

Who carries the visual signal into the central brain, how much, and to
whom? Feeds the phase-2 wiring decision (which VPN classes + which
central targets enter the scalp-forward circuit).

Pathway model: our cascade layers (Mi/Tm, T4/T5) -> VPNs (visual
projection neurons: LC/LPLC/MeVP/...) -> central-brain targets.

Outputs outputs/inventory.json:
  - per VPN class: cells, synapses received FROM our circuit layers,
    synapses sent INTO central brain (w>=5), top target types, soma span
  - per top target type: cells, received synapses, soma span
  - neurotransmitter per class (for sign mapping)

Run from repository root:
    python experiments/exp016_vpn_central/inventory.py   (~4 min)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

from ffbm import data as fdata

OUT = Path(__file__).resolve().parent / "outputs"
W_MIN = 5


def span_um(pos_map, ids):
    pts = np.array([pos_map[b] for b in ids if b in pos_map])
    if len(pts) < 2:
        return None
    return round(float(np.linalg.norm(pts.max(0) - pts.min(0))), 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = fdata.load_annotations()
    soma = fdata.neuron_positions(ann)
    types = ann.set_index("bodyId")["type"]

    # central brain = cb_* superclasses; VPN = visual_projection
    sc = ann["superclass"].fillna("")
    cb_ids = set(ann.loc[sc.str.startswith("cb_"), "bodyId"].tolist())
    vpn = ann[sc == "visual_projection"]
    vpn_ids = set(vpn["bodyId"].tolist())
    print(f"VPN cells {len(vpn):,} | central-brain cells {len(cb_ids):,}")

    # our cascade's feed into VPNs: the circuit dict's edge tables are
    # pre-filtered INTO the five layers, so query the raw visual edges
    from circuit import build_bilateral_circuit
    bi = build_bilateral_circuit()
    ve = fdata.load_visual_edges()
    mid_set = set(bi["mid_ids"].tolist())
    t45_set = set(bi["t45_ids"].tolist())
    s = ve[ve["body_pre"].isin(mid_set | t45_set)
           & ve["body_post"].isin(vpn_ids)].copy()
    s["layer"] = np.where(s["body_pre"].isin(t45_set), "T45", "MID")
    feed_syn = {k: int(v) for k, v in s.groupby("layer")["weight"].sum()
                .items()}
    print(f"circuit -> VPN synapses: {feed_syn} "
          f"(from {s['body_pre'].nunique():,} distinct pre cells)")

    # full connectome: VPN -> central brain, w>=5
    t = pf.read_table(fdata.RAW / "connectome-weights.feather")
    pre = t.column("body_pre").combine_chunks().to_numpy(zero_copy_only=False)
    post = t.column("body_post").combine_chunks().to_numpy(zero_copy_only=False)
    wt = t.column("weight").combine_chunks().to_numpy(zero_copy_only=False)
    m = wt >= W_MIN
    pre, post, wt = pre[m], post[m], wt[m]
    m = np.isin(pre, list(vpn_ids)) & np.isin(post, list(cb_ids))
    pre, post, wt = pre[m], post[m], wt[m]
    print(f"VPN -> CB edges (w>={W_MIN}): {len(pre):,} pairs, "
          f"{int(wt.sum()):,} synapses")

    pre_t = pd.Series(pre).map(types).fillna("?")
    post_t = pd.Series(post).map(types).fillna("?")
    df = pd.DataFrame({"pre_t": pre_t.values, "post_t": post_t.values,
                       "w": wt})

    # per VPN-class rollup (top 20 by synapses into CB)
    by_pre = df.groupby("pre_t").agg(
        pairs=("w", "size"), syn=("w", "sum")).sort_values(
        "syn", ascending=False)
    by_pre["cells"] = [int((vpn["type"] == t_).sum()) for t_ in by_pre.index]

    # neurotransmitters per class
    nt = pd.read_feather(fdata.RAW / "body-neurotransmitters.feather")
    vpn_nt_all = nt.set_index("body")["consensus_nt"]

    rows = []
    for t_ in by_pre.head(20).index:
        sub = df[df["pre_t"] == t_]
        tops = sub.groupby("post_t")["w"].sum().sort_values(
            ascending=False).head(5)
        ids_t = vpn.loc[vpn["type"] == t_, "bodyId"].tolist()
        nts = vpn_nt_all.reindex(ids_t).dropna().value_counts()
        rows.append({
            "vpn_class": t_, "cells": int(by_pre.loc[t_, "cells"]),
            "cb_synapses": int(by_pre.loc[t_, "syn"]),
            "cb_pairs": int(by_pre.loc[t_, "pairs"]),
            "top_targets": {k: int(v) for k, v in tops.items()},
            "nt": nts.index[0] if len(nts) else None,
            "soma_span_um": span_um(soma, ids_t),
        })

    # target rollup (top 20)
    by_post = df.groupby("post_t").agg(
        pairs=("w", "size"), syn=("w", "sum")).sort_values(
        "syn", ascending=False)
    by_post["cells"] = [int((ann["type"] == t_).sum())
                        for t_ in by_post.index]
    tgt_rows = []
    for t_ in by_post.head(20).index:
        ids_t = ann.loc[ann["type"] == t_, "bodyId"].tolist()
        tgt_rows.append({
            "target_type": t_, "cells": int(by_post.loc[t_, "cells"]),
            "syn_from_vpn": int(by_post.loc[t_, "syn"]),
            "soma_span_um": span_um(soma, ids_t),
        })

    rep = {
        "filter": f"VPN(superclass=visual_projection) -> CB(cb_*), w>={W_MIN}",
        "circuit_feed_into_vpn_synapses": feed_syn,
        "vpn_classes": rows,
        "targets": tgt_rows,
    }
    (OUT / "inventory.json").write_text(json.dumps(rep, indent=1))
    print("\ntop VPN classes (into CB):")
    for r in rows[:10]:
        print(f"  {r['vpn_class']:<12} cells {r['cells']:>5} "
              f"syn {r['cb_synapses']:>8,} nt={r['nt']} "
              f"span {r['soma_span_um']} um")
    print("\ntop targets:")
    for r in tgt_rows[:10]:
        print(f"  {r['target_type']:<12} cells {r['cells']:>5} "
              f"syn {r['syn_from_vpn']:>8,} span {r['soma_span_um']} um")
    print(f"\nwritten {OUT / 'inventory.json'}")


if __name__ == "__main__":
    main()
