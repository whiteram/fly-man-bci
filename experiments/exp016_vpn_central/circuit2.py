"""exp016 phase 2 -- bilateral + VPN -> central-brain circuit.

Extends the exp015 bilateral cascade with the dominant visual
downstream pathway (inventory phase 1):

    Mi/Tm + T4/T5 --(e_m2v, dataset signs)--> VPN (top-10 LC/LPLC
    classes, all cholinergic) --(e_v2c, ACh)--> central-brain targets
    (top receivers by synapse count, mostly PVLP/AOTU).

Positions for VPN/CB cells are SOMA locations (same um frame as the
synapse sites; verified site-soma distance 16-90 um on Mi1), injected
into the circuit's pre_pos/post_pos maps so the standard delay/kernel
machinery works unchanged. These layers are OPTIONAL for the pipeline:
they are built only because this dict carries vpn_ids/cb_ids.

Run from repository root:
    python experiments/exp016_vpn_central/circuit2.py    # report
"""

import importlib.util
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
from ffbm.params import cal as params_cal

from circuit import build_bilateral_circuit

OUT = Path(__file__).resolve().parent / "outputs"
W_MIN = 5
N_CB_TARGETS = 250
CAL = params_cal()


def build_vpn_circuit(bi=None):
    """Bilateral cascade + VPN + central targets (one circuit dict)."""
    if bi is None:
        bi = build_bilateral_circuit()
    ann = fdata.load_annotations()
    soma = fdata.neuron_positions(ann)
    sc = ann["superclass"].fillna("")

    # VPN selection: top classes by CB synapse mass (inventory phase 1)
    vpn_classes = set(CAL["VPN_CLASSES"])
    vpn = ann[(sc == "visual_projection") & ann["type"].isin(vpn_classes)
              & ann["bodyId"].isin(soma.keys())]
    vpn_ids = np.array(sorted(vpn["bodyId"].tolist()), dtype=np.int64)
    vpn_set = set(vpn_ids.tolist())

    # full connectome, VPN(selected) -> central brain, w >= 5
    t = pf.read_table(fdata.RAW / "connectome-weights.feather")
    pre = t.column("body_pre").combine_chunks().to_numpy(
        zero_copy_only=False)
    post = t.column("body_post").combine_chunks().to_numpy(
        zero_copy_only=False)
    wt = t.column("weight").combine_chunks().to_numpy(zero_copy_only=False)
    m = wt >= W_MIN
    pre, post, wt = pre[m], post[m], wt[m]
    m = np.isin(pre, list(vpn_set))
    vpn_pre, vpn_post, vpn_wt = pre[m], post[m], wt[m]
    # received-synapse ranking (factorized ids: bodyIds are large, a raw
    # bincount would allocate GBs)
    uniq_post, inv = np.unique(vpn_post, return_inverse=True)
    rec_syn = np.bincount(inv, weights=vpn_wt)
    order = np.argsort(rec_syn)[::-1]
    cb_list = []
    for i in order:
        if rec_syn[i] <= 0:
            break
        b = int(uniq_post[i])
        if b in soma:
            cb_list.append(b)
        if len(cb_list) >= N_CB_TARGETS:
            break
    cb_ids = np.array(sorted(cb_list), dtype=np.int64)
    cb_set = set(cb_list)
    types = ann.set_index("bodyId")["type"]
    cb_types = pd.Series([str(types.get(b, "?")) for b in cb_ids])
    print(f"VPN {len(vpn_ids):,} cells ({len(vpn_classes)} classes) -> "
          f"CB targets {len(cb_ids):,} cells; top CB types: "
          + ", ".join(f"{ty}x{n}" for ty, n in
                      cb_types.value_counts().head(6).items()))

    # MID/T45 -> VPN from the visual edge table (has dataset signs)
    ve = fdata.load_visual_edges()
    mid_set = set(bi["mid_ids"].tolist())
    t45_set = set(bi["t45_ids"].tolist())
    e_m2v = ve[ve["body_pre"].isin(mid_set | t45_set)
               & ve["body_post"].isin(vpn_set)]
    e_m2v = e_m2v[["body_pre", "body_post", "weight", "sign"]].copy()
    e_m2v = e_m2v[e_m2v["weight"] >= W_MIN].sort_values(
        ["body_pre", "body_post"]).reset_index(drop=True)

    # VPN -> CB (all cholinergic -> sign +1)
    m = np.isin(vpn_post, list(cb_set))
    e_v2c = pd.DataFrame({"body_pre": vpn_pre[m], "body_post": vpn_post[m],
                          "weight": vpn_wt[m].astype(np.float32),
                          "sign": np.ones(int(m.sum()), dtype=np.float32)})
    e_v2c = e_v2c.sort_values(["body_pre", "body_post"]).reset_index(
        drop=True)
    print(f"edges: MID,T45->VPN {len(e_m2v):,} pairs / "
          f"{int(e_m2v['weight'].sum()):,} syn | "
          f"VPN->CB {len(e_v2c):,} pairs / "
          f"{int(e_v2c['weight'].sum()):,} syn")

    # soma positions injected into the position maps (same um frame as
    # the synapse sites; VPN is both post and pre, CB is post). VPN/CB
    # ids are REMAPPED to a compact 1..n range: annotation bodyIds reach
    # 1.6e9 and the id-indexed lookup tables (_indices / bincount) would
    # allocate many GB. The new ids never index another layer's table,
    # so the compact space cannot collide.
    vmap = {int(old): int(new) for new, old in enumerate(vpn_ids, 1)}
    cmap = {int(old): int(new)
            for new, old in enumerate(cb_ids, len(vpn_ids) + 1)}
    pp = dict(bi["pre_pos"])
    qq = dict(bi["post_pos"])
    for old, new in vmap.items():
        pp[new] = soma[old]
        qq[new] = soma[old]
    for old, new in cmap.items():
        pp[new] = soma[old]
        qq[new] = soma[old]
    e_m2v["body_post"] = e_m2v["body_post"].map(vmap)
    e_v2c["body_pre"] = e_v2c["body_pre"].map(vmap)
    e_v2c["body_post"] = e_v2c["body_post"].map(cmap)

    circuit = dict(bi)
    circuit["pre_pos"] = pp
    circuit["post_pos"] = qq
    circuit["vpn_ids"] = np.array(sorted(vmap.values()), dtype=np.int64)
    circuit["cb_ids"] = np.array(sorted(cmap.values()), dtype=np.int64)
    circuit["vpn_type"] = np.array([str(types.get(b, "?"))
                                    for b in vpn_ids])
    circuit["e_m2v"] = e_m2v
    circuit["e_v2c"] = e_v2c
    return circuit


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    c = build_vpn_circuit()
    pp, qq = c["pre_pos"], c["post_pos"]
    pts = np.vstack([np.array([pp[b] for b in c["vpn_ids"]]),
                     np.array([qq[b] for b in c["cb_ids"]])])
    print(f"VPN/CB cloud span: "
          f"{np.round(pts.max(0) - pts.min(0), 1)} um, "
          f"r_max {np.linalg.norm(pts - pts.mean(0), axis=1).max():.1f} um")
    rep = {"n_vpn": int(len(c["vpn_ids"])), "n_cb": int(len(c["cb_ids"])),
           "n_pairs_m2v": int(len(c["e_m2v"])),
           "n_pairs_v2c": int(len(c["e_v2c"])),
           "syn_m2v": int(c["e_m2v"]["weight"].sum()),
           "syn_v2c": int(c["e_v2c"]["weight"].sum())}
    (OUT / "circuit2_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
