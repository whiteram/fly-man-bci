"""exp020 probe: who DIRECTLY drives descending neurons (DN)?

Question (user): can the connectome supply a direct drive onto DN, so
motor-imagery persistence can be routed through a real path instead of
the weak PRO->CEN edge group?

Analysis on the raw connectome (weight >= 5, same convention as the
builders):
  1. DN population: superclass descending_neuron -- count, rootSide,
     type families (DNa/DNb/...)
  2. DN direct inputs ranked by presynaptic superclass and by type
     (pairs + synapse counts)
  3. PRO -> DN direct synapses (expected ~0)
  4. two-hop PRO -> X -> DN: rank intermediates X by synapse count,
     flag whether X lives inside the simulated VNC/CEN regions
  5. the VNC->CEN leg in isolation: ascending-class neurons' synapses
     onto DN (already inside the ASC_R edge group of the sim)

Run from repository root:
    python experiments/exp020_proprioception/dn_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata

OUT = Path(__file__).resolve().parent / "outputs"
W_MIN = 5

ann = fdata.load_annotations()
ann = ann.set_index("bodyId", drop=False)
sup = ann["superclass"].fillna("")
typ = ann["type"].fillna("?")
side = ann["rootSide"].fillna("?")

# ---- 1. DN population --------------------------------------------------
dn_mask = sup == "descending_neuron"
dn_ids = np.array(sorted(ann.index[dn_mask].astype(int)), dtype=np.int64)
dn_types = typ[dn_ids].value_counts().head(12)
print(f"DN cells: {len(dn_ids):,}; rootSide:",
      dict(side[dn_ids].value_counts()))
print("top DN types:", dict(dn_types))

# ---- weights ------------------------------------------------------------
t = pf.read_table(fdata.RAW / "connectome-weights.feather")
pre = t.column("body_pre").combine_chunks().to_numpy(zero_copy_only=False)
post = t.column("body_post").combine_chunks().to_numpy(zero_copy_only=False)
wt = t.column("weight").combine_chunks().to_numpy(zero_copy_only=False)
m = wt >= W_MIN
pre, post, wt = pre[m], post[m], wt[m]

body_ids_index = np.array(sorted(ann.index.astype(int)), dtype=np.int64)
pos_of = {int(b): i for i, b in enumerate(body_ids_index)}
sup_arr = sup.reindex(body_ids_index).to_numpy()
typ_arr = typ.reindex(body_ids_index).to_numpy()


def top_table(pre_rows, syn_rows, label, topn=15):
    """aggregate an incoming (pre, syn) stream by pre type/superclass."""
    idx = np.fromiter((pos_of.get(int(b), -1) for b in pre_rows),
                      dtype=np.int64, count=len(pre_rows))
    ok = idx >= 0
    df = pd.DataFrame({
        "superclass": sup_arr[idx[ok]],
        "type": typ_arr[idx[ok]],
        "syn": syn_rows[ok],
    })
    by_sc = df.groupby("superclass")["syn"].agg(["sum", "count"]) \
        .sort_values("sum", ascending=False)
    by_ty = df.groupby("type")["syn"].agg(["sum", "count"]) \
        .sort_values("sum", ascending=False)
    print(f"\n== {label}: {ok.sum():,} pairs / "
          f"{df['syn'].sum():,.0f} syn ==")
    print("-- by superclass --")
    print(by_sc.head(10).to_string())
    print("-- by type --")
    print(by_ty.head(topn).to_string())
    return by_sc, by_ty


# ---- 2. direct inputs to DN --------------------------------------------
m_dn = np.isin(post, dn_ids)
by_sc, by_ty = top_table(pre[m_dn], wt[m_dn], "direct inputs -> DN")

# ---- 3. PRO -> DN direct -----------------------------------------------
cls = ann["class"].fillna("")
sub = ann["subclass"].fillna("")
pro_mask = (cls == "mechanosensory_proprioceptive") \
    | (sub == "chordotonal organ") | (sub == "campaniform sensilla") \
    | typ.str.contains("hair plate", case=False)
pro_ids = np.array(sorted(ann.index[pro_mask].astype(int)),
                   dtype=np.int64)
m_pro_dn = np.isin(pre, pro_ids) & np.isin(post, dn_ids)
print(f"\nPRO cells: {len(pro_ids):,}")
print(f"PRO->DN direct: {m_pro_dn.sum()} pairs / "
      f"{wt[m_pro_dn].sum():.0f} syn")

# ---- 4. two-hop PRO -> X -> DN -----------------------------------------
m_pro_out = np.isin(pre, pro_ids)
x_ids = np.unique(post[m_pro_out])
x_ids = x_ids[~np.isin(x_ids, pro_ids)]
m_x_dn = np.isin(pre, x_ids) & np.isin(post, dn_ids)
x_pre = pre[m_x_dn]
if len(x_pre):
    idx_x = np.array([pos_of.get(int(b), -1) for b in x_pre])
    df2 = pd.DataFrame({
        "type": typ_arr[idx_x], "superclass": sup_arr[idx_x],
        "syn": wt[m_x_dn]})
    hop2 = df2.groupby(["superclass", "type"])["syn"] \
        .agg(["sum", "count"]).sort_values("sum", ascending=False)
    print(f"\ntwo-hop PRO->X->DN: {m_x_dn.sum():,} pairs / "
          f"{df2['syn'].sum():,.0f} syn; intermediates "
          f"{len(x_ids):,}")
    print(hop2.head(20).to_string())
else:
    print("\ntwo-hop PRO->X->DN: none at w>=5")
    hop2 = None

# ---- 5. ascending leg (VNC region -> DN) -------------------------------
vnc_classes = ["ascending_neuron", "sensory_ascending",
               "sensory_ascending_tbc", "vnc_tbc", "vnc_sensory_tbc",
               "efferent_ascending"]
mask_asc = sup.isin(vnc_classes)
asc_ids = np.array(sorted(ann.index[mask_asc].astype(int)),
                   dtype=np.int64)
m_asc_dn = np.isin(pre, asc_ids) & np.isin(post, dn_ids)
by_sc_a, by_ty_a = top_table(pre[m_asc_dn], wt[m_asc_dn],
                             "ascending (VNC) -> DN", topn=10)

# ---- 6. DN output side (descending axons leave the brain) --------------
m_dn_out = np.isin(pre, dn_ids)
out_post = post[m_dn_out]
idx_o = np.fromiter((pos_of.get(int(b), -1) for b in out_post),
                    dtype=np.int64, count=len(out_post))
ok_o = idx_o >= 0
df_o = pd.DataFrame({"superclass": sup_arr[idx_o[ok_o]],
                     "syn": wt[m_dn_out][ok_o]})
print("\n== DN outgoing by target superclass ==")
print(df_o.groupby("superclass")["syn"].agg(["sum", "count"])
      .sort_values("sum", ascending=False).head(8).to_string())

# ---- 7. cb_intrinsic DN drivers in detail + lateralization -------------
idx_dn = np.fromiter((pos_of.get(int(b), -1) for b in pre[m_dn]),
                     dtype=np.int64, count=int(m_dn.sum()))
okk = idx_dn >= 0
df_dn = pd.DataFrame({"sc": sup_arr[idx_dn[okk]],
                      "ty": typ_arr[idx_dn[okk]],
                      "syn": wt[m_dn][okk]})
cen_ty = df_dn[df_dn["sc"] == "cb_intrinsic"].groupby("ty")["syn"] \
    .agg(["sum", "count"]).sort_values("sum", ascending=False)
print("\n== top cb_intrinsic (CX/LAL/SEZ) types -> DN ==")
print(cen_ty.head(15).to_string())

sites = fdata.site_positions(fdata.load_neuron_sites(), "PreSyn")
xs = np.array([sites.get(int(b), (np.nan, np.nan, np.nan))[0]
               for b in dn_ids])
print(f"\nDN terminal x_um: median {np.nanmedian(xs):.1f}, "
      f"L(x<0) {int(np.nansum(xs < 0))} / R(x>0) "
      f"{int(np.nansum(xs > 0))}")
pro_xs = np.array([sites.get(int(b), (np.nan, np.nan, np.nan))[0]
                   for b in pro_ids])
pro_side = side[pro_ids].to_numpy()
print("PRO x_um median by rootSide (sign convention):",
      {s: float(np.nanmedian(pro_xs[pro_side == s]))
       for s in ("L", "R")})
cen_ty_out = {k: {"syn": int(r["sum"]), "pairs": int(r["count"])}
              for k, r in cen_ty.head(15).iterrows()}
dn_side = {"L": int(np.nansum(xs < 0)), "R": int(np.nansum(xs > 0))}

rep = {
    "n_dn": int(len(dn_ids)),
    "dn_rootside": {k: int(v) for k, v in
                    side[dn_ids].value_counts().items()},
    "dn_top_types": {k: int(v) for k, v in dn_types.items()},
    "dn_input_by_superclass": {k: {"syn": int(r["sum"]),
                                   "pairs": int(r["count"])}
                               for k, r in by_sc.head(12).iterrows()},
    "dn_input_by_type": {k: {"syn": int(r["sum"]),
                             "pairs": int(r["count"])}
                         for k, r in by_ty.head(20).iterrows()},
    "pro_direct_dn": {"pairs": int(m_pro_dn.sum()),
                      "syn": int(wt[m_pro_dn].sum())},
    "two_hop": None if hop2 is None else {
        f"{a}/{b}": {"syn": int(r["sum"]), "pairs": int(r["count"])}
        for (a, b), r in hop2.head(20).iterrows()},
    "n_two_hop_intermediates": int(len(x_ids)),
    "ascending_to_dn_by_type": {k: {"syn": int(r["sum"]),
                                    "pairs": int(r["count"])}
                                for k, r in by_ty_a.head(10).iterrows()},
    "cen_intrinsic_to_dn_by_type": cen_ty_out,
    "dn_side_by_terminal_x": dn_side,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "dn_probe.json").write_text(json.dumps(rep, indent=1))
print(f"\nsaved {OUT / 'dn_probe.json'}")
