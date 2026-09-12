"""Extract the visual-system subnetwork from MaleCNS v1.0 (see docs/data.md).

Run from the repository root:  python scripts/build_visual_subnetwork.py
"""

import pandas as pd

RAW = "data/raw"
DERIVED = "data/derived"

VISUAL_SUPERCLASSES = [
    "ol_intrinsic",        # optic lobe intrinsic (lamina/medulla/lobula/lobula plate)
    "visual_projection",   # VPN - project visual info to central brain
    "ol_sensory",          # photoreceptors
    "visual_centrifugal",  # VCN - feedback from central brain to optic lobe
]

import os
os.makedirs(DERIVED, exist_ok=True)

ann = pd.read_feather(f"{RAW}/body-annotations.feather")
nt = pd.read_feather(f"{RAW}/body-neurotransmitters.feather")

nodes = ann[["bodyId", "type", "superclass", "class", "somaSide",
             "statusLabel", "dimorphism"]].copy()
nodes = nodes.merge(nt[["body", "consensus_nt", "predicted_nt_confidence"]],
                    left_on="bodyId", right_on="body", how="left").drop(columns="body")
nodes["bodyId"] = nodes["bodyId"].astype("int32")

visual_ids = set(nodes.loc[nodes["superclass"].isin(VISUAL_SUPERCLASSES), "bodyId"].tolist())
print(f"visual neurons (by superclass): {len(visual_ids):,}")

cw = pd.read_feather(f"{RAW}/connectome-weights.feather")
cw = cw.astype({"body_pre": "int32", "body_post": "int32", "weight": "int32"})
print(f"total edges: {len(cw):,}")

mask = cw["body_pre"].isin(visual_ids) & cw["body_post"].isin(visual_ids)
edges = cw[mask].copy()
print(f"edges within visual system: {len(edges):,} "
      f"({len(edges)/len(cw)*100:.1f}% of all edges)")
print(f"total synapses in visual subnetwork: {int(edges['weight'].sum()):,}")

nodes = nodes[nodes["bodyId"].isin(visual_ids)].reset_index(drop=True)

# NT sign on edges
sign = nodes.set_index("bodyId")["consensus_nt"]
edges["nt_pre"] = edges["body_pre"].map(sign)
excit = {"acetylcholine"}
edges["sign"] = edges["nt_pre"].map(
    lambda x: 1 if x in excit else (-1 if pd.notna(x) else 0))

nodes.to_parquet(f"{DERIVED}/visual_nodes.parquet")
edges[["body_pre", "body_post", "weight", "nt_pre", "sign"]].to_parquet(
    f"{DERIVED}/visual_edges.parquet")

# type-to-type aggregation for circuit validation
tmap = nodes.set_index("bodyId")["type"]
edges["type_pre"] = edges["body_pre"].map(tmap)
edges["type_post"] = edges["body_post"].map(tmap)

print("\n=== node stats ===")
print("neurons by superclass:")
print(nodes["superclass"].value_counts().to_string())
print("\ntop 15 cell types:")
print(nodes["type"].value_counts().head(15).to_string())

print("\n=== circuit sanity check: top input types to T4 (ON motion) ===")
t4 = edges[edges["type_post"].str.startswith("T4", na=False)].groupby("type_pre")["weight"].sum().nlargest(8)
print(t4.to_string())
print("\n=== top input types to T5 (OFF motion) ===")
t5 = edges[edges["type_post"].str.startswith("T5", na=False)].groupby("type_pre")["weight"].sum().nlargest(8)
print(t5.to_string())
