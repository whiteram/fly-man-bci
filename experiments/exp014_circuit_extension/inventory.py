"""exp014 phase 1 -- lobula / lobula-plate circuit-extension inventory.

Pure data query (no simulation): who is downstream of T4/T5, how big and
how numerous are those partners, and where do they sit? Output ranks the
candidate classes for phase-2 wiring (spiking-proxy LPTCs first) and
estimates each class's spatial span -- the dipole-length lever that makes
tangential cells interesting for the scalp signal at x400.

Run from repository root:
    python experiments/exp014_circuit_extension/inventory.py   (~5 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    ann = fdata.load_annotations()
    types = nodes.set_index("bodyId")["type"]

    t45 = nodes.loc[nodes["type"].str.startswith(("T4", "T5")), "bodyId"]
    t45_set = set(t45)
    print(f"T4/T5 cells: {len(t45):,}")

    # ---- downstream of T4/T5, grouped by postsynaptic type ----
    ds = edges[edges["body_pre"].isin(t45_set)].copy()
    ds["tp"] = ds["body_pre"].map(types)
    ds["tq"] = ds["body_post"].map(types)
    ds = ds.dropna(subset=["tq"])
    grp = ds.groupby("tq").agg(
        n_edges=("body_post", "size"),
        n_syn=("weight", "sum"),
        n_cells=("body_post", "nunique"))
    grp = grp.sort_values("n_syn", ascending=False)
    print("\ntop-25 T4/T5 downstream classes (by synapse count):")
    print(grp.head(25).to_string())

    # ---- spatial span of the top classes (site clouds, both kinds) ----
    cand = grp.head(25)
    cand_bodies = nodes.loc[nodes["type"].isin(cand.index), "bodyId"]
    soma = fdata.neuron_positions(ann)
    sites = fdata.load_neuron_sites()
    sel = sites[sites["body"].isin(set(cand_bodies))]
    print(f"\nsite rows for candidate classes: {len(sel):,}")
    span = {}
    for ty, g in nodes[nodes["type"].isin(cand.index)].groupby("type"):
        bodies = set(g["bodyId"])
        pts = sel[sel["body"].isin(bodies)]
        if len(pts):
            xyz = pts[["x_um", "y_um", "z_um"]].to_numpy()
            cen = xyz.mean(axis=0)
            dia = float(np.linalg.norm(xyz - cen, axis=1).max() * 2)
        else:
            dia = None
        somas = [soma[b] for b in bodies if b in soma]
        soma_spread = (float(np.linalg.norm(
            np.vstack(somas) - np.vstack(somas).mean(axis=0), axis=1).max()
        ) * 2 if len(somas) > 1 else 0.0)
        span[ty] = {"site_cloud_diameter_um": round(dia, 1) if dia else None,
                    "soma_cloud_diameter_um": round(soma_spread, 1),
                    "n_cells": int(len(bodies))}

    # ---- upstream partners of the biggest downstream classes ----
    upstream = {}
    for ty in cand.index[:12]:
        bodies = set(nodes.loc[nodes["type"] == ty, "bodyId"])
        up = edges[edges["body_post"].isin(bodies)].copy()
        up["tp"] = up["body_pre"].map(types)
        up = up.dropna(subset=["tp"])
        top = up.groupby("tp")["weight"].sum().sort_values(
            ascending=False).head(6)
        upstream[ty] = {k: int(v) for k, v in top.items()}

    summary = {
        "n_t45": int(len(t45_set)),
        "downstream_top25": [
            {"type": ty, "n_edges": int(r.n_edges), "n_syn": int(r.n_syn),
             "n_cells": int(r.n_cells), **span.get(ty, {})}
            for ty, r in grp.head(25).iterrows()],
        "upstream_of_top12": upstream,
        "notes": "site_cloud_diameter = 2*max |site - centroid| over ALL "
                 "synapse sites of the class (the x400 dipole-length lever)",
    }
    (OUT / "inventory.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwritten to {OUT / 'inventory.json'}")


if __name__ == "__main__":
    main()
