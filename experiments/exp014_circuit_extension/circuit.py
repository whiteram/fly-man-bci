"""exp014 phase 2 -- lobula-plate extension of the left-lobe cascade.

Adds the VS (vertical-system tangential) cells to the exp005 circuit:
direct T4/T5 -> VS wiring from the dataset (2/3 of VS's T4/T5 input
arrives from the d subtypes; 18 cells, large dendritic spans -> the
strongest single-dipole candidates at x400). Spiking proxy (VS has Na
spikes in flies); returns the extension tables consumed by run.py.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm import data as fdata

import run as exp005


def build_with_lop():
    """exp005.build_circuit() + left-lobe VS population and T4/T5->VS edges."""
    circuit = exp005.build_circuit()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    t45_ids = circuit["t45_ids"]
    midline = float(np.mean([pp[b][0] for b in t45_ids]))

    nodes = fdata.load_visual_nodes()
    vs = nodes[nodes["type"] == "VS"]
    vs_ids = np.array(sorted(b for b in vs["bodyId"] if b in qq),
                      dtype=np.int64)
    vs_left = np.array([b for b in vs_ids if qq[b][0] < midline],
                       dtype=np.int64)
    print(f"VS cells: {len(vs_ids)} total, {len(vs_left)} left lobe")

    edges = fdata.load_visual_edges()
    types = nodes.set_index("bodyId")["type"]
    e = edges[edges["body_pre"].isin(set(t45_ids))
              & edges["body_post"].isin(set(vs_left))].copy()
    e = e[["body_pre", "body_post", "weight", "sign"]]
    print(f"T4/T5->VS edges: {len(e):,} | synapses: {int(e['weight'].sum()):,}"
          f" | signs: {sorted(e['sign'].unique().tolist())}")
    circuit["vs_ids"] = vs_left
    circuit["e_vs"] = e.reset_index(drop=True)
    return circuit


if __name__ == "__main__":
    build_with_lop()
