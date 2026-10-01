"""bci/condstate6 anatomy: type the CEN_C table and profile the
afferents of the DAN-family (PAM/PPL1/PPM) and MBON pools.

Zero-acquisition support for the pathway verdict: after the MBMD
split (class-MBON -> class-DAN GABA rows moved out of CEN_C at gain
178), the DAN_err cells keep essentially NO inhibitory synapses in
CEN -- the MBON GABA feedback is the ONLY suppressor channel, so any
gate suppression must flow through it (or through conductance shunting
from excitatory background, which the usmb115 control then rules out).

Usage (conda ffbm, repo root):
    python bci/condstate6/anatomy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    from ffbm import data as fdata
    from ffbm import regions as freg

    ann = fdata.load_annotations()
    tb = ann.set_index("bodyId")["type"]
    circuit, _ = freg.build_circuit(
        {"visual_bilateral": True, "ol_rest": True, "central_brain": True,
         "olfactory": True})
    # exp017's CEN raw ids (sorted, soma-bearing) share positions with
    # the circuit's sorted compact ids
    mask = (ann["superclass"].fillna("").str.startswith("cb_")
            | (ann["superclass"] == "descending_neuron"))
    soma = fdata.neuron_positions(ann)
    raw = np.array(sorted(
        ann.loc[mask & ann["bodyId"].isin(soma), "bodyId"].tolist()),
        dtype=np.int64)
    cen_ids = np.asarray(circuit["extra_pops"]["CEN"]["ids"])
    assert len(raw) == len(cen_ids)
    comp2raw = pd.Series(raw, index=cen_ids)

    tab = circuit["extra_edges"]["CEN_C"]["table"]
    t = tab.assign(
        pre_t=tab["body_pre"].map(comp2raw).map(tb).to_numpy(),
        post_t=tab["body_post"].map(comp2raw).map(tb).to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / "cen_typed.parquet")
    print(f"typed CEN_C: {len(t):,} rows "
          f"({t['pre_t'].notna().mean():.1%} typed) "
          f"-> {OUT / 'cen_typed.parquet'}")

    dan = t["post_t"].fillna("").str.startswith(("PAM", "PPL1", "PPL2"))
    inh = t[dan & (t["sign"] < 0)]
    print(f"\nDAN-family afferent rows: {int(dan.sum()):,}; "
          f"inhibitory rows: {len(inh)} "
          f"(wsum {inh['weight'].sum():.0f}) -- vs MBMD's 349-row GABA "
          f"feedback split out of this table")
    print("\ntop DAN-family afferent types:")
    print(t[dan].groupby("pre_t")["weight"].sum()
          .sort_values(ascending=False).head(10).to_string())
    mbon = t["post_t"].fillna("").str.startswith("MBON")
    print("\ntop MBON afferent types (non-KC):")
    nk = t[mbon].groupby("pre_t")["weight"].sum() \
        .sort_values(ascending=False)
    print(nk[~nk.index.str.startswith(("KC", "Kenyon"))]
          .head(10).to_string())


if __name__ == "__main__":
    main()
