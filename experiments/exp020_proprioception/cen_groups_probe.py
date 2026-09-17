"""exp020 probe: CEN chem-group sizing for the MI-v2 DN-drive design.

Sizes the groups that circuit3._chem_groups_cen will register:
  - midline calibration from PRO rootSide (terminal-site x medians)
  - CEN side split by soma x at that midline
  - cell counts for the drive candidates: LAL*@side, DN@side,
    PS*/GNG*/VES* (top cb_intrinsic DN drivers)

Run from repository root:
    python experiments/exp020_proprioception/cen_groups_probe.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata

ann = fdata.load_annotations()
sup = ann["superclass"].fillna("")
typ = ann["type"].fillna("?")
side = ann["rootSide"].fillna("?")

# midline from PRO: terminal-x medians per rootSide
sites = fdata.site_positions(fdata.load_neuron_sites(), "PreSyn")
cls = ann["class"].fillna("")
sub = ann["subclass"].fillna("")
pro_m = (cls == "mechanosensory_proprioceptive") \
    | (sub == "chordotonal organ") | (sub == "campaniform sensilla") \
    | typ.str.contains("hair plate", case=False)
px = {s: np.median([sites[b][0] for b in ann.loc[pro_m & (side == s),
                                                  "bodyId"]
                     if b in sites]) for s in ("L", "R")}
mid = float((px["L"] + px["R"]) / 2)
print(f"PRO terminal-x medians L {px['L']:.1f} / R {px['R']:.1f} "
      f"-> midline {mid:.1f} um (x>mid = L)")

soma = fdata.neuron_positions(ann)
cen_m = sup.str.startswith("cb_") | (sup == "descending_neuron")
cen = ann.loc[cen_m & ann["bodyId"].isin(soma)]
sx = np.array([soma[int(b)][0] for b in cen["bodyId"]])
sL = sx > mid
print(f"CEN cells with soma: {len(cen):,}; side split "
      f"L {int(sL.sum())} / R {int((~sL).sum())}")

d = cen.assign(side=np.where(sL, "L", "R"),
               ty=cen["type"].fillna("?").values)
d["is_dn"] = (d["superclass"].fillna("") == "descending_neuron")
for label, m in [("LAL*", d["ty"].str.startswith("LAL")),
                 ("PS*", d["ty"].str.startswith("PS")),
                 ("GNG*", d["ty"].str.startswith("GNG")),
                 ("VES*", d["ty"].str.startswith("VES")),
                 ("DN", d["is_dn"])]:
    g = d[m].groupby("side").size()
    nty = d[m]["ty"].nunique()
    print(f"{label:5s}: {int(g.sum()):4d} cells, {nty:3d} types, "
          f"L {int(g.get('L', 0))} / R {int(g.get('R', 0))}")

print("\ntop DN types by side:")
dn = d[d["is_dn"]]
print(dn.groupby(["ty", "side"]).size().unstack(fill_value=0)
        .sort_values(by=["L", "R"], ascending=False).head(6).to_string())
print("\nLAL types by side:")
lal = d[d["ty"].str.startswith("LAL")]
print(lal.groupby(["ty", "side"]).size().unstack(fill_value=0)
         .sort_values(by=["L", "R"], ascending=False).head(8).to_string())
