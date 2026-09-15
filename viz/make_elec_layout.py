"""Regenerate viz/data/elec_layout_1010.json.

Two sources, in priority order:
  1. viz/data/calib_1020_export.json -- the JSON copied from the page's
     "10-20 标定" panel (paste what the user hands over here, then run
     this script).  Its `layout` section is used verbatim.
  2. the ANCHORS constants below, through ffbm.vizprep.standard_1020
     (calibration provenance lives in those constants).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import ffbm.vizprep as vp

ANCHORS = {
    "Cz": [0.03966, -0.16267, -0.98588],
    "Fz": [-0.02286, -0.99859, -0.04794],
    "Oz": [-0.00497, 0.93516, 0.3542],
    "A1": [-0.88792, -0.09955, 0.44911],
    "A2": [0.87808, -0.19289, 0.43792],
}
NI_ARC_DEG = 260.0   # nasion-inion arc measured on the ghost-head mesh
                     # (both landmarks below the ear-line plane)
FPZ_ARC_DEG = 94.0   # Fpz re-pinned 10 deg above the 10%-rule spot
                     # (0.40*260 = 104): on the brow, not by the nose

dst = Path(__file__).parent / "data" / "elec_layout_1010.json"
old = json.loads(dst.read_text(encoding="utf-8"))

export = Path(__file__).parent / "data" / "calib_1020_export.json"
if export.exists():
    exp = json.loads(export.read_text(encoding="utf-8"))
    if "handles" in exp and "ni_arc_deg" in exp:
        # preferred: re-derive from the 6 handles so the CURRENT rules
        # (incl. the th-scaled transverse ring + global yaw) apply
        h = exp["handles"]
        anchors = {k: h[k] for k in ("Cz", "Fz", "Oz", "A1", "A2")}
        layout = vp.standard_1020(anchors, system="1010",
                                  ni_arc_deg=exp["ni_arc_deg"],
                                  yaw_deg=exp.get("yaw_deg", 0.0))
        out = {k: [round(float(x), 5) for x in v]
               for k, v in layout.items()}
        print(f"source: calibration handles re-derived "
              f"(ni={exp['ni_arc_deg']} deg, ear th={exp.get('ear_th_deg')} deg, "
              f"yaw={exp.get('yaw_deg', 0.0)} deg)")
    else:
        out = {k: [float(x) for x in v] for k, v in exp["layout"].items()}
        print("source: calibration export layout (verbatim)")
else:
    layout = vp.standard_1020(ANCHORS, system="1010",
                              ni_arc_deg=NI_ARC_DEG,
                              fpz_arc_deg=FPZ_ARC_DEG)
    out = {k: [round(float(x), 5) for x in v] for k, v in layout.items()}
    print(f"source: anchors + ni_arc={NI_ARC_DEG}, fpz_arc={FPZ_ARC_DEG}")

assert set(old.keys()) == set(out.keys()), "channel set changed!"
# canonical channel order = the existing file's (index alignment!)
out = {k: out[k] for k in old}
moved = {}
for k in out:
    dot = sum(old[k][i] * out[k][i] for i in range(3))
    n = math.sqrt(sum(x * x for x in old[k])) * \
        math.sqrt(sum(x * x for x in out[k]))
    moved[k] = math.degrees(math.acos(max(-1.0, min(1.0, dot / n))))
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"wrote {dst}")
print("moved > 0.5 deg:", {k: round(d, 1)
                           for k, d in moved.items() if d > 0.5})
