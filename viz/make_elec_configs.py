"""Generate viz/data/elec_configs.json -- the electrode-configuration
catalogue consumed by the visualization page.

Configs:
  std_1020_45 : the 45-channel 10-20/10-10 hybrid currently in use
  std_1010_64 : the classic 64-channel 10-10 cap (chain midpoints + Iz)

Both are derived with the SAME calibrated frame (the handles / ni / yaw /
roll of viz/data/calib_1020_export.json when present, else the ANCHORS
defaults below), so switching between them never changes the calibration.

To add a CUSTOM config: append an entry to the generated file (or better,
extend this script) -- {"id", "label", "kind": "custom", "count",
"channels": {name: [x, y, z]}} -- and reload the page.  Custom configs are
displayed as-is; the rule-based 10-20 calibration sliders are disabled for
them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import ffbm.vizprep as vp

ANCHORS = {
    "Cz": [0.04023, 0.15186, -0.98758],
    "Fz": [0.02054, -0.58735, -0.80907],
    "Oz": [0.01184, 0.98972, 0.1425],
    "A1": [-0.78251, -0.08606, 0.61666],
    "A2": [0.72993, -0.11244, 0.67421],
}
NI_ARC_DEG = 223.6
FPZ_ARC_DEG = None
YAW_DEG = 0.0
ROLL_DEG = -0.3


def frame_params():
    """Calibration params: prefer the panel export, fall back to the
    constants above."""
    exp = Path(__file__).parent / "data" / "calib_1020_export.json"
    if exp.exists():
        e = json.loads(exp.read_text(encoding="utf-8"))
        h = e.get("handles")
        if h and "ni_arc_deg" in e:
            return {k: h[k] for k in ("Cz", "Fz", "Oz", "A1", "A2")}, \
                float(e["ni_arc_deg"]), float(e.get("yaw_deg", 0.0)), \
                float(e.get("roll_deg", 0.0))
    return ANCHORS, NI_ARC_DEG, YAW_DEG, ROLL_DEG


def as_channels(layout, order=None, extra_order=None):
    """Drop the landmark rows; channels only.  `order` pins the channel
    order (must match the exported viz_data meta for the 45); `extra_order`
    appends sites that are not in `order`."""
    drop = ("Nasion", "Inion")
    chan = {k: [round(float(x), 5) for x in v]
            for k, v in layout.items() if k not in drop}
    if order is not None:
        assert set(order) == set(chan), "order anchor mismatch"
        chan = {k: chan[k] for k in order}
    elif extra_order:
        assert set(extra_order) == set(chan)
        chan = {k: chan[k] for k in extra_order}
    return chan


def main():
    anchors, ni, yaw, roll = frame_params()
    cap45 = vp.standard_1020(anchors, system="1010", ni_arc_deg=ni,
                             yaw_deg=yaw, roll_deg=roll)
    cap64 = vp.standard_1010_full(anchors, ni_arc_deg=ni,
                                  yaw_deg=yaw, roll_deg=roll)
    # canonical channel order = the exported viz_data meta (the 45); the
    # 64 appends the new 10-10 sites in MIDPOINT_SITES + Iz order
    meta = json.loads((Path(__file__).parent / "data" / "viz_data.json")
                      .read_text(encoding="utf-8"))["meta"]["scalp"]
    order45 = meta["elec_names"]
    extras = [n for n, _a, _b in vp.MIDPOINT_SITES] + ["Iz"]
    order64 = order45 + [n for n in extras if n not in order45]
    cat = {
        "_schema": "electrode configs: name -> unit direction [x,y,z]; "
                   "add a custom entry with kind='custom' and reload",
        "version": 2,
        "frame": {"ni_arc_deg": ni, "yaw_deg": yaw, "roll_deg": roll},
        "configs": [
            {"id": "std_1020_45", "label": "10-20 系统",
             "kind": "std_1020_45", "count": 45,
             "channels": as_channels(cap45, order=order45)},
            {"id": "std_1010_64", "label": "10-10 扩展",
             "kind": "std_1010_64", "count": 64,
             "channels": as_channels(cap64, extra_order=order64)},
        ],
    }
    dst = Path(__file__).parent / "data" / "elec_configs.json"
    dst.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"wrote {dst}")
    for c in cat["configs"]:
        print(f"  {c['id']}: {c['count']} channels")


if __name__ == "__main__":
    main()
