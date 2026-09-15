"""Generate viz/data/elec_configs.json -- the electrode-configuration
catalogue consumed by the visualization page.

Configs:
  std_1020_45 : the 45-channel 10-20/10-10 hybrid currently in use
  std_1010_64 : the classic 64-channel 10-10 cap (chain midpoints + Iz)
  std_105_128 : the 128-level cap -- the 64 subdivided once more to 5%
                steps (10-5 family); new sites are named by matching
                MNE-Python's standard_1005 montage (343 official 10-05
                positions) to chain-pair midpoints

All are derived with the SAME calibrated frame (the handles / ni / yaw /
roll of viz/data/calib_1020_export.json when present, else the ANCHORS
defaults below), so switching between them never changes the calibration.

To add a CUSTOM config: append an entry to the generated file (or better,
extend this script) -- {"id", "label", "kind": "custom", "count",
"channels": {name: [x, y, z]}} -- and reload the page.  Custom configs are
displayed as-is; the rule-based 10-20 calibration sliders are disabled for
them.  See ELEC_CONFIGS.md for the full documentation.
"""
import json
import sys
from pathlib import Path

import numpy as np

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


# 10-10 chains (adjacent sites 10% apart) used for the 5% subdivision
CHAINS = [
    ["Nasion", "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
     "Inion"],
    ["Fp1", "AF3", "F3", "FC3", "C3", "CP3", "P3", "PO3", "O1"],
    ["Fp2", "AF4", "F4", "FC4", "C4", "CP4", "P4", "PO4", "O2"],
    ["F1", "FC1", "C1", "CP1", "P1"],
    ["F2", "FC2", "C2", "CP2", "P2"],
    ["Fp1", "AF7", "F7", "FT7", "T7", "TP7", "P7", "PO7", "O1"],
    ["Fp2", "AF8", "F8", "FT8", "T8", "TP8", "P8", "PO8", "O2"],
    ["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    ["FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8"],
    ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"],
    ["TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8"],
    ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    ["PO7", "PO3", "POz", "PO4", "PO8"],
    ["O1", "Oz", "O2"],
    ["AF7", "AF3", "AFz", "AF4", "AF8"],
]


def mne_1005_unit_dirs():
    """MNE standard_1005 (the 10-05 system, 343 named sites) as
    {name: unit vector}, in the montage's own frame."""
    from mne.channels import make_standard_montage
    import warnings
    warnings.filterwarnings("ignore")
    m = make_standard_montage("standard_1005")
    pos = m.get_positions()["ch_pos"]
    return {k: np.asarray(v, dtype=np.float64) /
            np.linalg.norm(v) for k, v in pos.items()}


def build_egi_256(cap64):
    """EGI HydroCel GSN 256 (MNE 'EGI_256', E1..E256): transform the real
    geodesic geometry into our calibrated frame.  Alignment uses the
    montage's own fiducials (nasion/lpa/rpa) -- identical Gram-Schmidt
    triads on both frames give an exact, chirality-correct rotation.
    Face/neck cone sites are dropped (38/40 deg, matching the
    pipeline's electrode-placement rules)."""
    from mne.channels import make_standard_montage
    import warnings
    warnings.filterwarnings("ignore")
    m = make_standard_montage("EGI_256")
    pos = m.get_positions()
    ch_pos = pos["ch_pos"]
    nas, lpa = (np.asarray(pos[k], float) for k in ("nasion", "lpa"))
    names = [n for n in ch_pos if np.linalg.norm(ch_pos[n]) > 1e-6]
    P = np.array([ch_pos[n] for n in names])
    # algebraic sphere fit (Kasa): net center
    A = np.hstack([2 * P, np.ones((len(P), 1))])
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]

    def unit(v):
        v = np.asarray(v, dtype=np.float64)
        return v / np.linalg.norm(v)

    def triad(ant_dir, left_dir):
        ba = unit(ant_dir)
        bl = unit(left_dir - ba * (ba @ left_dir))
        return np.column_stack([ba, bl, np.cross(ba, bl)])

    # our calibrated frame
    cz = unit(cap64["Cz"])
    ant = unit(cap64["Fz"] - cap64["Oz"])
    ant = ant - cz * (ant @ cz)
    ant = unit(ant)
    l_ours = unit(cap64["A1"] - ant * (ant @ cap64["A1"]))
    R_ours = triad(ant, l_ours)
    # EGI frame triad from its fiducials (relative to the net center)
    R_mne = triad(unit(nas - c), unit(lpa - c))
    R = R_ours @ R_mne.T                      # MNE frame -> our frame
    out = {}
    for i, n in enumerate(names):
        v = unit(P[i] - c)
        v = R @ v
        # neck cone only: brow-level midline sites (Fpz equivalents) are
        # legitimate scalp electrodes and must survive
        if v @ (-cz) > np.cos(np.radians(40.0)):
            continue
        out[n] = v
    return out


def build_105_sites(cap64):
    """5%-subdivision candidates on the 10-10 chains.  Naming: each
    chain-adjacent pair (a, b) also exists in MNE's standard_1005; the
    midpoint of the MNE pair identifies the official 10-05 name (nearest
    10-05 site within the MNE frame -- no cross-frame fitting needed).
    Returns {name: unit dir} for confidently named new sites."""
    mne_dirs = mne_1005_unit_dirs()
    used = set(cap64) | {"Nasion", "Inion"}
    out, skipped = {}, 0
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:]):
            if a not in cap64 or b not in cap64:
                continue
            if a not in mne_dirs or b not in mne_dirs:
                skipped += 1
                continue
            m = cap64[a] + cap64[b]
            m = m / np.linalg.norm(m)
            m_mne = mne_dirs[a] + mne_dirs[b]
            m_mne = m_mne / np.linalg.norm(m_mne)
            best_n, best_d = None, 1.0
            for n, p in mne_dirs.items():
                d = float(np.degrees(np.arccos(np.clip(m_mne @ p, -1, 1))))
                if d < best_d:
                    best_n, best_d = n, d
            if best_n is None or best_d > 2.0:
                skipped += 1
                continue
            if best_n in used or best_n in out:
                continue
            out[best_n] = m
    if skipped:
        print(f"  ({skipped} chain segments skipped -- no confident name)")
    return out


def prune_to_target(new_sites, cap64, target):
    """Keep `target` new sites: prefer plain 10-10-grid names over the
    exotic 'h'-suffixed 10-05 ones, then alphabetical for stability."""
    def score(item):
        name = item[0]
        return (name.endswith("h"), name)
    keep = sorted(new_sites.items(), key=score)[:target]
    return dict(keep)


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

    configs = [
        {"id": "std_1020_45", "label": "10-20 系统",
         "kind": "std_1020_45", "count": 45,
         "channels": as_channels(cap45, order=order45)},
        {"id": "std_1010_64", "label": "10-10 扩展",
         "kind": "std_1010_64", "count": 64,
         "channels": as_channels(cap64, extra_order=order64)},
    ]

    try:
        new_sites = build_105_sites(cap64)
        n_cand = len(new_sites)
        new_sites = prune_to_target(new_sites, cap64,
                                    128 - len(order64))
        order128 = order64 + sorted(new_sites)
        cap128 = dict(cap64)
        cap128.update(new_sites)
        configs.append(
            {"id": "std_105_128", "label": "10-5 细分",
             "kind": "std_105_128", "count": len(order128),
             "channels": as_channels(cap128, extra_order=order128)})
        print(f"  10-5 subdivision: {n_cand} named candidates -> "
              f"kept {len(new_sites)} (total {len(order128)})")
    except ImportError:
        print("  mne not available -- skipping the 10-5/128 config")

    try:
        egi = build_egi_256(cap64)
        order_egi = sorted(egi.keys())
        configs.append(
            {"id": "egi_256", "label": "EGI 256 导",
             "kind": "egi_256", "count": len(order_egi),
             "channels": {k: [round(float(x), 5) for x in egi[k]]
                          for k in order_egi}})
        print(f"  EGI HydroCel 256: {len(order_egi)} sites on the cap "
              "(face/neck cones excluded)")
    except ImportError:
        print("  mne not available -- skipping the EGI 256 config")

    cat = {
        "_schema": "electrode configs: name -> unit direction [x,y,z]; "
                   "add a custom entry with kind='custom' and reload",
        "version": 3,
        "frame": {"ni_arc_deg": ni, "yaw_deg": yaw, "roll_deg": roll},
        "configs": configs,
    }
    dst = Path(__file__).parent / "data" / "elec_configs.json"
    dst.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"wrote {dst}")
    for c in cat["configs"]:
        print(f"  {c['id']}: {c['count']} channels")


if __name__ == "__main__":
    main()
