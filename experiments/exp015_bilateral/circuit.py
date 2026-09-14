"""exp015: bilateral (both-lobe) visual cascade circuit.

Same 5-layer cascade as the mainline left-lobe circuit (exp005 builder),
but keeping BOTH optic lobes: every layer's id list is the union of both
sides (globally sorted by bodyId), and every edge table is filtered to
SAME-SIDE pre/post pairs so each lobe keeps its own wiring (R1-6 of one
eye never innervate the other lamina; Mi/Tm of one lobe stay ipsilateral).

Because per-cell wiring (L1/2 tetrad mass, Mi/Tm fan-in) is unchanged by
mirroring, the round-5 working point (g_hist = 0.16 nS etc.) should
transfer unchanged -- verified in run.py.

Positions are the native MaleCNS coordinates of BOTH lobes (um); the
bilateral cloud is roughly twice as wide along the inter-eye axis, which
the occipital placement absorbs along the shift axis (see run.py report).

Run from repository root:
    python experiments/exp015_bilateral/circuit.py     # geometry report
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata
from ffbm import vizprep as vp

# exp005's run.py loaded under a fixed name: this package's own run.py
# would shadow a plain `import run`
_spec = importlib.util.spec_from_file_location(
    "exp005_run", ROOT / "experiments" / "exp005_medulla_ds" / "run.py")
exp005 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp005)

OUT = Path(__file__).resolve().parent / "outputs"

R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SCALE = 400.0


def build_bilateral_circuit():
    """Both-lobe cascade with per-group edge tables and positions."""
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    sites = fdata.load_neuron_sites()
    pre_pos = fdata.site_positions(sites, "PreSyn")
    post_pos = fdata.site_positions(sites, "PostSyn")
    types = nodes.set_index("bodyId")["type"]

    def ids_of(mask_types, pos_map, kind="pre"):
        src = pre_pos if kind == "pre" else post_pos
        return np.array(sorted(b for b in nodes.loc[
            nodes["type"].isin(mask_types), "bodyId"] if b in src),
            dtype=np.int64)

    r_ids = ids_of(("R1-R6",), pre_pos)
    l_ids = ids_of(exp005.LAMINA_TYPES, post_pos, "post")
    mid_ids = ids_of(exp005.MID_TYPES, pre_pos)
    t45_ids = ids_of(exp005.T45_SUBTYPES, pre_pos)

    # midline from the pooled x 2-means over all four layers (identical to
    # the exp005 selector, just without dropping the right side)
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    all_x = np.concatenate([
        [pre_pos[b][0] for b in r_ids],
        [post_pos[b][0] for b in l_ids],
        [pre_pos[b][0] for b in mid_ids],
        [pre_pos[b][0] for b in t45_ids]])
    midline, _ = exp005.split_sides(all_x)

    def side_of(ids, pos_map):
        return np.array([pos_map[b][0] < midline for b in ids])

    side_r = side_of(r_ids, pre_pos)
    side_l = side_of(l_ids, post_pos)
    side_mid = side_of(mid_ids, pre_pos)
    side_t45 = side_of(t45_ids, pre_pos)
    print(f"circuit (both lobes): R {len(r_ids):,} "
          f"(L {side_r.sum():,}/R {(~side_r).sum():,}) | "
          f"L {len(l_ids):,} ({side_l.sum():,}/{(~side_l).sum():,}) | "
          f"Mi/Tm {len(mid_ids):,} ({side_mid.sum():,}/{(~side_mid).sum():,}) "
          f"| T4/T5 {len(t45_ids):,} "
          f"({side_t45.sum():,}/{(~side_t45).sum():,})")

    r_type = np.array([types.get(b) for b in r_ids])
    l_type = np.array([types.get(b) for b in l_ids])
    mid_type = np.array([types.get(b) for b in mid_ids])
    t45_type = np.array([types.get(b) for b in t45_ids])

    # same-side edge filter: each lobe wired to itself only
    side_by_body = {}
    for ids, side, pm in ((r_ids, side_r, pre_pos), (l_ids, side_l, post_pos),
                          (mid_ids, side_mid, pre_pos),
                          (t45_ids, side_t45, pre_pos)):
        for b, s in zip(ids.tolist(), side.tolist()):
            side_by_body[int(b)] = bool(s)

    e = edges.assign(tp=edges["body_pre"].map(types),
                     tq=edges["body_post"].map(types))
    all_pre_ok = set(r_ids.tolist()) | set(l_ids.tolist()) \
        | set(mid_ids.tolist())
    all_post_ok = set(l_ids.tolist()) | set(mid_ids.tolist()) \
        | set(t45_ids.tolist())

    def pair_edges(pre_types, post_types):
        s = e[e["tp"].isin(pre_types) & e["tq"].isin(post_types)]
        s = s[s["body_pre"].isin(all_pre_ok)
              & s["body_post"].isin(all_post_ok)]
        sp = s["body_pre"].map(side_by_body)
        sq = s["body_post"].map(side_by_body)
        cross = int((sp != sq).sum())
        s = s[sp == sq]
        return s[["body_pre", "body_post", "weight", "sign"]].copy(), cross

    e_rl, cross_rl = pair_edges(("R1-R6",), exp005.LAMINA_TYPES)
    e_lm, cross_lm = pair_edges(exp005.LAMINA_TYPES, exp005.MID_TYPES)
    e_mt, cross_mt = {}, 0
    for mt in exp005.MID_TYPES:
        e_mt[mt], c = pair_edges((mt,), exp005.T45_SUBTYPES)
        cross_mt += c
    n_syn = int(e_rl["weight"].sum() + e_lm["weight"].sum()
                + sum(v["weight"].sum() for v in e_mt.values()))
    print(f"edges: R->L {len(e_rl):,} | L->Mi/Tm {len(e_lm):,} | "
          f"Mi/Tm->T4/T5 {sum(len(v) for v in e_mt.values()):,} "
          f"| synapses {n_syn:,} "
          f"(dropped cross-lobe: {cross_rl}/{cross_lm}/{cross_mt})")

    # left-lobe reference geometry (eye axis, hex grating axis, phase0) --
    # the head-placement axis and the stimulus-texture axis must stay the
    # LEFT lobe's own, computed exactly as exp005 does for one lobe
    r_left = r_ids[side_r]
    l_left = l_ids[side_l]
    mid_left = mid_ids[side_mid]
    t45_left = t45_ids[side_t45]
    r_pos = np.array([pre_pos[b] for b in r_left])
    t45_pos = np.array([pre_pos[b] for b in t45_left])
    u_eye_left = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye_left = u_eye_left / np.linalg.norm(u_eye_left)
    u_eye_center = r_pos.mean(axis=0)
    u_eye_center /= np.linalg.norm(u_eye_center)

    mi1_ids = mid_left[mid_type[side_mid] == "Mi1"]
    mi1_hex_vals = np.array([hex1.get(b, np.nan) for b in mi1_ids])
    mi1_pos = np.array([pre_pos[b] for b in mi1_ids])
    ok = np.isfinite(mi1_hex_vals)
    a = np.array([1.0, 0.0, 0.0]) - u_eye_left * u_eye_left[0]
    a /= np.linalg.norm(a)
    bvec = np.cross(u_eye_left, a)
    X = np.stack([mi1_pos[ok] @ a, mi1_pos[ok] @ bvec], axis=1)
    coef, *_ = np.linalg.lstsq(X, mi1_hex_vals[ok], rcond=None)
    u_dir = coef[0] * a + coef[1] * bvec
    u_dir /= np.linalg.norm(u_dir)
    pred = X @ coef
    r2 = 1.0 - np.sum((mi1_hex_vals[ok] - pred) ** 2) / np.sum(
        (mi1_hex_vals[ok] - mi1_hex_vals[ok].mean()) ** 2)
    phase0_x = r_pos @ u_dir
    lam = (phase0_x.max() - phase0_x.min()) / 2.0
    phase0 = 2.0 * np.pi * (phase0_x - phase0_x.mean()) / lam
    elec = t45_pos.mean(axis=0) + exp005.ELECTRODE_OFFSET_UM * u_dir

    return dict(r_ids=r_ids, l_ids=l_ids, mid_ids=mid_ids,
                t45_ids=t45_ids, r_type=r_type, l_type=l_type,
                mid_type=mid_type, t45_type=t45_type,
                e_rl=e_rl, e_lm=e_lm, e_mt=e_mt,
                pre_pos=pre_pos, post_pos=post_pos, phase0=phase0,
                elec=elec, hex_r2=r2,
                side_r=side_r, side_l=side_l, side_mid=side_mid,
                side_t45=side_t45, midline_x=midline,
                u_eye_left=u_eye_left, u_eye_center=u_eye_center)


def geometry_report(circuit):
    """Bilateral cloud geometry at x400 inside the 4-sphere head."""
    pp = circuit["pre_pos"]
    qq = circuit["post_pos"]
    pts = np.vstack([np.array([pp[b] for b in circuit["r_ids"]]),
                     np.array([qq[b] for b in circuit["l_ids"]]),
                     np.array([pp[b] for b in circuit["mid_ids"]]),
                     np.array([pp[b] for b in circuit["t45_ids"]])])
    center = pts.mean(axis=0)
    rel = pts - center
    span = rel.max(axis=0) - rel.min(axis=0)
    r_max = np.linalg.norm(rel, axis=1).max()
    u_eye = circuit["u_eye_left"]
    scaled = rel * SCALE
    shift = vp.occipital_shift(scaled, u_eye, R_BRAIN)
    after = np.linalg.norm(scaled + shift, axis=1).max()
    proj = scaled @ u_eye
    perp = np.sqrt(np.maximum(np.sum(scaled ** 2, axis=1) - proj ** 2, 0))
    rep = {
        "per_layer": {"R": int(len(circuit["r_ids"])),
                      "L": int(len(circuit["l_ids"])),
                      "MID": int(len(circuit["mid_ids"])),
                      "T45": int(len(circuit["t45_ids"]))},
        "span_um": [round(float(x), 1) for x in span],
        "r_max_native_um": round(float(r_max), 1),
        "r_max_x400_um": round(float(r_max * SCALE), 0),
        "perp_max_x400_um": round(float(perp.max()), 0),
        "brain_radius_um": R_BRAIN,
        "margin_frac_0p98_um": round(0.98 * R_BRAIN, 0),
        "shift_mm": round(float(np.linalg.norm(shift)) / 1000.0, 1),
        "r_max_after_shift_um": round(float(after), 0),
        "fits_in_brain": bool(after <= 0.98 * R_BRAIN + 1e-6),
        "u_eye_left": [round(float(x), 4) for x in u_eye],
    }
    return rep


def l12_mass_report(circuit):
    """Per-side L1/L2 tetrad mass -- the g_hist transfer check."""
    l_ids = circuit["l_ids"]
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(len(l_ids))
    e_rl = circuit["e_rl"]
    mass = np.bincount(l_index[e_rl["body_post"].to_numpy(np.int64)],
                       weights=e_rl["weight"].to_numpy(np.float64),
                       minlength=len(l_ids))
    side = circuit["side_l"]
    lt = circuit["l_type"]
    innerv = np.isin(lt, ("L1", "L2")) & (mass >= 50.0)
    rep = {}
    for name, m in (("left", side & innerv), ("right", (~side) & innerv)):
        rep[name] = {"n": int(m.sum()),
                     "mass_median": round(float(np.median(mass[m])), 1),
                     "mass_p25": round(float(np.percentile(mass[m], 25)),
                                       1),
                     "mass_p75": round(float(np.percentile(mass[m], 75)),
                                       1)}
    return rep


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = build_bilateral_circuit()
    rep = {"geometry": geometry_report(circuit),
           "l12_mass": l12_mass_report(circuit)}
    (OUT / "bilateral_geometry.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
