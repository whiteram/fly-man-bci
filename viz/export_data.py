"""Export simulation data for the 3D visualization page (viz/index.html).

Runs one naturalistic simulation of the left-lobe cascade through the
SHARED pipeline (ffbm.pipeline: OU background noise, conductance MT
synapses, per-edge delays) with three internal fly-head electrodes
(eye / lamina / medulla, sealed-head kernel) plus a 17-electrode scalp
array in the x400 human-head thought experiment (4-sphere kernel,
occipital placement), background human EEG and SNR metrics, and saves
everything the page needs:

  positions of every neuron per layer (centered, um)
  a sample of edges for the flow visualization
  electrode positions + head-sphere geometry
  1 kHz traces: stimulus luminance, per-layer rates, 3 phi channels,
  17 scalp channels (clean + background, channel 0 on the eye axis)

Run from repository root:
    python viz/export_data.py        (~15 min, simulation dominates)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

from ffbm import pipeline as fp
from ffbm import params as fparams
from ffbm import regions as freg
from ffbm import vizprep as vp
from ffbm.forward import FourSpherePairField, SealedHeadPairField, \
    four_sphere_rows

from circuit import exp005

OUT = ROOT / "viz" / "data"
DT = exp005.DT

# naturalistic protocol (exp007-style): continuous waveforms, no square
# flashes -- the resulting EEG traces are wavy, not blocky.  All protocol
# hyperparameters live in the user-modifiable registry (src/ffbm/
# params.py -> docs/PARAMS.md); edit there, not here.
_ST = fparams.SECTIONS["stimulus"]
SEED = _ST["seed"][0]
T_EPOCHS = [tuple(e) for e in _ST["t_epochs"][0]]
T_END = T_EPOCHS[-1][2]
STIM_CONTRAST = _ST["stim_contrast"][0]
I_LUM = _ST["i_lum"][0]
DRIFT_SPEED = _ST["drift_speed"][0]
LAM_UM = _ST["lam_um"][0]
EPOCH_COLORS = {"dark": "#33465a", "flicker": "#f7c948", "drift+": "#34d399",
                "drift-": "#22d3ee", "band+": "#f472b6", "band-": "#c084fc"}

N_FLOW_EDGES = 1400


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="export viz data for a chosen region set")
    ap.add_argument("--regions", type=str, default=None,
                    help="comma-separated active regions "
                         f"(known: {','.join(freg.known_regions())}); "
                         "default = params registry regions_default")
    ap.add_argument("--elec-layout", type=str, default=None,
                    help="JSON file mapping electrode names to unit "
                         "directions (standard_1020 output or the "
                         "electrode-editor export); overrides the "
                         "capped Fibonacci layout")
    ap.add_argument("--visual-input", type=str, default="natural_1d",
                    help="visual input id from viz/data/"
                         "visual_inputs.json (natural_1d = 1/f flicker "
                         "+ drift protocol; video ids map grayscale "
                         "frames through the ommatidia sampling)")
    ap.add_argument("--pool", action="store_true",
                    help="EXPERIMENTAL: runtime kernel pooling (3-5x "
                         "faster, <1% target projection error -- "
                         "accuracy refinement still in progress)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end smoke test: 400 ms protocol, "
                         "20k kernel-pair cap, output to viz/data_smoke/ "
                         "(does not touch the real viz_data.json)")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        fparams.SECTIONS["stimulus"]["t_epochs"] = (
            [("dark", 0.0, 200.0), ("flicker", 200.0, 400.0)],
            "ms", "chosen", "smoke mode")
        OUT = ROOT / "viz" / "data_smoke"
        OUT.mkdir(parents=True, exist_ok=True)
        print("SMOKE mode: 400 ms protocol, 20k pair cap, "
              f"output -> {OUT}")
    # (re-)derive protocol constants here -- smoke may have shortened
    # them, and the nested stim/record closures read these locals
    _stim_sec = fparams.SECTIONS["stimulus"]
    T_EPOCHS = [tuple(e) for e in _stim_sec["t_epochs"][0]]
    T_END = T_EPOCHS[-1][2]
    STIM_CONTRAST = _stim_sec["stim_contrast"][0]
    I_LUM = _stim_sec["i_lum"][0]
    DRIFT_SPEED = _stim_sec["drift_speed"][0]
    LAM_UM = _stim_sec["lam_um"][0]
    SEED = _stim_sec["seed"][0]
    # smoke defaults to visual-only regions: the CNS-extension assembly
    # (62k+33k+15k cells) dominates the runtime and is irrelevant when
    # smoke-testing the stimulus/pooling/page pathway
    regions_arg = args.regions
    if args.smoke and regions_arg is None:
        regions_arg = "visual_bilateral"
    regions = ({r.strip(): True for r in regions_arg.split(",")}
               if regions_arg else None)

    OUT.mkdir(parents=True, exist_ok=True)
    # region-optional assembly (ffbm.regions): OFF regions are absent
    # from the circuit -- no populations, synapses or kernels built
    circuit, active = freg.build_circuit(regions)
    print(f"regions: {', '.join(active)}")
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids = circuit["r_ids"]
    l_ids = circuit["l_ids"]
    mid_ids = circuit["mid_ids"]
    t45_ids = circuit["t45_ids"]
    side_r = circuit["side_r"]
    side_l = circuit["side_l"]
    side_mid = circuit["side_mid"]
    side_t45 = circuit["side_t45"]
    n_r, n_l, n_mid, n_t45 = (len(r_ids), len(l_ids), len(mid_ids),
                              len(t45_ids))
    t45_type = circuit["t45_type"]
    mid_type = circuit["mid_type"]
    l_type = circuit["l_type"]

    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)
    mid_index = np.full(int(mid_ids.max()) + 1, -1, dtype=np.int64)
    mid_index[mid_ids] = np.arange(n_mid)
    t45_index = np.full(int(t45_ids.max()) + 1, -1, dtype=np.int64)
    t45_index[t45_ids] = np.arange(n_t45)

    e_rl = circuit["e_rl"].copy()
    e_rl["weight"] = e_rl["weight"] * e_rl["sign"]
    e_lm = circuit["e_lm"].copy()
    e_lm["weight"] = e_lm["weight"] * e_lm["sign"]
    e_mt = circuit["e_mt"]

    r_pos = np.array([pp[b] for b in r_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    mid_pos = np.array([pp[b] for b in mid_ids])
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos]).mean(axis=0)
    # head-anchor axis: the LEFT lobe's own eye axis (the bilateral
    # R-mean minus T45-mean degenerates to noise at the midline)
    u_eye = circuit["u_eye_left"]

    # legacy fly-head reference electrodes sit on the LEFT lobe
    r_pos_l, l_pos_l, t45_pos_l = (r_pos[side_r], l_pos[side_l],
                                   t45_pos[side_t45])
    eye_elec = (r_pos_l.mean(axis=0)
                + (float(np.max((r_pos_l - r_pos_l.mean(axis=0)) @ u_eye))
                   + 20.0) * u_eye)
    lam_elec = l_pos_l.mean(axis=0) + 30.0 * u_eye
    med_elec = t45_pos_l.mean(axis=0) + 30.0 * (
        (t45_pos_l.mean(axis=0) - l_pos_l.mean(axis=0))
        / np.linalg.norm(t45_pos_l.mean(axis=0) - l_pos_l.mean(axis=0)))
    electrodes = np.array([eye_elec, lam_elec, med_elec])
    r_max = np.linalg.norm(np.vstack([r_pos, l_pos, mid_pos, t45_pos,
                                      electrodes]) - center, axis=1).max()
    r1 = 1.05 * r_max + 10.0

    def pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        return (np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]))

    group_pairs = {"RL": pairs(e_rl), "LM": pairs(e_lm)}
    for mt in exp005.MID_TYPES:
        group_pairs[f"MT_{mt}"] = pairs(e_mt[mt])
    # kernel-pair cap for the big extra-region groups: the aggregate is
    # linear in the per-edge currents, so a stratified subsample only
    # adds sampling noise (~1/sqrt(n) at n=300k); without the cap the
    # 17-electrode kernel build alone costs ~4.5 h
    KERNEL_PAIR_CAP = 20_000 if args.smoke else 300_000
    rng_k = np.random.default_rng(7)
    reweight = {name: 1.0 for name in group_pairs}
    kernel_keep = {name: None for name in group_pairs}
    for gname, espec in (circuit.get("extra_edges") or {}).items():
        if not espec.get("forward", True):
            continue         # VNC etc.: simulated, no scalp kernels
        pr, po = pairs(espec["table"])
        if len(pr) > KERNEL_PAIR_CAP:
            keep = np.sort(rng_k.choice(len(pr), KERNEL_PAIR_CAP,
                                        replace=False))
            reweight[gname] = len(pr) / KERNEL_PAIR_CAP   # unbiased
            kernel_keep[gname] = keep
            pr, po = pr[keep], po[keep]
        group_pairs[gname] = (pr, po)
    # rhabdome dipole per R toward its OWN eye (mirror the left eye axis
    # across the midline for the right lobe)
    xhat = np.array([1.0, 0.0, 0.0])
    u_right = u_eye - 2.0 * float(np.dot(u_eye, xhat)) * xhat
    u_right = u_right / np.linalg.norm(u_right)
    rh_dir = np.where(side_r[:, None], u_eye[None, :], u_right[None, :])
    photo_pair = (r_pos, r_pos + 23.5 * rh_dir)

    ker = {}
    core_groups = {"RL", "LM"} | {f"MT_{m}" for m in exp005.MID_TYPES}
    for name, (pr, po) in group_pairs.items():
        if name not in core_groups:
            continue   # legacy fly-head reference: visual cascade only
        ker[name] = SealedHeadPairField(
            pr, po, electrodes, center=center, r1=r1, r2=1.3 * r1,
            sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA)
    ker["PHOTO"] = SealedHeadPairField(
        photo_pair[0], photo_pair[1], electrodes, center=center, r1=r1,
        r2=1.3 * r1, sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA)

    # thought-experiment channels (exp010/011): the network magnified
    # inside a human 4-layer head (brain/CSF/skull/scalp), electrode
    # ARRAY on the scalp (channel 0 on the eye-side anchor + a
    # quasi-uniform Fibonacci cover, sorted by angle from it). The
    # network is placed occipitally: the T4/T5 (output/"cortex") end of
    # the cascade is pinned near the inner skull wall, like the real
    # visual cortex at the occipital pole.
    # exp015: BOTH lobes in native geometry span 692 um -> 277 mm at
    # x400, which does NOT fit; fit_scale_shift picks the largest scale
    # that fits after placement (pole-pinned for the elongated single
    # lobe, centered with the inter-eye axis on a diameter for the
    # bilateral V: x400 vs ~x200).
    R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
    SIGMAS = (0.33, 1.79, 0.013, 0.33)
    N_SCALP_ELEC = 17

    u_occ = t45_pos.mean(axis=0) - center     # toward the T4/T5 junction
    u_occ = u_occ / np.linalg.norm(u_occ)
    u_anchor = -u_occ                          # 0 deg = eye side
    # NOTE: iterate .values() -- the pre-exp015 code unpacked .items()
    # correctly as (name, (pr, po)); a bare `for pr, po in ...items()`
    # silently binds the dict KEY to pr (a str) and the pair-tuple to po
    all_pairs = {**group_pairs, "PHOTO": photo_pair}
    native_pts = np.vstack([p for pair in all_pairs.values()
                            for p in pair])
    SCALE, shift_vec, r_after = vp.fit_scale_shift(native_pts, center,
                                                   u_occ, R_BRAIN)
    margin = 0.98 * R_BRAIN - r_after
    print(f"thought-experiment scale x{SCALE:.0f} (nominal x400), "
          f"junction pinned at the occipital pole "
          f"(shift {np.linalg.norm(shift_vec) / 1000:.1f} mm, "
          f"brain-margin {margin:.0f} um)")

    scaled_pairs = {name: (center + (pr - center) * SCALE + shift_vec,
                           center + (po - center) * SCALE + shift_vec)
                    for name, (pr, po) in {**group_pairs,
                                           "PHOTO": photo_pair}.items()}

    # neck axis (VNC direction) for the page's head form and the
    # electrode neck-exclusion cone
    extra_pops = circuit.get("extra_pops") or {}
    if "VNC" in extra_pops:
        vnc_pos = np.array([pp[b] for b in extra_pops["VNC"]["ids"]])
        neck_dir = vnc_pos.mean(axis=0) - center
        neck_dir = neck_dir / np.linalg.norm(neck_dir)
    else:
        neck_dir = None

    # electrode array on the CAP region only: face and neck cones
    # excluded (params head_model.elec_*_excl_deg); rows sorted by
    # angle from the face axis
    scalp_dirs = vp.scalp_electrode_dirs_capped(
        N_SCALP_ELEC, u_anchor, neck_dir)
    scalp_ang = np.degrees(np.arccos(np.clip(scalp_dirs @ u_anchor, -1, 1)))
    elec_names = None
    if args.elec_layout:
        layout = json.loads(Path(args.elec_layout).read_text(
            encoding="utf-8"))
        layout.pop("Nasion", None)
        layout.pop("Inion", None)
        elec_names = list(layout.keys())
        scalp_dirs = np.array([layout[k] for k in elec_names])
        scalp_ang = np.degrees(np.arccos(
            np.clip(scalp_dirs @ u_anchor, -1, 1)))
        print(f"electrode layout: {len(elec_names)} named channels "
              f"from {args.elec_layout}")

    def pool_cluster(gname, pr_s, po_s, k):
        """Cluster the group's (pre, post) 6D coordinates into k
        representatives; returns the pool_info entry."""
        C6 = np.hstack([pr_s, po_s])
        train = np.random.default_rng(11).choice(
            len(pr_s), min(50_000, len(pr_s)), replace=False)
        cent, _ = kmeans2(C6[train], k, minit="points", iter=10, seed=13)
        labels = np.empty(len(pr_s), dtype=np.int64)
        c22 = (cent ** 2).sum(1)
        for s0 in range(0, len(pr_s), 50_000):
            sl = slice(s0, min(s0 + 50_000, len(pr_s)))
            d2 = ((C6[sl] ** 2).sum(1)[:, None] + c22[None, :]
                  - 2.0 * C6[sl] @ cent.T)
            labels[sl] = d2.argmin(1)
        counts = np.bincount(labels, minlength=len(cent))
        keepc = np.nonzero(counts)[0]
        remap = np.full(len(cent), -1, dtype=np.int64)
        remap[keepc] = np.arange(len(keepc))
        labels = remap[labels]
        sum_pr = np.zeros((len(keepc), 3))
        sum_po = np.zeros((len(keepc), 3))
        np.add.at(sum_pr, labels, pr_s)
        np.add.at(sum_po, labels, po_s)
        order = np.argsort(labels, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts[keepc])[:-1]])
        return {"rep_pr": sum_pr / counts[keepc, None],
                "rep_po": sum_po / counts[keepc, None],
                "order": order, "starts": starts,
                "k": len(keepc), "mult": 1.0}

    def rebuild_group_kernel(gname):
        """Rebuild all 45 electrode coefficient rows for one pooled
        group (after an accuracy-driven k increase)."""
        pi = pool_info[gname]
        coef_scalp[gname] = four_sphere_rows(
            pi["rep_pr"], pi["rep_po"], scalp_elec,
            center=center, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL,
            r4=R_SCALP, sigma1=SIGMAS[0], sigma2=SIGMAS[1],
            sigma3=SIGMAS[2], sigma4=SIGMAS[3]) * reweight.get(gname, 1.0)

    # ---- runtime kernel pooling ----
    # The 4-sphere field is smooth in dipole position: cluster each
    # group's (pre, post) 6D coordinates into ~sqrt(n) representatives,
    # build the kernel ONLY for the representatives, and apply it to the
    # pooled per-cluster currents.  Validated in-run against an exact
    # single-electrode coefficient (see pool report at the end).
    pool_info = {}
    # NOTE: kernel pooling is experimental -- in-run validation found a
    # group with ~24% projection error and the refine path needs work.
    # Enable explicitly with --pool; default exports stay exact.
    if args.pool:
        from scipy.cluster.vq import kmeans2
        for gname, (pr_s, po_s) in scaled_pairs.items():
            n = len(pr_s)
            k = int(np.clip(round(np.sqrt(n) * 7.0), 400, 4000))
            if n < 2 * k:
                continue                        # small group: exact
            pool_info[gname] = pool_cluster(gname, pr_s, po_s, k)
        n_rep = sum(i["k"] for i in pool_info.values())
        print(f"kernel pooling: {len(pool_info)}/{len(scaled_pairs)} "
              f"groups -> {n_rep} representative dipoles "
              f"(vs {sum(len(p[0]) for p in scaled_pairs.values())} edges)",
              flush=True)

    import time as _time
    _t0 = _time.time()
    # all electrodes in one batched call per group: shared fg table +
    # thread-parallel per-electrode series (four_sphere_rows)
    scalp_elec = center + 0.985 * R_SCALP * scalp_dirs       # (S, 3)
    coef_scalp = {}
    for name, (pr_s, po_s) in scaled_pairs.items():   # shift included
        pi = pool_info.get(name)
        pr_use, po_use = ((pi["rep_pr"], pi["rep_po"]) if pi
                          else (pr_s, po_s))
        coef_scalp[name] = four_sphere_rows(
            pr_use, po_use, scalp_elec,
            center=center, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL,
            r4=R_SCALP, sigma1=SIGMAS[0], sigma2=SIGMAS[1],
            sigma3=SIGMAS[2], sigma4=SIGMAS[3]) * reweight.get(name, 1.0)
        print(f"scalp kernel {name}: {coef_scalp[name].shape[1]} dipoles "
              f"({_time.time() - _t0:.0f} s)", flush=True)
    print(f"scalp kernels built (S=400, 4-sphere, {len(scalp_dirs)} electrodes, "
          f"{_time.time() - _t0:.0f} s)")
    print("kernel rows per group: "
          + ", ".join(f"{k}={v.shape[1]}" for k, v in coef_scalp.items()))

    # exact single-electrode coefficient: pooling accuracy reference
    pool_exact0 = None
    if pool_info:
        elec0 = center + 0.985 * R_SCALP * scalp_dirs[0]
        pool_exact0 = {
            name: FourSpherePairField(
                pr_s, po_s, elec0[None, :], center=center, r1=R_BRAIN,
                r2=R_CSF, r3=R_SKULL, r4=R_SCALP, sigma1=SIGMAS[0],
                sigma2=SIGMAS[1], sigma3=SIGMAS[2],
                sigma4=SIGMAS[3]).coef[0] * reweight.get(name, 1.0)
            for name, (pr_s, po_s) in scaled_pairs.items()}
        print("pooling reference (electrode 0, exact) built",
              flush=True)

    # ---- simulation via the shared pipeline (ffbm.pipeline) ----
    # naturalistic stimulus: 1/f flicker, drifting 1/f texture along the
    # T4/T5 preferred axis (hex regression), DS-scale bandpassed texture;
    # reverse epochs replay the forward movie exactly backwards
    from ffbm import data as fdata
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]
    a1 = np.array([1.0, 0.0, 0.0]) - u_eye * u_eye[0]
    a1 /= np.linalg.norm(a1)
    b1 = np.cross(u_eye, a1)
    # hex-axis regression on LEFT-lobe Mi1 only: the two lobes' hex axes
    # are mirrored in physical x and would cancel in a joint fit
    mi1_ids = mid_ids[(mid_type == "Mi1") & side_mid]
    mi1_pos = np.array([pp[b] for b in mi1_ids])

    def hex_dir(hex_series):
        h = np.array([hex_series.get(b, np.nan) for b in mi1_ids])
        ok = np.isfinite(h)
        X = np.stack([mi1_pos[ok] @ a1, mi1_pos[ok] @ b1], axis=1)
        coef, *_ = np.linalg.lstsq(X, h[ok], rcond=None)
        d = coef[0] * a1 + coef[1] * b1
        return d / np.linalg.norm(d)

    u_dir, v_dir = hex_dir(hex1), hex_dir(hex2)
    e_ds = (np.cos(np.radians(-135.0)) * u_dir
            + np.sin(np.radians(-135.0)) * v_dir)
    e_ds /= np.linalg.norm(e_ds)
    x_r = r_pos @ e_ds

    rng_s = np.random.default_rng(SEED)

    def one_over_f(n, exponent=1.0):
        x = rng_s.standard_normal(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        y = np.fft.irfft(np.fft.rfft(x) / f ** (exponent / 2.0), n)
        return y / y.std()

    n_ms = int(T_END)
    flicker = one_over_f(n_ms)
    n_tex = 8192
    span = (x_r.max() - x_r.min()) + DRIFT_SPEED * T_END + 4 * LAM_UM
    texture = one_over_f(n_tex)
    kx = np.fft.rfftfreq(n_tex, d=span / n_tex)
    band = np.fft.rfft(rng_s.standard_normal(n_tex))
    lo, hi = 1.0 / 45.0, 1.0 / 15.0
    amp = np.where((kx >= lo) & (kx <= hi), 1.0 / np.maximum(kx, lo), 0.0)
    texture_bp = np.fft.irfft(band * amp, n_tex)
    texture_bp /= texture_bp.std()
    tex_dx = span / n_tex

    def tex_at(x, table):
        idx = np.clip(((x - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return table[idx]

    def epoch_of(t):
        for name, t0, t1 in T_EPOCHS:
            if t0 <= t < t1:
                return name
        return "dark"

    def luminance(t):
        """Per-R luminance (1 = dark base); non-negative by construction."""
        ep = epoch_of(t)
        one = np.ones(n_r)
        if ep == "flicker":
            return np.maximum(one + STIM_CONTRAST * flicker[min(int(t),
                                                                n_ms - 1)],
                              0.05)
        if ep in ("drift+", "drift-"):
            t0 = 4500.0
            s = t - t0 if ep == "drift+" else 1500.0 - (t - 6000.0)
            return np.maximum(one + STIM_CONTRAST * tex_at(
                x_r - DRIFT_SPEED * s, texture), 0.05)
        if ep in ("band+", "band-"):
            t0 = 7500.0
            s = t - t0 if ep == "band+" else 1500.0 - (t - 9000.0)
            return np.maximum(one + STIM_CONTRAST * tex_at(
                x_r - DRIFT_SPEED * s, texture_bp), 0.05)
        return one

    # ---- alternative visual input: grayscale video -> ommatidia ----
    # mapping config catalogue: viz/data/visual_inputs.json; the frames
    # are blurred (ommatidial PSF), percentile-normalized, then sampled
    # at each photoreceptor's eye-plane coordinate -> per-R luminance
    visual_meta = {"id": args.visual_input, "kind": "line_scan",
                   "label": "自然协议（1/f 闪烁+漂移）"}
    video_frames = None
    if args.visual_input != "natural_1d":
        vi_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cfg = None
        if vi_path.exists():
            cat = json.loads(vi_path.read_text(encoding="utf-8"))
            cfg = next((v for v in cat["inputs"]
                        if v["id"] == args.visual_input), None)
        if cfg is None or cfg.get("kind") != "video":
            print(f"visual input '{args.visual_input}' not found or not "
                  "kind=video -- falling back to the natural 1/f protocol")
        else:
            frames = np.load(ROOT / "viz" / cfg["source"])
            if frames.ndim == 4:               # RGB(A) frames -> luma
                frames = (frames[..., 0] * 0.2126
                          + frames[..., 1] * 0.7152
                          + frames[..., 2] * 0.0722)
            from scipy.ndimage import gaussian_filter
            blur = float(cfg.get("blur_px", 0.0))
            if blur > 0:                       # ommatidial PSF blur
                frames = np.stack([gaussian_filter(
                    f.astype(np.float64), blur) for f in frames])
            lo_p, hi_p = np.percentile(frames, [1, 99])
            frames = np.clip((frames - lo_p)
                             / max(float(hi_p - lo_p), 1e-9), 0.0, 1.0)
            H, W = frames.shape[1:3]
            x_i = r_pos @ a1                   # eye-plane coordinates
            y_i = r_pos @ b1
            # percentile extent (not min/max): a few far outliers must
            # not stretch the mapping and leave the grid mostly empty
            mg = float(cfg.get("margin", 0.05))
            x_lo, x_hi = np.percentile(x_i, [1, 99])
            y_lo, y_hi = np.percentile(y_i, [1, 99])
            x_pad = (x_hi - x_lo) * mg
            y_pad = (y_hi - y_lo) * mg
            pxi = np.clip(np.round(
                (x_i - (x_lo - x_pad))
                / max((x_hi + x_pad) - (x_lo - x_pad), 1e-9)
                * (W - 1)).astype(int), 0, W - 1)
            pyi = np.clip(np.round(
                (y_i - (y_lo - y_pad))
                / max((y_hi + y_pad) - (y_lo - y_pad), 1e-9)
                * (H - 1)).astype(int), 0, H - 1)
            lo_l = float(cfg.get("lo", 0.05))
            hi_l = float(cfg.get("hi", 3.0))
            S = lo_l + (hi_l - lo_l) * frames[:, pyi, pxi]  # (T, n_r)
            fps_v = float(cfg.get("fps", 30.0))

            def luminance(t):
                return S[min(int(t / 1000.0 * fps_v), S.shape[0] - 1)]

            video_frames = frames              # kept for preview export
            visual_meta = {"id": cfg["id"], "kind": "video",
                           "label": cfg.get("label", cfg["id"]),
                           "fps": fps_v}
            print(f"visual input: {cfg['id']} -> {S.shape[0]} frames "
                  f"@ {fps_v} fps, {len(pxi)} photoreceptors sampled")

    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    n_field = int(T_END / DT) // 2
    pool_err = []
    phi = np.zeros((n_field, 3))
    phi_scalp = np.zeros((n_field, len(scalp_dirs)))
    rate = {k: np.zeros(n_field) for k in
            ("R", "L", "MID", "T4", "T5")}
    stim = np.zeros(n_field)
    cal = dict(fp.CAL)

    extra_post = {g: s["post"] for g, s in
                  (circuit.get("extra_edges") or {}).items()}

    # ---- forward application: buffer per-record currents and project
    # with one float32 GEMM per CHUNK records.  Per-record (S,N)@(N,)
    # GEMVs are bandwidth-bound; batching keeps the coefficient rows in
    # cache and lets BLAS stream the reads.  Pooling (reduceat) moves to
    # flush time so a mid-run pool refinement stays consistent with the
    # buffered records (they store kept per-edge y, pooled only at
    # flush with the then-current clustering).
    CHUNK = 256
    gnames = list(group_pairs) + ["PHOTO"]
    coef_f32 = {name: coef_scalp[name].astype(np.float32)
                for name in gnames}
    # row-major (CHUNK, N): records append CONTIGUOUS rows (a strided
    # column write costs a cache miss per dipole); flush feeds the
    # transposed view straight into the GEMM (BLAS native, no copy)
    ybuf = {name: np.zeros((CHUNK, coef_f32[name].shape[1]), np.float32)
            for name in gnames}
    print(f"forward buffers: {CHUNK} records x "
          f"{sum(b.shape[1] for b in ybuf.values())} dipoles "
          f"({sum(b.nbytes for b in ybuf.values()) / 1e6:.0f} MB)",
          flush=True)
    jbuf = 0

    def flush_scalp(j0, c):
        for name in gnames:
            Y = ybuf[name][:c]                       # (c, N) view
            pi = pool_info.get(name)
            if pi is not None:                       # pool over dipoles
                Y = np.add.reduceat(Y, pi["starts"], axis=1)
            phi_scalp[j0:j0 + c] += (coef_f32[name] @ Y.T).T
        phi_scalp[j0:j0 + c] *= 1e-12

    def record(j, k, t, st, inc_f, sp):
        nonlocal jbuf
        syn, pops = st["syn"], st["pops"]
        i_photo = cal["I_R_BASE"] + inc_f
        mech = st["mech"]

        def y_of(name):
            if name == "PHOTO":
                return i_photo
            # graded pools and conductance MT both expose per-edge
            # v-dependent currents; legacy current-mode RL/LM return y
            if name == "RL":
                return syn[name].edge_currents(pops["L"].v)
            if name == "LM":
                return syn[name].edge_currents(pops["MID"].v)
            post = extra_post.get(name, "T45")
            return syn[name].edge_currents(pops[post].v)

        ys = {}
        for name in gnames:
            y = y_of(name)
            keep = kernel_keep.get(name)
            if keep is not None:
                y = y[keep]
            ys[name] = y
            ybuf[name][jbuf] = y
        # fly-head reference channels: one GEMV per core group
        acc3 = np.zeros(3)
        for name in ker:
            acc3 += ker[name].coef @ ys[name]
        phi[j] = acc3 * 1e-12
        # in-run pooling accuracy check (two records per run): exact
        # single-electrode row vs pooled projection; refine failing
        # groups in place -- flush-time reduceat picks the new
        # clustering up for every record still in the buffer
        if pool_exact0 is not None and j in (0, n_field // 2):
            pool_err.clear()
            for name, pi in pool_info.items():
                y = ys[name]
                ex = float(pool_exact0[name] @ y)
                pl = float(coef_scalp[name][0]
                           @ np.add.reduceat(y[pi["order"]], pi["starts"]))
                rel = abs(ex - pl) / max(abs(ex), 1e-30)
                pool_err.append(rel)
                if rel > 0.01:
                    pi["k"] = min(int(pi["k"] * 3), 20000)
                    pi.update(pool_cluster(
                        name, scaled_pairs[name][0],
                        scaled_pairs[name][1], pi["k"]))
                    rebuild_group_kernel(name)
                    coef_f32[name] = coef_scalp[name].astype(np.float32)
                    print(f"pooling: group {name} refined to k={pi['k']} "
                          f"(rel err {rel * 100:.1f}%)", flush=True)
        jbuf += 1
        if jbuf == CHUNK:
            flush_scalp(j - CHUNK + 1, CHUNK)
            jbuf = 0
        if mech:   # graded R/L: display their release rates (%) instead
            rate["R"][j] = float(st["r_release"](
                pops["R"].v, cal["R_RELEASE_MAP_MV"]).mean()) * 100.0
            rate["L"][j] = float(st["l_release"](
                pops["L"].v, cal["L_RELEASE_MAP_MV"]).mean()) * 100.0
        else:
            rate["R"][j] = sp["R"].sum() * 1000.0 / n_r
            rate["L"][j] = sp["L"].sum() * 1000.0 / n_l
        rate["MID"][j] = sp["MID"].sum() * 1000.0 / n_mid
        rate["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        rate["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()
        stim[j] = float(np.mean(luminance(t)))

    print(f"simulating {T_END / 1000:.1f} s ...")
    _t1 = _time.time()
    fp.simulate(circuit, cal, lambda t: I_LUM * (luminance(t) - 1.0),
                SEED, T_END, on_sample=record)
    if jbuf:
        flush_scalp(n_field - jbuf, jbuf)
    print(f"biology loop: {_time.time() - _t1:.0f} s "
          f"({n_field} records)")
    np.save(OUT / "_debug_phi_scalp.npy", phi_scalp * 1.7)
    np.save(OUT / "_debug_phi.npy", phi)
    print(f"phi_scalp checksum: L2={np.linalg.norm(phi_scalp) * 1.7:.6e} "
          f"(baseline A/B file: {OUT / '_debug_phi_scalp.npy'})")

    # absolute-amplitude calibration: the return-current geometry was
    # discriminated to NEURITE (exp013): all scalp amplitudes scale by
    # the neurite/mean-position kernel-norm ratio 1.7. The three
    # fly-head internal electrodes stay in raw pipeline units.
    phi_scalp = phi_scalp * 1.7

    # ---- background human EEG + SNR (shared helpers, unit-tested) ----
    bg = vp.generate_background_eeg(scalp_dirs, u_anchor, n_field)
    # SNR reads the [1500, 4500) sample window -- shorter runs (smoke)
    # skip it; the page treats a missing snr as "not computed"
    snr = None
    if n_field >= 4500:
        snr = vp.snr_metrics(phi_scalp.T * 1e6, bg, 1500, 4500)
        snr["best_elec_deg"] = round(float(scalp_ang[snr["best_elec"]]), 1)
        snr["best_elec_name"] = (elec_names[snr["best_elec"]]
                                 if elec_names else None)
        best_tag = (snr["best_elec_name"]
                    if snr["best_elec_name"]
                    else f"{snr['best_elec_deg']:.0f} deg")
        print(f"background EEG: fly signal {snr['sig_uv']:.2f} uV vs bg "
              f"{snr['bg_uv']:.2f} uV at {best_tag} -> "
              f"d'=2 needs ~{snr['k_for_dprime2']} trials "
              f"(band {snr['band_hz'][0]:.0f}-{snr['band_hz'][1]:.0f} Hz: "
              f"~{snr['k_for_dprime2_band']})")
    if pool_err:
        print(f"kernel pooling accuracy: max rel err "
              f"{max(pool_err) * 100:.3f}% over {len(pool_err)} checks "
              "(exact electrode-0 reference)")

    # ---- stimulus view frames (human video | fly-eye sampling) ----
    stim_view_meta = None
    if video_frames is not None:
        from scipy.ndimage import zoom
        ph, pw = 72, 96
        fps_v = visual_meta["fps"]
        human_u8 = np.zeros((video_frames.shape[0], ph, pw), np.uint8)
        for f in range(video_frames.shape[0]):
            z = zoom(video_frames[f],
                     (ph / video_frames.shape[1],
                      pw / video_frames.shape[2]), order=1)
            human_u8[f] = (z * 255).astype(np.uint8)
        gh, gw = 72, 96
        gyi = np.clip(pyi * gh // H, 0, gh - 1)
        gxi = np.clip(pxi * gw // W, 0, gw - 1)
        # fly-eye view: pooled per-photoreceptor luminance binned on the
        # eye-plane grid (sparse by nature -- the ommatidia lattice);
        # smoothing/interpolation is the PAGE's rendering choice
        fly_u8 = np.zeros((S.shape[0], gh, gw), np.uint8)
        for f in range(S.shape[0]):
            acc = np.zeros((gh, gw))
            np.add.at(acc, (gyi, gxi), S[f])
            nz = acc[acc > 0]
            scale = np.percentile(nz, 95) if nz.size else 1.0
            fly_u8[f] = (np.clip(acc / max(float(scale), 1e-9),
                                 0.0, 1.0) * 255).astype(np.uint8)
        (OUT / "stim_frames.bin").write_bytes(
            human_u8.tobytes() + fly_u8.tobytes())
        stim_view_meta = {"n": int(S.shape[0]), "human": [ph, pw],
                          "fly": [gh, gw], "fps": fps_v}
        print(f"stim view frames: {OUT / 'stim_frames.bin'} "
              f"({(OUT / 'stim_frames.bin').stat().st_size / 1e6:.1f} MB)")

    # ---- assemble viz data ----
    def pts(arr):
        return np.round(arr - center, 1).tolist()

    rng2 = np.random.default_rng(5)
    flow = []
    for name, (pr, po) in group_pairs.items():
        grp_layer = {"RL": 0, "LM": 1}.get(name, 2)
        n_sample = int(N_FLOW_EDGES * len(pr) / sum(
            len(v[0]) for v in group_pairs.values())) + 1
        idx = rng2.choice(len(pr), size=min(n_sample, len(pr)),
                          replace=False)
        for i in idx:
            flow.append([*np.round(pr[i] - center, 1).tolist(),
                         *np.round(po[i] - center, 1).tolist(), grp_layer])

    # phi in uV, rounded
    phi_uv = np.round(phi * 1e6, 2)
    is_video = visual_meta["kind"] == "video"
    data = {
        "meta": {
            "t_end_ms": T_END,
            "visual_input": visual_meta,
            "epochs": (
                [{"name": "video", "t0": 0.0, "t1": T_END,
                  "color": "#94a3b8"}] if is_video else
                [{"name": name, "t0": t0, "t1": t1,
                  "color": EPOCH_COLORS[name]}
                 for name, t0, t1 in T_EPOCHS]),
            "epoch_labels": (
                {"video": "视频输入"} if is_video else
                {"dark": "黑暗", "flicker": "1/f 闪烁",
                 "drift+": "纹理漂移 →", "drift-": "纹理漂移 ←",
                 "band+": "带通纹理 →", "band-": "带通纹理 ←"}),
            "head_r_um": round(r1, 1),
            "scalp": {"model": "4sphere", "place": "occipital",
                      "scale": round(SCALE, 1),
                      "scale_nominal": 400,
                      "radii_um": [7.8e4, 8.0e4, 8.5e4, 9.2e4],
                      "sigmas": list(SIGMAS),
                      "shift_um": [round(float(x), 1) for x in shift_vec],
                      "n_elec": len(scalp_dirs),
                      "elec_dist_um": 0.985 * 9.2e4,
                      "elec_dir": [round(float(x), 4) for x in u_anchor],
                      "neck_dir": (None if neck_dir is None else
                                   [round(float(x), 4) for x in neck_dir]),
                      "elec_dirs": [[round(float(x), 4) for x in d]
                                    for d in scalp_dirs],
                      "stim_view": stim_view_meta,
                      "elec_deg": [round(float(a), 1)
                                   for a in scalp_ang],
                      "elec_names": elec_names},
            "regions": active,
            "amplitude_calibration": {"geometry": "neurite",
                                      "factor": 1.7,
                                      "source": "exp013 discrimination "
                                                "(8:0:0 over 25 configs)"},
            "calibration": {**cal, "source":
                            "ffbm.pipeline CAL (round-3 winner; see "
                            "scripts/outputs/working_point_calibration_r3.json)"},
            "bg_eeg": {**vp.BG_DEFAULTS, "snr": snr},
            "layers": [
                {"name": "R1-R6 光感受器", "color": "#a855f7", "n": n_r},
                {"name": "L1-L3 板层", "color": "#38bdf8", "n": n_l},
                {"name": "Mi/Tm 髓质中间神经元", "color": "#4ade80",
                 "n": n_mid},
                {"name": "T4", "color": "#f87171", "n": int(is_t4.sum())},
                {"name": "T5", "color": "#fb923c", "n": int(is_t5.sum())},
            ],
            "electrodes": [
                {"name": "眼表面", "color": "#ef4444"},
                {"name": "板层", "color": "#f59e0b"},
                {"name": "髓质", "color": "#22c55e"},
            ],
        },
        "positions": None,      # filled after extra-region layers below
        "electrodes_pos": np.round(electrodes - center, 1).tolist(),
        "flow_edges": flow,
        "t_ms": np.arange(n_field).tolist(),
        "stim": np.round(stim, 2).tolist(),
        "rates": {k: np.round(v, 1).tolist() for k, v in rate.items()},
        "phi_uV": phi_uv.tolist(),
        "phi_scalp_all_uV": np.round(phi_scalp.T * 1e6, 3).tolist(),
        "phi_scalp_bg_uV": np.round(bg, 2).tolist(),
    }
    # extra-region layers for the point cloud. VNC IS shown (its true
    # native geometry makes it stick out of the head sphere at x202 --
    # the honest picture of "the fly CNS does not fit in a human head");
    # it only stays out of the kernels/fit (forward=False)
    layer_names = {"VPN": "VPN 投射神经元", "CB": "中央脑目标(exp016)",
                   "OLR": "其余视叶", "CEN": "中央脑",
                   "VNC": "腹索 VNC（仅动力学·不投影）"}
    layer_colors = {"VPN": "#f472b6", "CB": "#c084fc", "OLR": "#8b9dc3",
                    "CEN": "#fbbf24", "VNC": "#5eead4"}
    positions = [pts(r_pos), pts(l_pos), pts(mid_pos),
                 pts(t45_pos[is_t4]), pts(t45_pos[is_t5])]
    for pname, spec in (circuit.get("extra_pops") or {}).items():
        pos_x = np.array([pp[b] for b in spec["ids"]])
        positions.append(np.round(pos_x - center, 1).tolist())
        data["meta"]["layers"].append(
            {"name": layer_names.get(pname, pname),
             "color": layer_colors.get(pname, "#8b9dc3"),
             "n": len(pos_x), "static": True})

    # cervical-connective axon lines: for a sample of descending
    # (CEN->VNC) and ascending (VNC->CEN) neurons, soma -> centroid of
    # their ACTUAL postsynaptic cells on the other side (data-driven
    # approximation of the axon path through the neck)
    conn = {"cen_vnc": [], "vnc_cen": []}
    extra_pops = circuit.get("extra_pops") or {}
    if {"CEN", "VNC"} <= set(extra_pops) \
            and {"VNC_C", "ASC_R"} <= set(circuit.get("extra_edges") or {}):
        pos_of = {}
        for pop_spec in extra_pops.values():
            for b in pop_spec["ids"]:
                pos_of[b] = np.asarray(pp[b], dtype=np.float64)
        rng_c = np.random.default_rng(11)
        for gname, key in (("VNC_C", "cen_vnc"), ("ASC_R", "vnc_cen")):
            e = circuit["extra_edges"][gname]["table"]
            gb = e.groupby("body_pre")["body_post"].apply(
                lambda s: s.to_numpy())
            pres = np.array(gb.index.to_numpy())
            if len(pres) > 200:
                pres = rng_c.choice(pres, 200, replace=False)
            for pb in pres:
                tgt = np.mean([pos_of[b] for b in gb[pb]], axis=0)
                conn[key].append(np.round(np.concatenate(
                    [pos_of[pb] - center, tgt - center]), 1).tolist())
        print(f"cervical connective lines: "
              f"{len(conn['cen_vnc'])} descending + "
              f"{len(conn['vnc_cen'])} ascending")
    data["connective"] = conn
    data["positions"] = positions
    path = OUT / "viz_data.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
