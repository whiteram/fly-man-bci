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
    ap.add_argument("--chem-input", type=str, default=None,
                    help="chemosensory input id from viz/data/"
                         "chem_inputs.json (odor/taste pulse trains "
                         "injected into the ORN/GRN regions; requires "
                         "the olfactory/gustatory regions ON)")
    ap.add_argument("--pool", action="store_true",
                    help="EXPERIMENTAL: runtime kernel pooling (3-5x "
                         "faster, <1% target projection error -- "
                         "accuracy refinement still in progress)")
    ap.add_argument("--gpu", action="store_true",
                    help="P3: run the biology loop on the GPU "
                         "(ffbm.gpu, CuPy). Bitwise-identical "
                         "trajectories; the scalp/fly-head forward GEMMs "
                         "run on cublas so phi differs from the CPU "
                         "export at float rounding (~1e-6 relative), "
                         "and --pool is not supported")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end smoke test: 400 ms protocol, "
                         "20k kernel-pair cap, output to viz/data_smoke/ "
                         "(does not touch the real viz_data.json)")
    ap.add_argument("--t-end", type=float, default=None,
                    help="override the trial length, ms (single dark "
                         "epoch; small-scale iteration -- chem pulse "
                         "times must fit inside)")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the staged build caches (ffbm.cache) and"
                    "rebuild the circuit and forward kernels from scratch")
    ap.add_argument("--out", type=str, default=None,
                    help="override the output directory (default "
                    "viz/data, smoke: viz/data_smoke)")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="scale every OU noise sigma (bci/arousal "
                    "state manipulation: quiet < 1 < active)")
    ap.add_argument("--std-gates", type=str, default=None,
                    help="opt-in short-term depression on extra edge "
                    "groups, 'GRP:U,tau_rec_ms;...' (bci/mmdev; "
                    "runtime mutation, no cache impact)")
    ap.add_argument("--base-scale", type=float, default=1.0,
                    help="scale every *_BASE bias current (bci/"
                    "gainstate: the working point IS a state axis -- "
                    "moves spontaneous firing AND stimulus gain)")
    ap.add_argument("--plastic-mb", action="store_true",
                    help="split KC->MBON synapses out of CEN_C into a "
                    "plastic edge group (bci/condit; runtime surgery, "
                    "salted kernel key -- one rebuild, then cached)")
    ap.add_argument("--plastic-window", type=str, default=None,
                    help="reinforcement gate window 'a,b' ms (mod=1 "
                    "inside; the DAN-drive proxy)")
    ap.add_argument("--plastic-lr", type=float, default=0.0015,
                    help="plasticity learning rate (per step, gated)")
    ap.add_argument("--plastic-tau-w", type=float, default=0.0,
                    help="slow homeostatic recovery of plastic w_scale "
                         "toward 1, ms (0 = frozen weights; extinction, "
                         "bci/condit2)")
    ap.add_argument("--plastic-state-in", type=str, default=None,
                    help="npz written by --plastic-state-out: continue "
                         "the session from these KCM weights (cross-"
                         "trial memory; GPU only)")
    ap.add_argument("--plastic-state-out", type=str, default=None,
                    help="write KCM w_scale + edge-id checksum after the "
                         "run (GPU only)")
    ap.add_argument("--al-gain", type=float, default=None,
                    help="AL-independence surgery: split AL->KC (PN->"
                         "Kenyon) rows out of CEN_C into group ALK at "
                         "this ABSOLUTE gain, escaping the chem working "
                         "point's recurrence damping (true odor -> KC "
                         "transmission; bci/condit3)")
    ap.add_argument("--gain-scale", type=str, default=None,
                    help="runtime pathway-gain modulation, 'GRP=f[,"
                         "GRP=f...]': multiply the named extra edge "
                         "groups' g_unit by f (resolved absolute; not "
                         "in any cache key -- g_unit is a runtime "
                         "scalar; bci/attend attention model)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the trial RNG seed (params: "
                    "stimulus.seed): varies delay jitter + OU background "
                    "noise; circuit/kernel caches are seed-independent")
    args = ap.parse_args()
    if (args.plastic_state_in or args.plastic_state_out) and (
            not args.plastic_mb or not args.gpu):
        ap.error("--plastic-state-in/out requires --plastic-mb --gpu")
    global OUT
    if args.smoke:
        fparams.SECTIONS["stimulus"]["t_epochs"] = (
            [("dark", 0.0, 200.0), ("flicker", 200.0, 400.0)],
            "ms", "chosen", "smoke mode")
        OUT = ROOT / "viz" / "data_smoke"
    if args.t_end:
        fparams.SECTIONS["stimulus"]["t_epochs"] = (
            [("dark", 0.0, float(args.t_end))], "ms", "chosen",
            f"short trial (--t-end {args.t_end:g})")
    if args.out:
        OUT = Path(args.out)
    if args.smoke or args.out:
        OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
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
    if args.seed is not None:
        SEED = args.seed          # trial-level variability override
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
    # from the circuit -- no populations, synapses or kernels built.
    # The registry default keeps non-visual regions OFF (cheap dev
    # default): say so loudly, a silent visual-only "full" export
    # shrinks the page's cell display to the optic lobes
    USE_CACHE = not args.no_cache
    from ffbm import cache as fcache
    ckey = fcache.circuit_key(regions) if USE_CACHE else None
    if USE_CACHE and fcache.have("circuit", ckey):
        circuit = fcache.load_circuit(ckey)
        cfg = dict(freg.DEFAULT_REGIONS)
        if regions:
            cfg.update({k: bool(v) for k, v in regions.items()})
        active = [n for n, on in cfg.items() if on]
    else:
        circuit, active = freg.build_circuit(regions)
        if USE_CACHE:
            fcache.save_circuit(circuit, ckey)
    _off = [r for r in freg.known_regions() if r not in active]
    print(f"regions: ON  = {', '.join(active) or '(none)'}")
    print(f"regions: OFF = {', '.join(_off) or '(none)'}"
          + ("   <-- pass --regions to enable" if _off else ""),
          flush=True)
    if args.plastic_mb:
        # runtime surgery (bci/condit): move KC->MBON pairs out of the
        # CEN_C recurrence table into a dedicated PLASTIC group; the
        # circuit cache stays untouched but the kernel key is salted so
        # the forward kernels rebuild for the split topology
        from ffbm import data as _fdata
        _ann = _fdata.load_annotations()
        _cls = _ann["class"].fillna("")
        _sup = _ann["superclass"].fillna("")
        _soma = _fdata.neuron_positions(_ann)
        _cen_raw = np.array(sorted(
            _ann.loc[(_sup.str.startswith("cb_")
                      | (_sup == "descending_neuron"))
            & _ann["bodyId"].isin(_soma), "bodyId"].astype(int)),
            dtype=np.int64)
        _cen_ids = circuit["extra_pops"]["CEN"]["ids"]
        _rmap = dict(zip(_cen_raw.tolist(), _cen_ids.tolist()))
        _kc = set(_rmap[int(b)] for b in
                  _ann.loc[_cls == "Kenyon_Cell", "bodyId"]
                  if int(b) in _rmap)
        _mbon = set(_rmap[int(b)] for b in
                    _ann.loc[_cls == "MBON", "bodyId"]
                    if int(b) in _rmap)
        _tab = circuit["extra_edges"]["CEN_C"]["table"]
        _m = _tab["body_pre"].isin(_kc) & _tab["body_post"].isin(_mbon)
        _kcm = _tab[_m].reset_index(drop=True)
        circuit["extra_edges"]["CEN_C"]["table"] = \
            _tab[~_m].reset_index(drop=True)
        circuit["extra_edges"]["KCM"] = {
            "pre": ("CEN",), "post": "CEN", "table": _kcm,
            "tau_s": circuit["extra_edges"]["CEN_C"]["tau_s"],
            "g_unit": circuit["extra_edges"]["CEN_C"]["g_unit"],
            "forward": True,
            "plast": {"lr": float(args.plastic_lr),
                      "tau_ms": 400.0,
                      **({"tau_w_ms": float(args.plastic_tau_w)}
                         if args.plastic_tau_w > 0 else {})}}
        # edge-identity checksum: the synapse reorders rows with
        # lexsort((pre, post)) -- a DETERMINISTIC function of the table
        # content -- so hashing the table's id bytes pins the w_scale
        # ordering; a state file from an invocation with a different KCM
        # edge list is rejected
        import hashlib
        KCM_CHECKSUM = hashlib.md5(np.concatenate([
            _kcm["body_pre"].to_numpy(np.int64),
            _kcm["body_post"].to_numpy(np.int64)]).tobytes()).hexdigest()
        print(f"plastic-mb: KCM split {len(_kcm)} KC->MBON pairs "
              f"from CEN_C (lr={args.plastic_lr}"
              + (f", tau_w={args.plastic_tau_w:g} ms"
                 if args.plastic_tau_w > 0 else "") + ")")
        ckey = (ckey + "+plmb") if ckey is not None else None
    if args.al_gain is not None:
        # AL-independence surgery (bci/condit3): the AL->KC (PN->Kenyon)
        # rows live in the CEN_C recurrence table, so the chem working
        # point (g 0.002) crushes the real odor->KC pathway 500x.  Split
        # them into an ALK feedforward group at an independent ABSOLUTE
        # gain.  AL local recurrence (PN<->LN) stays damped: the minimal
        # step is the feedforward ORN ->(ORN_C, full gain) PN ->(ALK) KC
        # pathway; stability must be calibrated (latch risk).
        if "_rmap" not in locals():
            from ffbm import data as _fdata
            _ann = _fdata.load_annotations()
            _sup = _ann["superclass"].fillna("")
            _soma = _fdata.neuron_positions(_ann)
            _cen_raw = np.array(sorted(
                _ann.loc[(_sup.str.startswith("cb_")
                          | (_sup == "descending_neuron"))
                & _ann["bodyId"].isin(_soma), "bodyId"].astype(int)),
                dtype=np.int64)
            _cen_ids = circuit["extra_pops"]["CEN"]["ids"]
            _rmap = dict(zip(_cen_raw.tolist(), _cen_ids.tolist()))
        _al = set(_rmap[int(b)] for b in
                  _ann.loc[_ann["class"].isin(
                      ("ALPN", "ALLN", "ALIN", "ALON")), "bodyId"]
                  if int(b) in _rmap)
        _kc = set(_rmap[int(b)] for b in
                  _ann.loc[_ann["class"] == "Kenyon_Cell", "bodyId"]
                  if int(b) in _rmap)
        _tab = circuit["extra_edges"]["CEN_C"]["table"]
        _m = _tab["body_pre"].isin(_al) & _tab["body_post"].isin(_kc)
        _alk = _tab[_m].reset_index(drop=True)
        circuit["extra_edges"]["CEN_C"]["table"] = \
            _tab[~_m].reset_index(drop=True)
        circuit["extra_edges"]["ALK"] = {
            "pre": ("CEN",), "post": "CEN", "table": _alk,
            "tau_s": circuit["extra_edges"]["CEN_C"]["tau_s"],
            "g_unit": float(args.al_gain), "forward": True}
        print(f"al-gain: ALK split {len(_alk)} AL->KC pairs from CEN_C "
              f"(gain={args.al_gain:g})", flush=True)
        ckey = (ckey + "+alg") if ckey is not None else None
    mod_fn = None
    if args.plastic_window:
        _wa, _wb = (float(v) for v in args.plastic_window.split(","))

        def mod_fn(t, _a=_wa, _b=_wb):
            return 1.0 if _a <= t < _b else 0.0
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

    # ---- staged kernel cache (ffbm.cache) --------------------------
    # keyed by (circuit key, electrode layout, pair cap, forward code):
    # switching --elec-layout rebuilds ONLY this stage.  --pool bypasses.
    KERNEL_PAIR_CAP = 20_000 if args.smoke else 300_000
    _kkey = None
    _kernel_hit = (USE_CACHE and not args.pool and ckey is not None
                   and fcache.have("kernels",
                                   fcache.kernel_key(ckey, args.elec_layout,
                                                     None, KERNEL_PAIR_CAP)))
    import time as _time                     # the build block below owns
    if _kernel_hit:                          # its own copy; hits need one too
        _kkey = fcache.kernel_key(ckey, args.elec_layout, None,
                                  KERNEL_PAIR_CAP)
        KARR, KSC = fcache.load_kernels(_kkey)
        group_pairs = {g: (KARR[f"gp.{g}.0"], KARR[f"gp.{g}.1"])
                       for g in KSC["gnames"]}
        reweight = {g: float(KSC["rw." + g]) for g in KSC["gnames"]}
        kernel_keep = {g: (KARR["keep." + g] if KARR["keep." + g].size
                           else None) for g in KSC["gnames"]}
        from types import SimpleNamespace
        ker = {g: SimpleNamespace(coef=KARR["fly." + g])
               for g in KSC["flynames"]}
        SCALE, margin = KSC["SCALE"], KSC["margin"]
        shift_vec = KARR["shift_vec"]
        coef_scalp = {g: KARR["coef." + g] for g in KSC["cgnames"]}
        pool_info, pool_exact0 = {}, None
        # mirror of the layout block below (always needed for meta/page)
        R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
        SIGMAS = (0.33, 1.79, 0.013, 0.33)
        N_SCALP_ELEC = 17
        u_occ = t45_pos.mean(axis=0) - center
        u_occ = u_occ / np.linalg.norm(u_occ)
        u_anchor = -u_occ
        _ep = circuit.get("extra_pops") or {}
        if "VNC" in _ep:
            neck_dir = np.array([pp[b] for b in _ep["VNC"]["ids"]]
                                ).mean(axis=0) - center
            neck_dir = neck_dir / np.linalg.norm(neck_dir)
        else:
            neck_dir = None
        scalp_dirs = vp.scalp_electrode_dirs_capped(
            N_SCALP_ELEC, u_anchor, neck_dir)
        scalp_ang = np.degrees(np.arccos(
            np.clip(scalp_dirs @ u_anchor, -1, 1)))
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
    else:

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
        if USE_CACHE and not args.pool:
            _kkey = fcache.kernel_key(ckey, args.elec_layout, None,
                                      KERNEL_PAIR_CAP)
            payload = {"shift_vec": shift_vec, "SCALE": float(SCALE),
                       "margin": float(margin),
                       "gnames": list(group_pairs),
                       "cgnames": list(coef_scalp),
                       "flynames": list(ker)}
            for g, (pr, po) in group_pairs.items():
                payload[f"gp.{g}.0"] = pr
                payload[f"gp.{g}.1"] = po
                _kp = kernel_keep.get(g)          # extra groups may be
                payload[f"keep.{g}"] = (_kp if _kp is not None   # absent
                                        else np.zeros(0, np.int64))
                payload[f"rw.{g}"] = float(reweight.get(g, 1.0))
            for g, c in coef_scalp.items():   # includes PHOTO
                payload[f"coef.{g}"] = c.astype(np.float32)
            for g in ker:
                payload[f"fly.{g}"] = ker[g].coef
            fcache.save_kernels(payload, _kkey)


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
            if frames.dtype == np.uint8:       # display-referred [0,255]
                frames = frames.astype(np.float64) / 255.0
            else:
                frames = frames.astype(np.float64)   # float: expect [0,1]
            if cfg.get("linearize_srgb", True):
                # stored frames are gamma-encoded (sRGB display values);
                # the BT.709 luma weights and the Gaussian PSF below are
                # linear-domain operations -- decode first.  Opt out per
                # input with "linearize_srgb": false for already-linear
                # synthetic stacks.
                frames = np.where(frames <= 0.04045,
                                  frames / 12.92,
                                  ((frames + 0.055) / 1.055) ** 2.4)
            if frames.ndim == 4:               # RGB(A) frames -> BT.709 luma
                frames = (frames[..., 0] * 0.2126
                          + frames[..., 1] * 0.7152
                          + frames[..., 2] * 0.0722)
            from scipy.ndimage import gaussian_filter
            blur = float(cfg.get("blur_px", 0.0))
            if blur > 0:                       # ommatidial PSF blur
                frames = np.stack([gaussian_filter(
                    f.astype(np.float64), blur) for f in frames])
            pct = np.asarray(cfg.get("norm_pct", [1.0, 99.0]), float)
            lo_p, hi_p = np.percentile(frames, pct)
            frames = np.clip((frames - lo_p)
                             / max(float(hi_p - lo_p), 1e-9), 0.0, 1.0)
            H, W = frames.shape[1:3]
            # per-eye mapping (default): mirror the right lobe about the
            # midsagittal plane so BOTH eyes project face-on in the SAME
            # left-eye basis (the two retinas are mirror-symmetric; the
            # raw right-eye projection is edge-on and would smear), then
            # normalize each eye's own 1-99 percentile footprint to the
            # full frame -- each eye samples one complete, equally
            # oriented copy of the image.  eye_map "union" keeps the
            # legacy behaviour: one shared extent over the raw pooled
            # projection, which leaves the frame's middle band
            # unsampled and gives each eye a different horizontal band.
            # (geometry validated in scripts/poc_eye_map_geometry.py)
            if str(cfg.get("eye_map", "per_eye")) in ("per_eye",
                                                      "half_split"):
                r_pos_map = r_pos.copy()
                mid = 0.5 * (r_pos[side_r, 0].mean()
                             + r_pos[~side_r, 0].mean())
                r_pos_map[~side_r, 0] = 2.0 * mid - r_pos_map[~side_r, 0]
                groups = (np.flatnonzero(side_r),
                          np.flatnonzero(~side_r))
            else:
                r_pos_map = r_pos
                groups = (np.arange(n_r),)
            x_i = r_pos_map @ a1                  # eye-plane coordinates
            y_i = r_pos_map @ b1
            # percentile extent (not min/max): a few far outliers must
            # not stretch the mapping and leave the grid mostly empty
            mg = float(cfg.get("margin", 0.05))
            pxf = np.empty(n_r)
            pyf = np.empty(n_r)
            for m in groups:
                x_lo, x_hi = np.percentile(x_i[m], [1, 99])
                y_lo, y_hi = np.percentile(y_i[m], [1, 99])
                x_pad = (x_hi - x_lo) * mg
                y_pad = (y_hi - y_lo) * mg
                pxf[m] = (x_i[m] - (x_lo - x_pad)) / max(
                    (x_hi + x_pad) - (x_lo - x_pad), 1e-9) * (W - 1)
                pyf[m] = (y_i[m] - (y_lo - y_pad)) / max(
                    (y_hi + y_pad) - (y_lo - y_pad), 1e-9) * (H - 1)
            if str(cfg.get("eye_map", "per_eye")) == "half_split":
                # each lobe samples its OWN half of the frame (right
                # lobe -> left half, left lobe -> right half;
                # bci/visualfield monocular-paradigm support)
                pxf[groups[0]] *= 0.5
                pxf[groups[1]] = 0.5 * (W - 1) + pxf[groups[1]] * 0.5
            # float coords -> bilinear sampling: nearest-neighbour
            # aliases at low video resolutions (sub-ommatidial pixels)
            pxf = np.clip(pxf, 0, W - 1)
            pyf = np.clip(pyf, 0, H - 1)
            pxi = np.round(pxf).astype(int)     # ints only for the
            pyi = np.round(pyf).astype(int)     # sparse preview binning
            from scipy.ndimage import map_coordinates
            lo_l = float(cfg.get("lo", 0.05))
            hi_l = float(cfg.get("hi", 3.0))
            S = lo_l + (hi_l - lo_l) * np.stack([
                map_coordinates(frames[f], [pyf, pxf], order=1,
                                mode="nearest")
                for f in range(frames.shape[0])])          # (T, n_r)
            fps_v = float(cfg.get("fps", 30.0))
            loop_v = bool(cfg.get("loop", False))

            def luminance(t):
                f = int(t / 1000.0 * fps_v)
                if loop_v:
                    f %= S.shape[0]            # loop instead of freezing
                return S[min(f, S.shape[0] - 1)]

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
    # per-extra-pop population firing rate (Hz), CPU path only -- the
    # debug channel that proves chem drive reached ORN/GRN and its
    # central partners (exp019 validation reads this)
    extra_rate = {n: np.zeros(n_field)
                  for n in (circuit.get("extra_pops") or {})}
    cal = dict(fp.CAL)
    if args.base_scale != 1.0:
        _bs = args.base_scale
        for _k in list(cal):
            if _k.endswith("_BASE") and isinstance(cal[_k],
                                                   (int, float)):
                cal[_k] *= _bs
        print(f"base scale: all *_BASE bias currents x {_bs} "
              "(bci/gainstate working-point axis)")
    if args.std_gates:
        for _part in args.std_gates.split(";"):
            _g, _params = _part.split(":")
            _u, _tau = _params.split(",")
            if _g not in (circuit.get("extra_edges") or {}):
                ap.error(f"--std-gates: edge group '{_g}' not in circuit")
            circuit["extra_edges"][_g]["std"] = [float(_u), float(_tau)]
            print(f"std gate: {_g} U={_u} tau_rec={_tau} ms")
    if args.noise_scale != 1.0:
        _ns = args.noise_scale
        if isinstance(cal.get("OU"), dict):
            cal["OU"] = {k: v * _ns for k, v in cal["OU"].items()}
        for k in list(cal):
            if k.startswith("OU") and "TAU" not in k \
                    and isinstance(cal[k], (int, float)):
                cal[k] *= _ns
        for _spec in (circuit.get("extra_pops") or {}).values():
            if "ou_sigma" in _spec:
                _spec["ou_sigma"] = _spec["ou_sigma"] * _ns
        print(f"noise scale: all OU sigmas x {_ns} "
              "(bci/arousal state manipulation)")

    # ---- chemosensory drive (exp019): a chem_inputs.json entry turns
    # into a stateless chem_fn(t) -> {pop: per-neuron pA increment}
    # through the circuit's chem_groups tables (annotation type ->
    # compact index, built by the region builders)
    chem_fn, chem_meta = None, None
    if args.chem_input:
        import fnmatch
        chems = json.loads((ROOT / "viz" / "data" / "chem_inputs.json")
                           .read_text(encoding="utf-8"))
        cspec = next((v for v in chems
                      if v.get("id") == args.chem_input), None)
        if cspec is None:
            ap.error(f"chem input '{args.chem_input}' not found in "
                     "viz/data/chem_inputs.json")
        pops_x = circuit.get("extra_pops") or {}
        groups_x = circuit.get("chem_groups") or {}
        edge_ms = float(cspec.get("edge_ms", 150.0))
        chans = []                       # (pop, index array, amp, pulses)
        for ch in cspec["channels"]:
            pop = ch["pop"]
            if pop not in pops_x:
                print(f"chem: pop {pop} not in circuit -- channel "
                      f"'{ch['group']}' skipped")
                continue
            gm = groups_x.get(pop) or {}
            sel = np.zeros(len(pops_x[pop]["ids"]), dtype=bool)
            hits = []
            for tname, idx in gm.items():
                if ch["group"] == "ALL" or fnmatch.fnmatch(tname,
                                                           ch["group"]):
                    sel[idx] = True
                    hits.append(tname)
            if not hits:
                print(f"chem: no {pop} types match '{ch['group']}' "
                      "-- channel skipped")
                continue
            chans.append((pop, np.flatnonzero(sel), float(ch["amp_pa"]),
                          [(float(a), float(b))
                           for a, b in ch["pulses"]], ch))
            print(f"chem: {pop}[{ch['group']}] -> {int(sel.sum())} "
                  f"neurons ({', '.join(sorted(hits)[:4])}"
                  f"{'...' if len(hits) > 4 else ''}), "
                  f"{ch['amp_pa']:.0f} pA x {len(ch['pulses'])} pulses")

        def level(pulses, t):
            v = 0.0
            for a, b in pulses:
                if a <= t < b:
                    rise = min(1.0, (t - a) / edge_ms)
                    fall = min(1.0, (b - t) / edge_ms)
                    v += min(rise, fall)
            return min(v, 1.0)

        def chem_fn(t):
            # multiple channels may target the SAME pop (e.g. the MRCP
            # staircase stacks same-group channels) -- SUM their
            # increments; historical entries never overlapped in time
            # so summation is identical to the previous last-wins
            # behavior for them
            out = {}
            for pop, idx, amp, pulses, _ch in chans:
                lv = level(pulses, t)
                if lv <= 0.0:
                    continue
                inc = out.get(pop)
                if inc is None:
                    inc = np.zeros(len(pops_x[pop]["ids"]))
                    out[pop] = inc
                inc[idx] += amp * lv
            return out

        chem_meta = {"id": cspec["id"],
                     "modality": cspec.get("modality", ""),
                     "label": cspec.get("label", cspec["id"]),
                     "t_end_ms": float(cspec.get("t_end_ms", T_END)),
                     "channels": [{"pop": pop, "group": ch["group"],
                                   "n": int(len(sel)), "amp_pa": amp,
                                   "pulses": pulses}
                                  for pop, sel, amp, pulses, ch in chans]}
        # chem-run working point (exp019 calibration): at the exp017
        # default gain the central-brain internal recurrence is an
        # excitation-locked attractor -- ANY suprathreshold chem input
        # latches the whole CEN at ~82 Hz and it never decays. With the
        # recurrence-only gain at 0.002 (feedforward groups untouched)
        # the response is antennal-lobe specific and decays after the
        # pulse. Visual runs keep exp017's behavior (override only when
        # --chem-input is active).  bci/seizure opts OUT via
        # "no_cen_damp": the 82 Hz latch IS the modeled seizure.
        # See exp019 README.
        _G_CEN_CHEM = 0.002
        if "cen_gain" in cspec:
            # bci/seizure gain sweep: explicit recurrence gain, between
            # the safe working point (0.002) and the default (0.004)
            circuit["extra_edges"]["CEN_C"]["g_unit"] = \
                float(cspec["cen_gain"])
            if "KCM" in circuit["extra_edges"]:
                circuit["extra_edges"]["KCM"]["g_unit"] = \
                    float(cspec["cen_gain"])
            print(f"chem calibration: CEN_C recurrence gain -> "
                  f"{cspec['cen_gain']} (explicit cen_gain)")
        elif not cspec.get("no_cen_damp", False) \
                and "CEN_C" in (circuit.get("extra_edges") or {}):
            circuit["extra_edges"]["CEN_C"]["g_unit"] = _G_CEN_CHEM
            if "KCM" in circuit["extra_edges"]:
                circuit["extra_edges"]["KCM"]["g_unit"] = _G_CEN_CHEM
            print(f"chem calibration: CEN_C recurrence gain -> "
                  f"{_G_CEN_CHEM} (chem-run working point, exp019)")
    if args.gain_scale:
        # runtime pathway-gain modulation (bci/attend): multiply named
        # extra edge groups' g_unit.  g_unit is a runtime scalar (never
        # in a cache key), so no kernel rebuild; applied AFTER the chem
        # working point so arms can scale suppressed groups back up.
        for _tok in args.gain_scale.split(","):
            _gn, _, _fac = _tok.partition("=")
            _gn = _gn.strip()
            if _gn not in circuit["extra_edges"]:
                raise SystemExit(
                    f"--gain-scale: unknown edge group {_gn!r} "
                    f"(have: {sorted(circuit['extra_edges'])})")
            _gu = circuit["extra_edges"][_gn]["g_unit"]
            _gu = cal[_gu] if isinstance(_gu, str) else float(_gu)
            circuit["extra_edges"][_gn]["g_unit"] = _gu * float(_fac)
            print(f"gain-scale: {_gn} g_unit x{float(_fac):g} "
                  f"-> {_gu * float(_fac):.4g}", flush=True)

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
    if not pool_info:
        # the float64 kernels are only consumed by the in-run pooling
        # validation (pool mode); drop them in exact mode -- 45 x 1.23M
        # f64 is ~443 MB and the application path is all-f32
        coef_scalp.clear()
    # row-major (CHUNK, N): records append CONTIGUOUS rows (a strided
    # column write costs a cache miss per dipole); flush feeds the
    # transposed view straight into the GEMM (BLAS native, no copy)
    if not args.gpu:
        ybuf = {name: np.zeros((CHUNK, coef_f32[name].shape[1]),
                               np.float32) for name in gnames}
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
        for name in extra_rate:
            extra_rate[name][j] = (1000.0 * int(sp[name].sum())
                                   / len(st["extra_ids"][name]))

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
    if args.gpu:
        # ---- P3 GPU path: biology loop + record buffers on device ----
        import cupy as cp
        from ffbm import gpu as fgpu

        rng_g = np.random.default_rng(SEED)
        st_g = fp.build_stack(circuit, cal, rng_g)
        trial = fgpu.GPUTrial(st_g)
        Kg = trial.K
        if args.plastic_state_in:
            _si = np.load(args.plastic_state_in)
            if str(_si["checksum"]) != KCM_CHECKSUM:
                raise SystemExit(
                    "plastic state mismatch: edge ids differ from this "
                    "session's KCM split -- refusing to load")
            _pw = trial.exp["KCM"].plast_w
            assert _si["w"].shape == _pw.shape
            _pw.set(np.ascontiguousarray(_si["w"]))
            print(f"plastic state IN: w mean {float(_si['w'].mean()):.4f}"
                  f" min {float(_si['w'].min()):.4f}", flush=True)
        ybuf_g = {name: cp.zeros((CHUNK, coef_f32[name].shape[1]),
                                 cp.float32) for name in gnames}
        coef_g = {name: cp.asarray(coef_f32[name]) for name in gnames}
        ker_g = {name: cp.asarray(ker[name].coef) for name in ker}
        keep_g = {name: (cp.asarray(kernel_keep[name])
                         if kernel_keep.get(name) is not None else None)
                  for name in gnames}
        phi_scalp_g = cp.zeros((n_field, len(scalp_dirs)), cp.float32)
        phi_g = cp.zeros((n_field, 3), cp.float64)
        is_t4_g = cp.asarray(is_t4)
        is_t5_g = cp.asarray(is_t5)
        # rate accumulators stay on device (sums); per-record host syncs
        # would dominate the loop -- means/scales applied at the end
        sum_g = {"R": cp.zeros(n_field), "L": cp.zeros(n_field),
                 "MID": cp.zeros(n_field, cp.int64),
                 "T4": cp.zeros(n_field, cp.int64),
                 "T5": cp.zeros(n_field, cp.int64)}
        mb = (sum(b.nbytes for b in ybuf_g.values())
              + sum(c.nbytes for c in coef_g.values())) / 1e6
        print(f"GPU forward buffers: {CHUNK} records, {mb:.0f} MB "
              f"on device", flush=True)

        def flush_scalp_gpu(j0, c):
            for name in gnames:
                if pool_info.get(name) is not None:
                    raise RuntimeError("--pool is not supported with "
                                       "--gpu")
                Y = ybuf_g[name][:c]
                phi_scalp_g[j0:j0 + c] += (coef_g[name] @ Y.T).T
            phi3 = cp.zeros((c, 3), cp.float64)
            for name in ker:
                Y = ybuf_g[name][:c]
                phi3 += (ker_g[name] @ Y.T).T
            phi_g[j0:j0 + c] = phi3 * 1e-12

        def record_gpu(k, j, t, inc_f):
            nonlocal jbuf
            for name in gnames:
                row = ybuf_g[name][jbuf]
                if name == "PHOTO":
                    row[:] = cp.asarray(
                        (cal["I_R_BASE"] + inc_f).astype(np.float32))
                elif name == "RL":
                    trial.graded["RL"].ecur_row(
                        Kg, trial.v_post("L"), row)
                elif name == "LM":
                    trial.graded["LM"].ecur_row(
                        Kg, trial.v_post("MID"), row)
                else:
                    post = extra_post.get(name, "T45")
                    trial.exp[name].ecur_row(
                        Kg, trial.v_post(post), row, keep_g[name])
            jbuf += 1
            if jbuf == CHUNK:
                flush_scalp_gpu(j - CHUNK + 1, CHUNK)
                jbuf = 0
            # device-side accumulators: no host sync per record
            sum_g["R"][j] = trial.r_release.sum()
            sum_g["L"][j] = trial.l_release.sum()
            m45 = trial._mask("T45")
            sum_g["MID"][j] = trial._mask("MID").sum()
            sum_g["T4"][j] = m45[is_t4_g].sum()
            sum_g["T5"][j] = m45[is_t5_g].sum()
            stim[j] = float(np.mean(luminance(t)))

        trial.run(lambda t: I_LUM * (luminance(t) - 1.0),
                  int(T_END / fp.DT_MS), on_record=record_gpu,
                  chem_fn=chem_fn, mod_fn=mod_fn)
        if jbuf:
            flush_scalp_gpu(n_field - jbuf, jbuf)
        cp.cuda.runtime.deviceSynchronize()
        if args.plastic_mb and "KCM" in getattr(trial, "exp", {}):
            _ws = trial.exp["KCM"].plast_w
            _el = trial.exp["KCM"].plast_elig
            print(f"plastic result: KCM w_scale mean "
                  f"{float(_ws.mean()):.4f} min {float(_ws.min()):.4f} "
                  f"floor-hit {int((_ws <= 0.0201).sum())}/{_ws.size} "
                  f"| elig max {float(_el.max()):.4f}",
                  flush=True)
            if args.plastic_state_out:
                _w_out = cp.asnumpy(_ws).astype(np.float32)
                np.savez(args.plastic_state_out, w=_w_out,
                         n=_w_out.size, checksum=KCM_CHECKSUM)
        phi_scalp_g *= 1e-12
        phi_scalp = cp.asnumpy(phi_scalp_g)
        phi = cp.asnumpy(phi_g)
        rate["R"] = cp.asnumpy(sum_g["R"]) * (100.0 / n_r)
        rate["L"] = cp.asnumpy(sum_g["L"]) * (100.0 / n_l)
        rate["MID"] = cp.asnumpy(sum_g["MID"]) * (1000.0 / n_mid)
        rate["T4"] = cp.asnumpy(sum_g["T4"]) * (1000.0 / int(is_t4.sum()))
        rate["T5"] = cp.asnumpy(sum_g["T5"]) * (1000.0 / int(is_t5.sum()))
        jbuf = 0
    else:
        fp.simulate(circuit, cal, lambda t: I_LUM * (luminance(t) - 1.0),
                    SEED, T_END, on_sample=record, chem_fn=chem_fn,
                    mod_fn=mod_fn)
        if jbuf:
            flush_scalp(n_field - jbuf, jbuf)
    if extra_rate and not args.gpu:
        np.savez(OUT / "_debug_extra_rates.npz", **extra_rate)
        for name, tr in extra_rate.items():
            print(f"extra rate {name}: mean {tr.mean():.1f} Hz, "
                  f"peak {tr.max():.0f} Hz")
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
            "regions": {"on": sorted(active),
                        "off": sorted(r for r in freg.known_regions()
                                      if r not in active)},
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
    if chem_meta is not None:
        data["meta"]["chem_input"] = chem_meta
    # extra-region layers for the point cloud. VNC IS shown (its true
    # native geometry makes it stick out of the head sphere at x202 --
    # the honest picture of "the fly CNS does not fit in a human head");
    # it only stays out of the kernels/fit (forward=False)
    layer_names = {"VPN": "VPN 投射神经元", "CB": "中央脑目标(exp016)",
                   "OLR": "其余视叶", "CEN": "中央脑",
                   "VNC": "腹索 VNC（仅动力学·不投影）",
                   "ORN": "嗅觉受体神经元 ORN（外周端·不投影）",
                   "GRN": "味觉受体神经元 GRN（外周端·不投影）"}
    layer_colors = {"VPN": "#f472b6", "CB": "#c084fc", "OLR": "#8b9dc3",
                    "CEN": "#fbbf24", "VNC": "#5eead4",
                    "ORN": "#fb7185", "GRN": "#34d399"}
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
