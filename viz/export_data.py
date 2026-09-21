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
import pandas as pd

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
    ap.add_argument("--plastic-mod-scale", type=float, default=1.0,
                    help="reinforcement gate STRENGTH (mod amplitude; "
                         "1.0 = full dopamine proxy -- the motivation/"
                         "reward-strength axis, bci/motivate)")
    ap.add_argument("--plastic-lr", type=float, default=0.0015,
                    help="plasticity learning rate (per step, gated)")
    ap.add_argument("--plastic-tau-w", type=float, default=0.0,
                    help="slow homeostatic recovery of plastic w_scale "
                         "toward 1, ms (0 = frozen weights; extinction, "
                         "bci/condit2)")
    ap.add_argument("--plastic-dual-tauw", type=str, default=None,
                    help="'g_ms,ab_ms': split KCM by presynaptic KC "
                         "subtype (KCg* vs the rest) into TWO plastic "
                         "groups with separate recovery time constants "
                         "-- the short-vs-long memory dissociation "
                         "(bci/dualmem; overrides --plastic-tau-w)")
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
    ap.add_argument("--mb-mod-gain", type=float, default=None,
                    help="MB-modulator surgery: split the APL (GABA) <->"
                         "KC/MBON feedback rows out of CEN_C into group "
                         "MBM at this ABSOLUTE gain -- restores divisive "
                         "normalization of KC output, raises the stable "
                         "KC->MBON readout-leg ceiling (DPM excluded: "
                         "dopaminergic positive loop)")
    ap.add_argument("--mb-mod-tau", type=float, default=None,
                    help="MBM synaptic time constant, ms (default: same "
                         "as CEN_C; fast-kinetics APL variant, "
                         "bci/condit3 follow-up)")
    ap.add_argument("--mb-mod-erev", type=float, default=None,
                    help="MBM inhibitory reversal potential, VOLTS "
                         "(default: cal E_REV_INH -0.080 = subtractive-"
                         "dominated).  Setting it to the resting "
                         "potential (-0.060) turns the APL feedback into "
                         "PURE SHUNTING: zero current at rest, "
                         "conductance that divides KC excitation -- the "
                         "divisive-normalization implementation the APL "
                         "arc concluded was missing (bci/shunt)")
    ap.add_argument("--plastic-w0", type=float, default=None,
                    help="initial plastic w_scale (default 1.0 = the "
                         "depression form's ceiling; with a NEGATIVE "
                         "--plastic-lr the rule flips to potentiation "
                         "from this baseline toward 1 and w0 doubles as "
                         "the recovery target, bci/dangate)")
    ap.add_argument("--plastic-kcm-gain", type=float, default=None,
                    help="ABSOLUTE g_unit for the plastic KC->MBON "
                         "group, escaping the CEN_C working-point "
                         "damping so MBON firing can carry the value "
                         "prediction V (bci/dangate)")
    ap.add_argument("--mbon-dan-gain", type=float, default=None,
                    help="value-feedback surgery: split MBON->DAN rows "
                         "out of CEN_C into group MBMD at this ABSOLUTE "
                         "gain -- the learned V inhibits the reward-"
                         "driven DANs, closing the circuit-level error "
                         "loop (bci/dangate; 349 inhib / 244 exc rows)")
    ap.add_argument("--plastic-dan-gate", type=str, default=None,
                    help="'tau_ms,gain[,t_ref_ms]': derive the "
                         "reinforcement gate from the DAN population "
                         "firing rate itself (low-passed; baseline "
                         "frozen over the last half of [0,t_ref)) -- "
                         "mod = clip((r-r0)*gain, 0, 1) replaces the "
                         "--plastic-window gate; a _dan_trace.npy "
                         "(t, r, mod) is written next to the output "
                         "(GPU only, bci/dangate)")
    ap.add_argument("--plastic-comp-reward", type=str, default=None,
                    help="DAN type pattern (e.g. 'PAM07*'): restrict "
                         "the error loop to ONE compartment -- the "
                         "reward channel and the dan-gate monitor see "
                         "only the matching cells, and mod is applied "
                         "ONLY to the plastic KC->MBON edges whose "
                         "postsynaptic MBON feeds back to them (the "
                         "MBMD GABA rows define the map; mod becomes "
                         "a per-edge vector, scalar elsewhere).  "
                         "Requires --plastic-mb + --mbon-dan-gain + "
                         "--plastic-dan-gate on the single-KCM split "
                         "(bci/comp1)")
    ap.add_argument("--kcm-fanin", type=float, default=None,
                    help="weight threshold W in [1,5): pull the raw "
                         "connectome's WEAK KC->MBON rows (W <= wt < 5) "
                         "into the KCM plastic pool on top of the "
                         "w>=5 split (bci/fanin1: the readout leg's "
                         "fan-in density -- the weak tail is +83%% "
                         "pairs / 33,496 -> 61,210 at W=1).  KC rows "
                         "are excitatory (sign +1).  Requires "
                         "--plastic-mb; salts the cache key")
    ap.add_argument("--gain-scale", type=str, default=None,
                    help="runtime pathway-gain modulation, 'GRP=f[,"
                         "GRP=f...]': multiply the named extra edge "
                         "groups' g_unit by f (resolved absolute; not "
                         "in any cache key -- g_unit is a runtime "
                         "scalar; bci/attend attention model)")
    ap.add_argument("--pop-rate", type=str, default=None,
                    help="comma list of population-rate groups to "
                         "record into _pop_rate.npz (Hz per neuron, "
                         "1-ms bins): annotation CLASS names subset "
                         "within CEN (MBON, Kenyon_Cell, DAN, ALPN, "
                         "..) or a whole extra-pop name (ORN, OLR, ..) "
                         "-- the central-subpopulation instrumentation "
                         "for the validation table (VALIDATION #5)")
    ap.add_argument("--chem-amp-scale", type=float, default=1.0,
                    help="multiply every chem channel amplitude by this "
                         "runtime factor (intensity sweeps with a "
                         "single entry, bci/pnrate dose-response; not "
                         "in any cache key)")
    ap.add_argument("--mbon-bias", type=float, default=None,
                    help="tonic bias current (pA, pass NEGATIVE to "
                         "hyperpolarize) on all CEN MBON cells for the "
                         "whole run, appended as an extra chem channel "
                         "(group MBON*, pulses spanning the trial): "
                         "moves MBONs off their binary operating point "
                         "so KC->MBON weights read out as graded V "
                         "(bci/dangate2). Runtime amplitude like "
                         "--chem-amp-scale, never in a cache key")
    ap.add_argument("--pop-rate-neurons", action="store_true",
                    help="with --pop-rate: also dump PER-NEURON spike "
                         "counts over the whole trial "
                         "(_pop_rate_neurons.npz) -- the sparse-code "
                         "instrument (responsive fraction per class, "
                         "bci/shunt)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the trial RNG seed (params: "
                    "stimulus.seed): varies delay jitter + OU background "
                    "noise; circuit/kernel caches are seed-independent")
    args = ap.parse_args()
    if (args.plastic_state_in or args.plastic_state_out) and (
            not args.plastic_mb or not args.gpu):
        ap.error("--plastic-state-in/out requires --plastic-mb --gpu")
    if args.plastic_dan_gate and not args.gpu:
        ap.error("--plastic-dan-gate requires --gpu (device spike "
                 "readback)")
    if args.plastic_comp_reward and not (
            args.plastic_mb and args.mbon_dan_gain is not None
            and args.plastic_dan_gate):
        ap.error("--plastic-comp-reward requires --plastic-mb + "
                 "--mbon-dan-gain + --plastic-dan-gate")
    if args.kcm_fanin is not None and (
            not args.plastic_mb or not (1.0 <= args.kcm_fanin < 5.0)):
        ap.error("--kcm-fanin needs --plastic-mb and W in [1,5)")
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
    # ---- shared annotation preamble for the CEN_C runtime surgeries
    # (plastic-mb / al-gain / mb-mod-gain) ----
    if (args.plastic_mb or args.al_gain is not None
            or args.mb_mod_gain is not None
            or args.mbon_dan_gain is not None):
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
        _dan = set(_rmap[int(b)] for b in
                   _ann.loc[_cls == "DAN", "bodyId"]
                   if int(b) in _rmap)
    if args.plastic_mb:
        # runtime surgery (bci/condit): move KC->MBON pairs out of the
        # CEN_C recurrence table into dedicated PLASTIC group(s); the
        # circuit cache stays untouched but the kernel key is salted so
        # the forward kernels rebuild for the split topology.  With
        # --plastic-dual-tauw the split is by presynaptic KC subtype
        # (KCg* = gamma vs the rest = alpha/beta-like), each group with
        # its OWN recovery tau -- the fly's short-vs-long-memory
        # compartment dissociation (bci/dualmem).
        import hashlib
        _tab = circuit["extra_edges"]["CEN_C"]["table"]
        _m = _tab["body_pre"].isin(_kc) & _tab["body_post"].isin(_mbon)
        _kcm_all = _tab[_m].reset_index(drop=True)
        circuit["extra_edges"]["CEN_C"]["table"] = \
            _tab[~_m].reset_index(drop=True)
        _inv_rmap = {int(v): int(k) for k, v in _rmap.items()}
        _typ = _ann.set_index("bodyId")["type"]
        if args.kcm_fanin is not None:
            # readout-leg fan-in density (bci/fanin1): the w>=5 cut
            # keeps 33,496 KC->MBON pairs; the weak tail (w in [W,5))
            # adds up to +83% pairs at W=1 -- pull those rows from the
            # RAW connectome (raw body-id space) and map them through
            # _rmap into the CEN table's id space; KC output is ACh
            # (sign +1).  The salt pins the new topology.
            import pyarrow.feather as _pfk
            _tw = _pfk.read_table(_fdata.RAW /
                                  "connectome-weights.feather")
            _wp = _tw.column("body_pre").combine_chunks().to_numpy(
                zero_copy_only=False)
            _wq = _tw.column("body_post").combine_chunks().to_numpy(
                zero_copy_only=False)
            _ww = _tw.column("weight").combine_chunks().to_numpy(
                zero_copy_only=False)
            _kcR = set(int(b) for b in
                       _ann.loc[_cls == "Kenyon_Cell", "bodyId"]
                       if int(b) in _rmap)
            _mbR = set(int(b) for b in
                       _ann.loc[_cls == "MBON", "bodyId"]
                       if int(b) in _rmap)
            _mw = ((_ww >= args.kcm_fanin) & (_ww < 5)
                   & np.isin(_wp, list(_kcR))
                   & np.isin(_wq, list(_mbR)))
            _have = set(zip(_kcm_all["body_pre"].to_numpy(np.int64),
                            _kcm_all["body_post"].to_numpy(np.int64)))
            _pp, _qq = circuit["pre_pos"], circuit["post_pos"]
            _keep = []
            for p, q, w in zip(_wp[_mw], _wq[_mw], _ww[_mw]):
                p2, q2 = int(_rmap[int(p)]), int(_rmap[int(q)])
                if (p2, q2) not in _have and p2 in _pp and q2 in _qq:
                    _keep.append((p2, q2, float(w)))
            if _keep:
                _wk = pd.DataFrame(_keep, columns=["body_pre",
                                                   "body_post",
                                                   "weight"])
                _wk["sign"] = 1.0        # KC output is ACh (excitatory)
                if "sign" not in _kcm_all.columns:
                    _kcm_all["sign"] = 1.0
                _kcm_all = pd.concat([_kcm_all, _wk], ignore_index=True)
            print(f"kcm-fanin: +{len(_keep)} weak rows (w in "
                  f"[{args.kcm_fanin:g},5)) -> KCM pool "
                  f"{len(_kcm_all)} edges", flush=True)
            ckey = (ckey + f"+kcf{args.kcm_fanin:g}"
                    ) if ckey is not None else None
        if args.plastic_dual_tauw:
            _tg, _tab_tau = (float(v) for v in
                             args.plastic_dual_tauw.split(","))
            _pt = _kcm_all["body_pre"].map(
                lambda p: str(_typ.get(_inv_rmap.get(int(p), -1), "?")))
            _isg = _pt.str.startswith("KCg").to_numpy()
            parts = (("KCMg", _kcm_all[_isg].reset_index(drop=True),
                      _tg), ("KCMab",
                             _kcm_all[~_isg].reset_index(drop=True),
                             _tab_tau))
        else:
            parts = (("KCM", _kcm_all,
                      float(args.plastic_tau_w)
                      if args.plastic_tau_w > 0 else None),)
        # KCM_STATE[gname] = (pre ids in SYNAPSE order, pre-type labels,
        #                     checksum) -- the per-group state identity
        KCM_STATE = {}
        for gname, ktab, tau_w in parts:
            circuit["extra_edges"][gname] = {
                "pre": ("CEN",), "post": "CEN", "table": ktab,
                "tau_s": circuit["extra_edges"]["CEN_C"]["tau_s"],
                "g_unit": (float(args.plastic_kcm_gain)
                           if args.plastic_kcm_gain is not None
                           else circuit["extra_edges"]["CEN_C"]["g_unit"]),
                "forward": True,
                "plast": {"lr": float(args.plastic_lr),
                          "tau_ms": 400.0,
                          **({"w0": float(args.plastic_w0)}
                             if args.plastic_w0 is not None else {}),
                          **({"tau_w_ms": tau_w}
                             if tau_w is not None else {})}}
            # edge identity: the synapse reorders rows with
            # lexsort((pre, post)) -- DETERMINISTIC in the table content
            # -- so hashing the table's id bytes pins the w_scale order
            _ord = np.lexsort((ktab["body_pre"].to_numpy(np.int64),
                               ktab["body_post"].to_numpy(np.int64)))
            _pre = ktab["body_pre"].to_numpy(np.int64)[_ord]
            KCM_STATE[gname] = (
                _pre,
                np.array([str(_typ.get(_inv_rmap.get(int(p), -1), "?"))
                          for p in _pre], dtype="<U24"),
                hashlib.md5(np.concatenate([
                    ktab["body_pre"].to_numpy(np.int64),
                    ktab["body_post"].to_numpy(np.int64)]).tobytes()
                ).hexdigest())
            _pw = f", tau_w={tau_w:g} ms" if tau_w is not None else ""
            print(f"plastic-mb: {gname} split {len(ktab)} KC->MBON "
                  f"pairs (lr={args.plastic_lr}{_pw})", flush=True)
        # salts distinguish topologies AND namespaces: plmb1 = single
        # KCM (fresh -- the pre-plmb2 dual pilot polluted "+plmb"),
        # plmb2 = dual subtype split
        ckey = ((ckey + "+plmb2") if args.plastic_dual_tauw
                else (ckey + "+plmb1")) if ckey is not None else None
    if args.al_gain is not None:
        # AL-independence surgery (bci/condit3): the AL->KC (PN->Kenyon)
        # rows live in the CEN_C recurrence table, so the chem working
        # point (g 0.002) crushes the real odor->KC pathway 500x.  Split
        # them into an ALK feedforward group at an independent ABSOLUTE
        # gain.  AL local recurrence (PN<->LN) stays damped: the minimal
        # step is the feedforward ORN ->(ORN_C, full gain) PN ->(ALK) KC
        # pathway; stability must be calibrated (latch risk).
        _al = set(_rmap[int(b)] for b in
                  _ann.loc[_ann["class"].isin(
                      ("ALPN", "ALLN", "ALIN", "ALON")), "bodyId"]
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
    if args.mb_mod_gain is not None:
        # MB-modulator surgery (bci/condit3 follow-up): APL (1 per side,
        # GABAergic wide field) carries a MASSIVE negative-feedback loop
        # with the KC compartment -- APL<->KC 9,326 pairs / 400k syn --
        # all inside the damped CEN_C table.  Splitting both directions
        # into MBM at an independent gain restores divisive normalization
        # of KC output, which should raise the stable ceiling for the
        # KC->MBON readout leg (true-odor conditioning visibility; the
        # nolr control latched at KCM x250 without it).  DPM is
        # deliberately EXCLUDED: it is dopaminergic (+1 sign) and its
        # 4k DPM->KC rows form a POSITIVE loop that amplifies runaway
        # instead of limiting it (measured: nolr 780 uV latch at
        # gain 0.2 with DPM included) -- DPM's real role is slow
        # neuropeptide modulation, not fast feedback.
        _apm = set(_rmap[int(b)] for b in
                   _ann.loc[_ann["type"] == "APL", "bodyId"]
                   if int(b) in _rmap)
        _kcmb = _kc | _mbon
        _tab = circuit["extra_edges"]["CEN_C"]["table"]
        _m = ((_tab["body_pre"].isin(_apm)
               & _tab["body_post"].isin(_kcmb))
              | (_tab["body_pre"].isin(_kcmb)
                 & _tab["body_post"].isin(_apm)))
        _mbm = _tab[_m].reset_index(drop=True)
        circuit["extra_edges"]["CEN_C"]["table"] = \
            _tab[~_m].reset_index(drop=True)
        circuit["extra_edges"]["MBM"] = {
            "pre": ("CEN",), "post": "CEN", "table": _mbm,
            "tau_s": (float(args.mb_mod_tau) if args.mb_mod_tau
                      else circuit["extra_edges"]["CEN_C"]["tau_s"]),
            "g_unit": float(args.mb_mod_gain), "forward": True,
            "use_sign": True,
            **({"e_rev_inh": float(args.mb_mod_erev)}
               if args.mb_mod_erev is not None else {})}
        _pre_n = int(_mbm["body_pre"].isin(_apm).sum())
        print(f"mb-mod-gain: MBM split {len(_mbm)} APL<->KC/MBON "
              f"pairs from CEN_C (gain={args.mb_mod_gain:g}; "
              f"{_pre_n} modulator->, {len(_mbm) - _pre_n} <-principal; "
              f"use_sign"
              + (f", e_rev_inh={args.mb_mod_erev:g} V (PURE SHUNT)"
                 if args.mb_mod_erev is not None else "") + ")",
              flush=True)
        ckey = (ckey + "+mbm") if ckey is not None else None
    if args.mbon_dan_gain is not None:
        # value-feedback surgery (bci/dangate): the error-gate circuit
        # closing.  The MBON->DAN rows (593 pairs: 349 GABA / 244
        # excitatory, net weighted sign -2127) stay buried in the damped
        # CEN_C table, so the learned value prediction V can never reach
        # the DANs.  Splitting them into MBMD at an ABSOLUTE gain lets
        # the potentiated KC->MBON readout (V) inhibit the reward-driven
        # PAM/PPL1 cells -- dopamine itself becomes error-coding:
        #   DAN rate  ~  R(reward drive) - V(MBON feedback)
        #   mod       =  clip((r - r0) * gain, 0, 1)
        # which replaces the experiment-layer Rescorla-Wagner scalar of
        # bci/blocking2 with circuit dynamics.
        _tab = circuit["extra_edges"]["CEN_C"]["table"]
        _m = _tab["body_pre"].isin(_mbon) & _tab["body_post"].isin(_dan)
        # minimal negative-feedback architecture: keep ONLY the GABA
        # rows (349 of 593).  The mix's net sign is driving-force
        # unstable in conductance mode (exc rows push with ~55 mV vs
        # the GABA rows' ~25 mV, so the consensus net -2127 arrives
        # net-EXCITATORY at the DANs: measured r max 209 -> 521 Hz on
        # the first paired trial with the full mix).  The appetitive
        # error loop only needs the inhibitory projection -- the fly's
        # MBON->PAM feedback for appetitive learning is the GABAergic
        # pathway; the 244 excitatory rows are dropped as a documented
        # modeling choice, not a connectome claim.
        _mbmd = _tab[_m & (_tab["sign"] < 0)].reset_index(drop=True)
        circuit["extra_edges"]["CEN_C"]["table"] = \
            _tab[~_m].reset_index(drop=True)
        circuit["extra_edges"]["MBMD"] = {
            "pre": ("CEN",), "post": "CEN", "table": _mbmd,
            "tau_s": circuit["extra_edges"]["CEN_C"]["tau_s"],
            "g_unit": float(args.mbon_dan_gain), "forward": True,
            "use_sign": True}
        # compartment-private error loop: expose the MBMD-contacted DAN
        # subset as a synthetic chem group so (a) the dan-gate monitor
        # and (b) a reward channel can address exactly the cells where
        # the V-subtraction happens (the fly's MBON->PAM feedback is
        # compartment-specific; driving/reading the whole cluster
        # dilutes the error signal)
        _tg = np.unique(_mbmd["body_post"].to_numpy(np.int64))
        _tg_idx = np.searchsorted(
            circuit["extra_pops"]["CEN"]["ids"], _tg)
        ((circuit.get("chem_groups") or {}).setdefault("CEN", {})
         )["DAN_err"] = _tg_idx
        print(f"mbon-dan-gain: MBMD split {len(_mbmd)} GABA MBON->DAN "
              f"pairs from CEN_C (gain={args.mbon_dan_gain:g}; "
              f"{len(_tg)} DAN_err cells)", flush=True)
        # compartment restriction (bci/comp1): a compartment is the
        # MBON set feeding back to ONE DAN type -- the reward channel
        # and the gate monitor will address only those cells, and mod
        # only reaches the plastic edges onto those MBONs
        COMP_CTX = None
        if args.plastic_comp_reward:
            import fnmatch as _fmcr
            _dpost_t = _mbmd["body_post"].map(
                lambda b: str(_typ.get(_inv_rmap.get(int(b), -1), "?")))
            _keep = _dpost_t.map(
                lambda t: _fmcr.fnmatch(t, args.plastic_comp_reward))
            if not _keep.any():
                raise SystemExit(
                    f"comp-reward: no MBMD DAN matches "
                    f"'{args.plastic_comp_reward}' (types: "
                    f"{sorted(set(_dpost_t))[:8]}...)")
            COMP_CTX = {
                "d_ids": np.unique(_mbmd.loc[_keep, "body_post"]
                                   .to_numpy(np.int64)),
                "mbons": set(_mbmd.loc[_keep, "body_pre"]
                             .to_numpy(np.int64).tolist())}
            print(f"comp-reward: {args.plastic_comp_reward} -> "
                  f"{len(COMP_CTX['d_ids'])} DAN cells, "
                  f"{len(COMP_CTX['mbons'])} feedback MBONs",
                  flush=True)
        ckey = (ckey + "+mdg") if ckey is not None else None
    # ---- population-rate instrumentation (VALIDATION #5) -----------
    # groups resolve AFTER the surgeries so DAN_err etc. could be added
    # later; class subsets index into the CEN pop, bare pop names cover
    # a whole extra pop
    POP_RATE = {}
    if args.pop_rate:
        if "_rmap" not in locals():
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
            _cen_ids_pr = circuit["extra_pops"]["CEN"]["ids"]
            _rmap = dict(zip(_cen_raw.tolist(),
                             _cen_ids_pr.tolist()))
        for _g in (s.strip() for s in args.pop_rate.split(",")):
            if not _g:
                continue
            if _g in (circuit.get("extra_pops") or {}):
                POP_RATE[_g] = ("POP", None,
                                len(circuit["extra_pops"][_g]["ids"]))
                continue
            _ids = set(_rmap[int(b)] for b in
                       _ann.loc[_cls == _g, "bodyId"]
                       if int(b) in _rmap)
            if not _ids:
                raise SystemExit(f"--pop-rate: no CEN class {_g!r}")
            _idx = np.searchsorted(
                circuit["extra_pops"]["CEN"]["ids"],
                np.array(sorted(_ids), dtype=np.int64))
            POP_RATE[_g] = ("CEN", _idx, len(_idx))
        print(f"pop-rate: "
              + ", ".join(f"{g}[{v[2]}]" for g, v in POP_RATE.items()),
              flush=True)
    mod_fn = None
    if args.plastic_window:
        _wa, _wb = (float(v) for v in args.plastic_window.split(","))

        def mod_fn(t, _a=_wa, _b=_wb, _s=float(args.plastic_mod_scale)):
            return _s if _a <= t < _b else 0.0
    # ---- circuit-level reinforcement gate (bci/dangate) --------------
    # replaces the window gate: the DAN (PAM/PPL1) population rate IS
    # the dopamine signal.  mod_fn reads the PREVIOUS step's spikes
    # (causal: dopamine follows the reward/MBON state), low-passes the
    # population rate, freezes a baseline r0 over the last half of the
    # pre-stimulus window and clips deviations above it.  With the
    # MBMD value-feedback split active, a learned V suppresses r below
    # its unpaired level -- the gate closes -- WITHOUT any experiment-
    # layer error arithmetic.  _DAN_CTX["mask"] is bound late (device
    # spike-mask view of the CEN pop, set right before trial.run).
    _DAN_CTX = {}
    if args.plastic_dan_gate:
        import fnmatch as _fm
        _dp = args.plastic_dan_gate.split(",")
        _dtau, _dgain = float(_dp[0]), float(_dp[1])
        _dtref = float(_dp[2]) if len(_dp) > 2 else 500.0
        _gm = (circuit.get("chem_groups") or {}).get("CEN", {})
        if "DAN_err" in _gm:
            # compartment-private monitor: exactly the DANs the MBMD
            # value-feedback reaches (bci/dangate)
            _didx = np.asarray(_gm["DAN_err"], dtype=np.int64)
            _src = f"DAN_err compartment ({len(_didx)} cells)"
        else:
            _sel = np.zeros(
                len(circuit["extra_pops"]["CEN"]["ids"]), dtype=bool)
            _hits = sorted(
                t for t in _gm
                if _fm.fnmatch(t, "PAM*") or _fm.fnmatch(t, "PPL*"))
            for _t in _hits:
                _sel[_gm[_t]] = True
            _didx = np.flatnonzero(_sel)
            _src = f"{len(_hits)} types ({', '.join(_hits[:4])}" \
                   f"{'...' if len(_hits) > 4 else ''})"
        _cmask = None
        if args.plastic_comp_reward:
            # monitor ONLY the compartment's cells; mod becomes a
            # per-edge vector zero outside the compartment
            _ids = circuit["extra_pops"]["CEN"]["ids"]
            _didx = np.flatnonzero(np.isin(_ids, COMP_CTX["d_ids"]))
            _src = f"compartment {args.plastic_comp_reward} " \
                   f"({len(_didx)} cells)"
            if "KCM" not in circuit["extra_edges"]:
                raise SystemExit("comp-reward needs the single-pool "
                                 "--plastic-mb split (no dual-tauw)")
            _kt = circuit["extra_edges"]["KCM"]["table"]
            _ord = np.lexsort((_kt["body_pre"].to_numpy(np.int64),
                               _kt["body_post"].to_numpy(np.int64)))
            _posts = _kt["body_post"].to_numpy(np.int64)[_ord]
            _cmask = np.isin(_posts, list(COMP_CTX["mbons"])
                             ).astype(np.float32)
        print(f"dan-gate: monitoring {_src} -> {len(_didx)} neurons, "
              f"tau={_dtau:g} ms, gain={_dgain:g}, t_ref={_dtref:g} ms"
              + (f"; comp gate {int(_cmask.sum())}/{len(_cmask)} "
                 f"edges" if _cmask is not None else ""), flush=True)
        _d_dt_s = float(fp.DT_MS) * 1e-3
        _d_alpha = 1.0 - np.exp(-_d_dt_s * 1e3 / _dtau)

        def mod_fn(t, _ctx=_DAN_CTX, _idx=_didx,
                   _a=_d_alpha, _g=_dgain, _tr=_dtref, _hz=1.0
                   / (len(_didx) * _d_dt_s), _cm=_cmask):
            _m = _ctx.get("mask")
            _s = int(_m[_idx].sum()) if _m is not None else 0
            _r = _ctx.get("r", 0.0) + (_s * _hz - _ctx.get("r", 0.0)) * _a
            _ctx["r"] = _r
            if t < _tr:                      # pre-stimulus: accumulate
                _ctx.setdefault("r0s", []).append(_r)   # baseline window
                _ctx.setdefault("log", []).append((t, _r, 0.0))
                return 0.0
            if "r0" not in _ctx:             # freeze on first crossing
                _r0s = _ctx["r0s"]
                _ctx["r0"] = (float(np.mean(_r0s[len(_r0s) // 2:]))
                             if _r0s else 0.0)
                print(f"dan-gate: baseline r0 = {_ctx['r0']:.2f} Hz",
                      flush=True)
            _v = (_r - _ctx["r0"]) * _g
            _mod = _v if _v > 0.0 else 0.0
            if _mod > 1.0:
                _mod = 1.0
            _ctx["log"].append((t, _r, _mod))
            return _mod if _cm is None else _mod * _cm
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
    if args.mbon_bias is not None and not args.chem_input:
        ap.error("--mbon-bias rides the chem channel machinery -- "
                 "requires --chem-input (any entry; the bias channel "
                 "spans the whole trial)")
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
        if args.mbon_bias is not None:
            # bci/dangate2: tonic MBON hyperpolarization as an extra
            # channel -- the chem machinery (fnmatch groups, GPU path,
            # chem_meta provenance) is reused wholesale.  "MBON*" hits
            # every MBONnn[@-like]@side type key (~97 cells); amp sign
            # passes straight through, so a negative value biases.
            cspec = dict(cspec)
            cspec["channels"] = list(cspec["channels"]) + [
                {"pop": "CEN", "group": "MBON*",
                 "amp_pa": float(args.mbon_bias),
                 "pulses": [[0.0, float(args.t_end)]]}]
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
            chans.append((pop, np.flatnonzero(sel),
                          float(ch["amp_pa"]) * float(args.chem_amp_scale),
                          [(float(a), float(b))
                           for a, b in ch["pulses"]], ch))
            _amp_s = float(ch["amp_pa"]) * float(args.chem_amp_scale)
            print(f"chem: {pop}[{ch['group']}] -> {int(sel.sum())} "
                  f"neurons ({', '.join(sorted(hits)[:4])}"
                  f"{'...' if len(hits) > 4 else ''}), "
                  f"{_amp_s:.0f} pA x {len(ch['pulses'])} pulses"
                  + (f" (x{args.chem_amp_scale:g} amp scale)"
                     if args.chem_amp_scale != 1.0 else ""))

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
        # the damp must NOT clobber an explicit --plastic-kcm-gain: the
        # KCM readout leg needs to escape the working point for MBON
        # firing to carry V (bci/dangate)
        _kcm_x = circuit["extra_edges"].get("KCM")
        _kcm_damp = _kcm_x is not None and \
            args.plastic_kcm_gain is None
        if "cen_gain" in cspec:
            # bci/seizure gain sweep: explicit recurrence gain, between
            # the safe working point (0.002) and the default (0.004)
            circuit["extra_edges"]["CEN_C"]["g_unit"] = \
                float(cspec["cen_gain"])
            if _kcm_damp:
                _kcm_x["g_unit"] = float(cspec["cen_gain"])
            print(f"chem calibration: CEN_C recurrence gain -> "
                  f"{cspec['cen_gain']} (explicit cen_gain)")
        elif not cspec.get("no_cen_damp", False) \
                and "CEN_C" in (circuit.get("extra_edges") or {}):
            circuit["extra_edges"]["CEN_C"]["g_unit"] = _G_CEN_CHEM
            if _kcm_damp:
                _kcm_x["g_unit"] = _G_CEN_CHEM
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
    pr_cpu = {g: np.zeros(n_field) for g in POP_RATE}
    prn_vec = (None if not args.pop_rate_neurons else
               {g: np.zeros(v[2], np.int32)
                for g, v in POP_RATE.items()})

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
        for _g, (_kind, _ix, _n_pr) in POP_RATE.items():
            pr_cpu[_g][j] = (
                1000.0 * int(
                    (sp[_g] if _kind == "POP"
                     else sp["CEN"][_ix]).sum()) / _n_pr)
        if prn_vec is not None:
            for _g, (_kind, _ix, _n_pr) in POP_RATE.items():
                prn_vec[_g] += (
                    sp[_g] if _kind == "POP"
                    else sp["CEN"][_ix]).astype(np.int32)

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
            import hashlib as _hl
            _si = np.load(args.plastic_state_in)
            _combined = _hl.md5("|".join(
                f"{_g}:{KCM_STATE[_g][2]}"
                for _g in sorted(KCM_STATE)).encode()).hexdigest()
            if str(_si["checksum"]) == _combined:
                _src = {g: _si[f"w_{g}"] for g in KCM_STATE}
            elif (len(KCM_STATE) == 1 and "KCM" in KCM_STATE
                    and str(_si["checksum"]) == KCM_STATE["KCM"][2]):
                _src = {"KCM": _si["w"]}    # legacy single-group file
            else:
                raise SystemExit(
                    "plastic state mismatch: edge ids differ from "
                    "this session's split -- refusing to load")
            for _g, _wv in _src.items():
                _pw = trial.exp[_g].plast_w
                assert _wv.shape == _pw.shape
                _pw.set(np.ascontiguousarray(_wv))
                print(f"plastic state IN {_g}: w mean "
                      f"{float(_wv.mean()):.4f} min "
                      f"{float(_wv.min()):.4f}", flush=True)
        if args.plastic_dan_gate:
            _DAN_CTX["mask"] = trial._mask("CEN")   # late-bound device
            # view; mod_fn reads the previous step's spikes
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
        pr_idx_g = {g: (None if v[0] == "POP" else cp.asarray(v[1]))
                    for g, v in POP_RATE.items()}
        pr_cnt_g = {g: cp.zeros(n_field, cp.int64) for g in POP_RATE}
        prn_vec_g = (None if not args.pop_rate_neurons else
                     {g: cp.zeros(v[2], cp.int32) for g, v in
                      POP_RATE.items()})
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
            for _g, _ix in pr_idx_g.items():
                _mk = (trial._mask(_g) if _ix is None
                       else trial._mask("CEN")[_ix])
                pr_cnt_g[_g][j] = _mk.sum()
                if prn_vec_g is not None:
                    prn_vec_g[_g] += _mk.astype(cp.int32)
            stim[j] = float(np.mean(luminance(t)))

        trial.run(lambda t: I_LUM * (luminance(t) - 1.0),
                  int(T_END / fp.DT_MS), on_record=record_gpu,
                  chem_fn=chem_fn, mod_fn=mod_fn)
        if args.std_gates:
            # bci/sleep3 diagnostics: the post-run depletion state of
            # every std-gated pool (d distribution + gated-current
            # weights) -- settles whether the gate actually bit
            for _gn in args.std_gates.split(";"):
                _pg = trial.exp.get(_gn.split(":")[0])
                if _pg is None or not _pg.std:
                    continue
                np.savez(OUT / f"_std_debug_{_gn.split(':')[0]}.npz",
                         d=cp.asnumpy(_pg.std_d), y=cp.asnumpy(_pg.y),
                         kick=cp.asnumpy(_pg.kick))
                print(f"std debug {_gn.split(':')[0]}: d mean "
                      f"{float(_pg.std_d.mean()):.4f} min "
                      f"{float(_pg.std_d.min()):.4f} frac<0.5 "
                      f"{float((_pg.std_d < 0.5).mean()):.3f}",
                      flush=True)
        if jbuf:
            flush_scalp_gpu(n_field - jbuf, jbuf)
        cp.cuda.runtime.deviceSynchronize()
        if args.plastic_mb:
            _state_out = {}
            for _g in KCM_STATE:
                if _g not in getattr(trial, "exp", {}):
                    continue
                _ws = trial.exp[_g].plast_w
                _el = trial.exp[_g].plast_elig
                print(f"plastic result {_g}: w_scale mean "
                      f"{float(_ws.mean()):.4f} min "
                      f"{float(_ws.min()):.4f} "
                      f"floor-hit {int((_ws <= 0.0201).sum())}/{_ws.size}"
                      f" | elig max {float(_el.max()):.4f}", flush=True)
                _state_out[f"w_{_g}"] = \
                    cp.asnumpy(_ws).astype(np.float32)
            if args.plastic_state_out and _state_out:
                _payload = {f"pre_{g}": v[0] for g, v in
                            KCM_STATE.items()}
                _payload.update({f"pre_type_{g}": v[1] for g, v in
                                 KCM_STATE.items()})
                _payload.update(_state_out)
                if len(KCM_STATE) == 1 and "KCM" in KCM_STATE:
                    # single group: legacy flat keys + legacy checksum
                    # (state-in takes the legacy branch; old analyze
                    # readers keep working)
                    _payload.update(
                        w=_state_out["w_KCM"],
                        n=_state_out["w_KCM"].size,
                        pre=KCM_STATE["KCM"][0],
                        pre_type=KCM_STATE["KCM"][1],
                        checksum=KCM_STATE["KCM"][2])
                else:
                    _payload["checksum"] = hashlib.md5("|".join(
                        f"{g}:{KCM_STATE[g][2]}"
                        for g in sorted(KCM_STATE)).encode()).hexdigest()
                np.savez(args.plastic_state_out, **_payload)
        if POP_RATE:
            _pr = {g: cp.asnumpy(pr_cnt_g[g]) * (1000.0 / POP_RATE[g][2])
                   for g in POP_RATE}
            np.savez(OUT / "_pop_rate.npz", **_pr)
            if prn_vec_g is not None:
                np.savez(OUT / "_pop_rate_neurons.npz",
                         **{g: cp.asnumpy(v) for g, v in
                            prn_vec_g.items()})
            _win = slice(int(1000), int(2000))    # ms -> 1-ms bins
            print("pop-rate (1-2 s, Hz/neuron): "
                  + ", ".join(f"{g} {_pr[g][_win].mean():.2f}"
                              for g in _pr), flush=True)
        if args.plastic_dan_gate and _DAN_CTX.get("log"):
            np.save(OUT / "_dan_trace.npy",
                    np.asarray(_DAN_CTX["log"], dtype=np.float64))
            _lg = _DAN_CTX["log"]
            _mmax = max(m for _, _, m in _lg)
            print(f"dan-gate: trace {len(_lg)} steps -> "
                  f"{OUT / '_dan_trace.npy'} | r max "
                  f"{max(r for _, r, _ in _lg):.1f} Hz, r0 "
                  f"{_DAN_CTX.get('r0', float('nan')):.2f} Hz, "
                  f"mod max {_mmax:.3f}, mod integral "
                  f"{sum(m for _, _, m in _lg) * float(fp.DT_MS):.1f} ms",
                  flush=True)
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
    if POP_RATE and not args.gpu:
        np.savez(OUT / "_pop_rate.npz", **pr_cpu)
        if prn_vec is not None:
            np.savez(OUT / "_pop_rate_neurons.npz", **prn_vec)
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
