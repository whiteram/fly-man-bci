"""Shared simulation pipeline for the visual cascade.

Single code path used by viz/export_data.py AND
scripts/calibrate_working_point.py -- until this module existed the two
scripts each maintained a copy of the loop and had already begun to drift.

Circuit: exp015 both-lobe cascade (the exp005 left-lobe dict is still
accepted -- the interface is identical); extra regions are OPTIONAL and
built only when the circuit dict carries extra_pops/extra_edges specs
(see ffbm.regions for assembly and the region on/off config).

Mechanism stack (see docs/TECHNICAL.md §5):
  - OU background noise (ColoredCurrentNoise) on all four populations,
    sigma per population (cal["OU"]); no white noise anywhere
  - RL/LM synapses: current-based, dataset signs folded into weights
  - MT synapses: conductance-based (dataset sign -> E_rev), unitless
    gating, g_unit scaling, semi-implicit membrane integration
  - per-edge transmission delays: SYN_DELAY_MS + site distance / v_axon,
    uniform jitter +- DELAY_JITTER_MS
The luminance protocol is injected as a callable so the naturalistic (viz)
and flash (calibration) protocols share the identical dynamics.
"""

from __future__ import annotations

import numpy as np

from .params import cal as _params_cal
from .simulation import (ColoredCurrentNoise, ExponentialSynapses,
                        GradedSynapsePool, LIFPopulation)

DT_MS = 0.5

# single source of truth: src/ffbm/params.py (docs/PARAMS.md, regenerate
# with `python -m ffbm.params`); CAL is assembled from that registry
CAL = _params_cal()


class PhotoCascade:
    """Per-R two-stage low-pass + slow adaptation (exp002/005 kinetics,
    identical equations -- vectorised here so the pipeline has no
    experiment-module imports)."""

    def __init__(self, n, dt, tau=10.0, tau_adapt=800.0, sag=0.30):
        self.k = 1.0 - np.exp(-dt / tau)
        self.ka = 1.0 - np.exp(-dt / tau_adapt)
        self.sag = sag
        self.y1 = np.zeros(n)
        self.y2 = np.zeros(n)
        self.g = np.ones(n)

    def step(self, inc_pa):
        self.y1 += self.k * (inc_pa - self.y1)
        self.y2 += self.k * (self.y1 - self.y2)
        target = np.where(np.asarray(inc_pa) > 0.0, 1.0 - self.sag, 1.0)
        self.g += self.ka * (target - self.g)
        return self.g * self.y2


def _indices(ids):
    idx = np.full(int(np.asarray(ids).max()) + 1, -1, dtype=np.int64)
    idx[ids] = np.arange(len(ids))
    return idx


def _release(v, rmap):
    lo, hi = rmap
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def build_stack(circuit, cal, rng):
    """Synapses / populations / noises / bases for one trial.

    cal["LAMINA_MECHANISTIC"] switches the lamina to the exp013 real
    biophysics: graded R1-6 (release rate from membrane potential) ->
    histamine-gated Cl conductance on graded L1/L2 (E_cl = -70 mV,
    depolarized leak v_K = -25 mV; I_L_BASE hack unused) -> graded
    release onto Mi/Tm with dataset signs. The legacy branch keeps the
    exp009 current-based proxy for regression."""
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids, l_ids = circuit["r_ids"], circuit["l_ids"]
    mid_ids, t45_ids = circuit["mid_ids"], circuit["t45_ids"]
    n_r, n_l = len(r_ids), len(l_ids)
    n_mid, n_t45 = len(mid_ids), len(t45_ids)
    l_index, mid_index, t45_index = (_indices(l_ids), _indices(mid_ids),
                                     _indices(t45_ids))

    e_rl = circuit["e_rl"].copy()
    e_lm = circuit["e_lm"].copy()
    e_mt = circuit["e_mt"]
    mech = bool(cal.get("LAMINA_MECHANISTIC"))

    def edges(e_sub):
        return (e_sub["body_pre"].to_numpy(np.int64),
                e_sub["body_post"].to_numpy(np.int64))

    def delays(e_sub):
        pre, post = edges(e_sub)
        if len(pre) == 0:
            return np.zeros(0)
        pa = np.array([pp[a] for a in pre], dtype=np.float64)
        qa = np.array([qq[b] for b in post], dtype=np.float64)
        d = np.linalg.norm(pa - qa, axis=1)
        return (cal["SYN_DELAY_MS"] + d / cal["V_AXON_UM_PER_MS"]
                + rng.uniform(-cal["DELAY_JITTER_MS"],
                              cal["DELAY_JITTER_MS"], len(d)))

    syn = {}
    if mech:
        # graded pools take PRE-SORTED inputs in the kernel edge order;
        # pre arrays are ROW indices (rows are monotone in body id, so
        # the pool's internal lexsort is the identity and pool.y stays
        # aligned with the per-edge forward kernels).
        r_row = {int(b): i for i, b in enumerate(r_ids.tolist())}
        l_row = {int(b): i for i, b in enumerate(l_ids.tolist())}
        pre_rl, post_rl = edges(e_rl)
        syn["RL"] = GradedSynapsePool(
            np.array([r_row[int(b)] for b in pre_rl]), post_rl,
            e_rl["weight"].to_numpy(np.float32), l_index, dt=DT_MS,
            tau_s=cal["TAU_RL"], n_post=n_l,
            g_unit=cal["G_UNIT_HIST_NS"], e_rev=cal["E_CL_MV"])
        pre_lm, post_lm = edges(e_lm)
        syn["LM"] = GradedSynapsePool(
            np.array([l_row[int(b)] for b in pre_lm]), post_lm,
            e_lm["weight"].to_numpy(np.float32), mid_index, dt=DT_MS,
            tau_s=cal["TAU_LM"], n_post=n_mid,
            g_unit=cal["G_UNIT_LM_NS"], e_rev=cal["E_REV_EXC"],
            sign=e_lm["sign"].to_numpy(np.float32),
            e_rev_inh=cal["E_REV_INH"])
    else:
        e_rl["weight"] = e_rl["weight"] * e_rl["sign"]
        e_lm["weight"] = e_lm["weight"] * e_lm["sign"]
        for name, e_sub, tgt, n_t, gain, tau in (
                ("RL", e_rl, l_index, n_l, cal["GAIN_RL"], cal["TAU_RL"]),
                ("LM", e_lm, mid_index, n_mid, cal["GAIN_LM"],
                 cal["TAU_LM"])):
            syn[name] = ExponentialSynapses(
                *edges(e_sub), e_sub["weight"].to_numpy(np.float32), tgt,
                dt=DT_MS, gain=gain, tau_s=tau, n_post=n_t,
                delay_ms=delays(e_sub))
    for mt in cal["MID_TAU_S"]:
        e_sub = e_mt[mt]
        syn[f"MT_{mt}"] = ExponentialSynapses(
            *edges(e_sub), e_sub["weight"].to_numpy(np.float32), t45_index,
            dt=DT_MS, gain=1.0, tau_s=cal["MID_TAU_S"][mt], n_post=n_t45,
            sign=e_sub["sign"].to_numpy(np.float32),
            delay_ms=delays(e_sub), conductance=True,
            g_unit=cal["G_UNIT_MT"], e_rev_exc=cal["E_REV_EXC"],
            e_rev_inh=cal["E_REV_INH"])

    # optional extra regions (ffbm.regions assembly): each entry in
    # circuit["extra_pops"] is a population spec {ids, tau_ms,
    # t_refrac_ms, i_base, ou_sigma}; each entry in circuit
    # ["extra_edges"] is a spike-driven conductance edge group
    # {pre: (pop names), post: pop name, table, tau_s, g_unit}. Both
    # dicts are absent (or empty) when no region beyond the visual
    # cascade is active -- zero cost.
    extra_pops = circuit.get("extra_pops") or {}
    extra_edges = circuit.get("extra_edges") or {}

    def _res(v):
        """Spec value: a number, or a CAL key name resolved per run (so
        working-point grids overriding cal actually reach the region)."""
        return cal[v] if isinstance(v, str) else v

    pop_index = {name: _indices(spec["ids"])
                 for name, spec in extra_pops.items()}
    for name, spec in extra_edges.items():
        e_sub = spec["table"]
        post = spec["post"]
        syn[name] = ExponentialSynapses(
            *edges(e_sub), e_sub["weight"].to_numpy(np.float32),
            pop_index[post], dt=DT_MS, gain=1.0, tau_s=_res(spec["tau_s"]),
            n_post=len(extra_pops[post]["ids"]),
            delay_ms=delays(e_sub), conductance=True,
            g_unit=_res(spec["g_unit"]), e_rev_exc=cal["E_REV_EXC"],
            e_rev_inh=cal["E_REV_INH"])

    pops = {"R": LIFPopulation(n_r, DT_MS, tau_m=cal["LIF"]["R"][0],
                               t_refrac=(1e9 if mech
                                         else cal["LIF"]["R"][1]),
                               R_m=cal["RIN_GOHM"], v_th=(1e9 if mech
                                                          else -50.0)),
            "L": LIFPopulation(n_l, DT_MS, tau_m=cal["LIF"]["L"][0],
                               t_refrac=(1e9 if mech
                                         else cal["LIF"]["L"][1]),
                               R_m=cal["RIN_GOHM"],
                               v_th=(1e9 if mech else -50.0),
                               v_rest=(cal["V_K_LMC_MV"] if mech
                                       else -70.0)),
            "MID": LIFPopulation(n_mid, DT_MS, tau_m=cal["LIF"]["MID"][0],
                                 t_refrac=cal["LIF"]["MID"][1],
                                 R_m=cal["RIN_GOHM"]),
            "T45": LIFPopulation(n_t45, DT_MS, tau_m=cal["LIF"]["T45"][0],
                                 t_refrac=cal["LIF"]["T45"][1],
                                 R_m=cal["RIN_GOHM"])}
    noises = {k: ColoredCurrentNoise(n, DT_MS, rng, tau_n=cal["OU_TAU_MS"],
                                     sigma=cal["OU"][k])
              for k, n in (("R", n_r), ("L", n_l), ("MID", n_mid),
                           ("T45", n_t45))}
    for name, spec in extra_pops.items():
        n_x = len(spec["ids"])
        pops[name] = LIFPopulation(n_x, DT_MS, tau_m=spec["tau_ms"],
                                   t_refrac=spec["t_refrac_ms"],
                                   R_m=cal["RIN_GOHM"])
        noises[name] = ColoredCurrentNoise(
            n_x, DT_MS, rng, tau_n=cal["OU_TAU_MS"],
            sigma=spec["ou_sigma"])
    t45_type = circuit["t45_type"]
    is_t4 = np.array([str(x).startswith("T4") for x in t45_type])
    is_t5 = np.array([str(x).startswith("T5") for x in t45_type])
    t45_base = np.zeros(n_t45)
    t45_base[is_t4] = cal["I_T4_BASE"]
    t45_base[is_t5] = cal.get("I_T5_BASE", 0.0)
    return {"syn": syn, "pops": pops, "noises": noises, "mech": mech,
            "r_ids": r_ids, "l_ids": l_ids, "mid_ids": mid_ids,
            "t45_ids": t45_ids,
            "n_r": n_r, "n_l": n_l, "n_mid": n_mid, "n_t45": n_t45,
            "is_t4": is_t4, "is_t5": is_t5, "l_index": l_index,
            "mid_index": mid_index, "t45_index": t45_index,
            "extra_pops": extra_pops,
            "extra_ids": {n: s["ids"] for n, s in extra_pops.items()},
            "extra_base": {n: np.full(len(s["ids"]), _res(s["i_base"]))
                           for n, s in extra_pops.items()},
            "extra_in": {n: [g for g, es in extra_edges.items()
                             if es["post"] == n]
                         for n in extra_pops},
            "extra_pre": {g: es["pre"] for g, es in extra_edges.items()},
            "l_base": (np.zeros(n_l) if mech
                       else np.full(n_l, cal["I_L_BASE"])),
            "mid_base": np.full(n_mid, cal["I_MID_BASE"]),
            "t45_base": t45_base,
            "r_release": _release, "l_release": _release, "cal": cal}


def simulate(circuit, cal, lum_inc_fn, seed, t_end_ms, on_sample=None,
             chem_fn=None):
    """Run one trial of the cascade.

    lum_inc_fn(t_ms) -> phototransduction input increment, pA, scalar or
    (n_r,). chem_fn(t_ms) -> {extra pop name: per-neuron RAW stimulus
    current, pA} (chemosensory drive, exp019) -- each active pop gets a
    PhotoCascade transduction stage (10 ms low-pass x2 + 800 ms
    adaptation to a 30% pedestal), so the per-neuron value here is the
    physical stimulus amplitude, not the drive the soma sees.
    on_sample(j, k, t, stack, inc_f, spikes) fires every 2 steps
    (1 kHz); spikes = {"R","L","MID","T45"} boolean masks. The RNG draw
    order is fixed (R, L, MID, T45 noises) so any given seed reproduces
    exactly across callers; chem_fn is deterministic stimulus and does
    not participate in the draw order.
    """
    rng = np.random.default_rng(seed)
    st = build_stack(circuit, cal, rng)
    pops, syn, noises = st["pops"], st["syn"], st["noises"]
    photo = PhotoCascade(st["n_r"], DT_MS)
    # chemosensory transduction cascades (exp019): ORN/GRN share the
    # visual cascade kinetics (10 ms low-pass x2 + 800 ms adaptation to
    # a 30% pedestal) -- real ORN responses are phasic-tonic, and the
    # adapted pedestal keeps central loops from latching (README)
    chem_casc = {name: PhotoCascade(len(ids), DT_MS)
                 for name, ids in st["extra_ids"].items()} \
        if chem_fn is not None else None
    if chem_casc is not None:
        # per-pop reusable buffers: a fresh zeros() per step churned
        # ~0.8 MB/step and ballooned RSS into swap (exp019 fast-run
        # finding); driven pops get one scratch each, undriven pops
        # share one preallocated zero per pop
        chem_zero = {name: np.zeros(len(ids))
                     for name, ids in st["extra_ids"].items()}
        chem_live = set()
    n_steps = int(t_end_ms / DT_MS)
    for k in range(n_steps):
        t = k * DT_MS
        inc_f = photo.step(lum_inc_fn(t))
        chem_inc = None
        if chem_fn is not None:
            raw = chem_fn(t)
            chem_inc = {}
            for name in st["extra_ids"]:
                if name in raw:
                    chem_live.add(name)
                    chem_inc[name] = chem_casc[name].step(raw[name])
                elif name in chem_live:
                    # stimulus off: keep decaying the cascade state,
                    # but only while it is away from baseline (the
                    # epsilon check skips ~pure-zero steps cheaply)
                    c = chem_casc[name]
                    if np.abs(c.y1).max() > 1e-9 \
                            or np.abs(c.y2).max() > 1e-9:
                        chem_inc[name] = c.step(chem_zero[name])
        if st["mech"]:
            sp_r = pops["R"].step(cal["I_R_BASE"] + inc_f
                                  + noises["R"].step())
            r_rl = st["r_release"](pops["R"].v, cal["R_RELEASE_MAP_MV"])
            syn["RL"].step(r_rl)
            di_l, dg_l = syn["RL"].to_neuron_drive()
            sp_l = pops["L"].step(di_l + noises["L"].step(), dg_l)
            r_lm = st["l_release"](pops["L"].v, cal["L_RELEASE_MAP_MV"])
            syn["LM"].step(r_lm)
            di_m, dg_m = syn["LM"].to_neuron_drive()
            sp_mid = pops["MID"].step(st["mid_base"] + di_m
                                      + noises["MID"].step(), dg_m)
        else:
            sp_r = pops["R"].step(cal["I_R_BASE"] + inc_f
                                  + noises["R"].step())
            sp_l = pops["L"].step(st["l_base"]
                                  + syn["RL"].to_neuron_current()
                                  + noises["L"].step())
            sp_mid = pops["MID"].step(st["mid_base"]
                                      + syn["LM"].to_neuron_current()
                                      + noises["MID"].step())
        i_t45 = st["t45_base"] + noises["T45"].step()
        g_t45 = np.zeros(st["n_t45"])
        for mt in cal["MID_TAU_S"]:
            di, dg = syn[f"MT_{mt}"].to_neuron_drive()
            i_t45 += di
            g_t45 += dg
        sp_t45 = pops["T45"].step(i_t45, g_t45)
        # optional extra regions, stepped in declaration (dependency)
        # order: each pop consumes this step's drive from its incoming
        # edge groups, so a chain MID->VPN->CB propagates within the
        # same step like the core cascade does
        sp_extra = {}
        for name in st["extra_pops"]:
            i_x = st["extra_base"][name] + noises[name].step()
            if chem_inc is not None and name in chem_inc:
                i_x = i_x + chem_inc[name]
            g_x = np.zeros(len(st["extra_ids"][name]))
            for grp in st["extra_in"][name]:
                di, dg = syn[grp].to_neuron_drive()
                i_x += di
                g_x += dg
            sp_extra[name] = pops[name].step(i_x, g_x)
        if not st["mech"]:
            syn["RL"].step(st["r_ids"][sp_r])
            syn["LM"].step(st["l_ids"][sp_l])
        spiked_mid = st["mid_ids"][sp_mid]
        for mt in cal["MID_TAU_S"]:
            syn[f"MT_{mt}"].step(spiked_mid)

        def spiked_ids(pop, sp_r_=sp_r, sp_l_=sp_l, sp_mid_=sp_mid,
                       sp_t45_=sp_t45, sp_extra_=sp_extra):
            if pop == "R":
                return st["r_ids"][sp_r_]
            if pop == "L":
                return st["l_ids"][sp_l_]
            if pop == "MID":
                return st["mid_ids"][sp_mid_]
            if pop == "T45":
                return st["t45_ids"][sp_t45_]
            return st["extra_ids"][pop][sp_extra_[pop]]

        for grp, pre_names in st["extra_pre"].items():
            syn[grp].step(np.concatenate(
                [spiked_ids(p) for p in pre_names]))
        if k % 2 == 0 and on_sample is not None:
            on_sample(k // 2, k, t, st, inc_f,
                      {"R": sp_r, "L": sp_l, "MID": sp_mid,
                       "T45": sp_t45, **sp_extra})
    return st
