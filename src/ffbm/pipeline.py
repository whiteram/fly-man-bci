"""Shared simulation pipeline for the visual cascade.

Single code path used by viz/export_data.py AND
scripts/calibrate_working_point.py -- until this module existed the two
scripts each maintained a copy of the loop and had already begun to drift.

Circuit: exp015 both-lobe cascade (the exp005 left-lobe dict is still
accepted -- the interface is identical); exp016 VPN -> central-brain
layers are OPTIONAL and built only when the circuit dict carries
vpn_ids/cb_ids.

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
        d = np.array([np.linalg.norm(pp[a] - qq[b])
                      for a, b in zip(pre, post)])
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

    # exp016 optional visual-downstream layers: VPNs (LC/LPLC classes)
    # and their central-brain targets, built only when the circuit
    # carries the extra keys; ACh (excitatory) throughout
    vpn_ids = circuit.get("vpn_ids")
    if vpn_ids is not None:
        cb_ids = circuit["cb_ids"]
        n_vpn, n_cb = len(vpn_ids), len(cb_ids)
        vpn_index = _indices(vpn_ids)
        cb_index = _indices(cb_ids)
        e_m2v = circuit["e_m2v"]
        e_v2c = circuit["e_v2c"]
        syn["M2V"] = ExponentialSynapses(
            *edges(e_m2v), e_m2v["weight"].to_numpy(np.float32), vpn_index,
            dt=DT_MS, gain=1.0, tau_s=cal["TAU_V_MS"], n_post=n_vpn,
            delay_ms=delays(e_m2v), conductance=True,
            g_unit=cal["G_UNIT_M2V"], e_rev_exc=cal["E_REV_EXC"],
            e_rev_inh=cal["E_REV_INH"])
        syn["V2C"] = ExponentialSynapses(
            *edges(e_v2c), e_v2c["weight"].to_numpy(np.float32), cb_index,
            dt=DT_MS, gain=1.0, tau_s=cal["TAU_V_MS"], n_post=n_cb,
            delay_ms=delays(e_v2c), conductance=True,
            g_unit=cal["G_UNIT_V2C"], e_rev_exc=cal["E_REV_EXC"],
            e_rev_inh=cal["E_REV_INH"])
        pops["VPN"] = LIFPopulation(n_vpn, DT_MS,
                                    tau_m=cal["LIF"]["MID"][0],
                                    t_refrac=cal["LIF"]["MID"][1],
                                    R_m=cal["RIN_GOHM"])
        pops["CB"] = LIFPopulation(n_cb, DT_MS, tau_m=cal["LIF"]["MID"][0],
                                   t_refrac=cal["LIF"]["MID"][1],
                                   R_m=cal["RIN_GOHM"])
        noises["VPN"] = ColoredCurrentNoise(
            n_vpn, DT_MS, rng, tau_n=cal["OU_TAU_MS"],
            sigma=cal["OU_VPN_CB"])
        noises["CB"] = ColoredCurrentNoise(
            n_cb, DT_MS, rng, tau_n=cal["OU_TAU_MS"], sigma=cal["OU_VPN_CB"])

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
            "vpn_ids": (None if vpn_ids is None else vpn_ids),
            "cb_ids": circuit.get("cb_ids"),
            "l_base": (np.zeros(n_l) if mech
                       else np.full(n_l, cal["I_L_BASE"])),
            "mid_base": np.full(n_mid, cal["I_MID_BASE"]),
            "t45_base": t45_base,
            "v_base": (None if vpn_ids is None
                       else np.full(len(vpn_ids), cal["I_V_BASE"])),
            "c_base": (None if vpn_ids is None
                       else np.full(len(circuit["cb_ids"]),
                                    cal["I_C_BASE"])),
            "r_release": _release, "l_release": _release, "cal": cal}


def simulate(circuit, cal, lum_inc_fn, seed, t_end_ms, on_sample=None):
    """Run one trial of the cascade.

    lum_inc_fn(t_ms) -> phototransduction input increment, pA, scalar or
    (n_r,). on_sample(j, k, t, stack, inc_f, spikes) fires every 2 steps
    (1 kHz); spikes = {"R","L","MID","T45"} boolean masks. The RNG draw
    order is fixed (R, L, MID, T45 noises) so any given seed reproduces
    exactly across callers.
    """
    rng = np.random.default_rng(seed)
    st = build_stack(circuit, cal, rng)
    pops, syn, noises = st["pops"], st["syn"], st["noises"]
    photo = PhotoCascade(st["n_r"], DT_MS)
    n_steps = int(t_end_ms / DT_MS)
    for k in range(n_steps):
        t = k * DT_MS
        inc_f = photo.step(lum_inc_fn(t))
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
        sp_v = sp_cb = None
        if "M2V" in syn:
            di_v, dg_v = syn["M2V"].to_neuron_drive()
            sp_v = pops["VPN"].step(st["v_base"] + di_v
                                    + noises["VPN"].step(), dg_v)
            di_c, dg_c = syn["V2C"].to_neuron_drive()
            sp_cb = pops["CB"].step(st["c_base"] + di_c
                                    + noises["CB"].step(), dg_c)
        if not st["mech"]:
            syn["RL"].step(st["r_ids"][sp_r])
            syn["LM"].step(st["l_ids"][sp_l])
        spiked_mid = st["mid_ids"][sp_mid]
        for mt in cal["MID_TAU_S"]:
            syn[f"MT_{mt}"].step(spiked_mid)
        if "M2V" in syn:
            syn["M2V"].step(np.concatenate(
                [spiked_mid, st["t45_ids"][sp_t45]]))
            syn["V2C"].step(st["vpn_ids"][sp_v])
        if k % 2 == 0 and on_sample is not None:
            on_sample(k // 2, k, t, st, inc_f,
                      {"R": sp_r, "L": sp_l, "MID": sp_mid,
                       "T45": sp_t45, "VPN": sp_v, "CB": sp_cb})
    return st
