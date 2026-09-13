"""Shared simulation pipeline for the left-lobe cascade.

Single code path used by viz/export_data.py AND
scripts/calibrate_working_point.py -- until this module existed the two
scripts each maintained a copy of the loop and had already begun to drift.

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
from .simulation import ColoredCurrentNoise, ExponentialSynapses, LIFPopulation

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


def build_stack(circuit, cal, rng):
    """Synapses / populations / noises / bases for one trial."""
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids, l_ids = circuit["r_ids"], circuit["l_ids"]
    mid_ids, t45_ids = circuit["mid_ids"], circuit["t45_ids"]
    n_r, n_l = len(r_ids), len(l_ids)
    n_mid, n_t45 = len(mid_ids), len(t45_ids)
    l_index, mid_index, t45_index = (_indices(l_ids), _indices(mid_ids),
                                     _indices(t45_ids))

    e_rl = circuit["e_rl"].copy()
    e_rl["weight"] = e_rl["weight"] * e_rl["sign"]
    e_lm = circuit["e_lm"].copy()
    e_lm["weight"] = e_lm["weight"] * e_lm["sign"]
    e_mt = circuit["e_mt"]

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
    for name, e_sub, tgt, n_t, gain, tau in (
            ("RL", e_rl, l_index, n_l, cal["GAIN_RL"], cal["TAU_RL"]),
            ("LM", e_lm, mid_index, n_mid, cal["GAIN_LM"], cal["TAU_LM"])):
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

    pops = {name: LIFPopulation(n, DT_MS, tau_m=tm, t_refrac=tr,
                                R_m=cal["RIN_GOHM"])
            for name, (n, (tm, tr)) in {
                "R": (n_r, cal["LIF"]["R"]), "L": (n_l, cal["LIF"]["L"]),
                "MID": (n_mid, cal["LIF"]["MID"]),
                "T45": (n_t45, cal["LIF"]["T45"])}.items()}
    noises = {k: ColoredCurrentNoise(n, DT_MS, rng, tau_n=cal["OU_TAU_MS"],
                                     sigma=cal["OU"][k])
              for k, n in (("R", n_r), ("L", n_l), ("MID", n_mid),
                           ("T45", n_t45))}
    t45_type = circuit["t45_type"]
    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    t45_base = np.zeros(n_t45)
    t45_base[is_t4] = cal["I_T4_BASE"]
    return {"syn": syn, "pops": pops, "noises": noises,
            "r_ids": r_ids, "l_ids": l_ids, "mid_ids": mid_ids,
            "n_r": n_r, "n_l": n_l, "n_mid": n_mid, "n_t45": n_t45,
            "is_t4": is_t4, "is_t5": is_t5, "l_index": l_index,
            "mid_index": mid_index, "t45_index": t45_index,
            "l_base": np.full(n_l, cal["I_L_BASE"]),
            "mid_base": np.full(n_mid, cal["I_MID_BASE"]),
            "t45_base": t45_base,
            "cal": cal}


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
        sp_r = pops["R"].step(cal["I_R_BASE"] + inc_f + noises["R"].step())
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
        syn["RL"].step(st["r_ids"][sp_r])
        syn["LM"].step(st["l_ids"][sp_l])
        spiked_mid = st["mid_ids"][sp_mid]
        for mt in cal["MID_TAU_S"]:
            syn[f"MT_{mt}"].step(spiked_mid)
        if k % 2 == 0 and on_sample is not None:
            on_sample(k // 2, k, t, st, inc_f,
                      {"R": sp_r, "L": sp_l, "MID": sp_mid,
                       "T45": sp_t45})
    return st
