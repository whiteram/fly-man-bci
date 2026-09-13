"""Working-point calibration, round 3: new mechanisms enabled.

Round-2 winner (score 5.8) still had two structural artifacts:
  - white noise (0.3 mV membrane fluctuation) -> knife-edge f-I, T4 dark
    needed a hand-tuned 195 pA base sitting exactly on the edge;
  - current-based MT synapses (fixed pA per spike regardless of v_post).

Round 3 calibrates under the upgraded physiology:
  - ColoredCurrentNoise (OU, tau=8 ms) on the T4/T5 population -> ~3-5 mV
    membrane fluctuation, graded stochastic firing;
  - conductance-based MT synapses (g_unit, E_rev by sign, driving-force
    dependent) with the raw dataset signs;
  - per-edge transmission delays (1 ms synaptic + dist/300um/ms axonal)
    on every stage.

Protocol per combo: 4 s (2 s dark + 2 s 350 pA flash), rates only.
Literature spontaneous-rate windows (Hz): MID 5-20, T4 2-12, T5 2-15.

Run from repository root:
    python scripts/calibrate_working_point.py     (~15 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

import run as exp005
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp009_run", ROOT / "experiments" / "exp009_lamina_polarity" / "run.py")
exp009 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exp009)

from ffbm.simulation import (ColoredCurrentNoise, ExponentialSynapses,
                             LIFPopulation)

OUT = ROOT / "scripts" / "outputs"
DT = exp005.DT
T_DARK, T_FLASH = 2000.0, 2000.0
T_END = T_DARK + T_FLASH
LEVEL_PA = 350.0
SEED = 42
V_AXON_UM_PER_MS = 300.0     # 0.3 m/s, small fly axons
SYN_DELAY_MS = 1.0

TARGETS = {"MID": (5.0, 20.0), "T4": (2.0, 12.0), "T5": (2.0, 15.0)}

# (I_MID_BASE, I_T4_BASE, OU_SIGMA_T45, G_UNIT_MT)
COMBOS = [
    (90.0, 175.0, 45.0, 0.020),
    (90.0, 165.0, 45.0, 0.020),
    (90.0, 175.0, 60.0, 0.020),
    (90.0, 165.0, 60.0, 0.020),
    (90.0, 175.0, 60.0, 0.030),
    (90.0, 165.0, 45.0, 0.030),
]


def edge_delays(pre_ids, post_ids, pp, qq):
    d = np.array([np.linalg.norm(pp[a] - qq[b])
                  for a, b in zip(pre_ids, post_ids)])
    return SYN_DELAY_MS + d / V_AXON_UM_PER_MS


def run_combo(circuit, params):
    i_mid_base, i_t4_base, ou_sigma, g_unit = params
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids, l_ids = circuit["r_ids"], circuit["l_ids"]
    mid_ids, t45_ids = circuit["mid_ids"], circuit["t45_ids"]
    n_r, n_l = len(r_ids), len(l_ids)
    n_mid, n_t45 = len(mid_ids), len(t45_ids)
    t45_type = circuit["t45_type"]
    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])

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

    def edges_of(e_sub):
        return (e_sub["body_pre"].to_numpy(np.int64),
                e_sub["body_post"].to_numpy(np.int64))

    syn = {
        "RL": ExponentialSynapses(
            *edges_of(e_rl), e_rl["weight"].to_numpy(np.float32), l_index,
            dt=DT, gain=exp009.GAIN_RL, tau_s=exp009.TAU_RL, n_post=n_l,
            delay_ms=edge_delays(*edges_of(e_rl), pp, qq)),
        "LM": ExponentialSynapses(
            *edges_of(e_lm), e_lm["weight"].to_numpy(np.float32), mid_index,
            dt=DT, gain=exp009.GAIN_LM, tau_s=exp009.TAU_LM, n_post=n_mid,
            delay_ms=edge_delays(*edges_of(e_lm), pp, qq)),
    }
    for mt in exp005.MID_TYPES:
        e_sub = e_mt[mt]
        syn[f"MT_{mt}"] = ExponentialSynapses(
            *edges_of(e_sub),
            e_sub["weight"].to_numpy(np.float32), t45_index, dt=DT,
            gain=1.0,                       # unitless gating
            tau_s=exp005.KINETICS["differentiated"][mt], n_post=n_t45,
            sign=e_sub["sign"].to_numpy(np.float32),
            delay_ms=edge_delays(*edges_of(e_sub), pp, qq),
            conductance=True, g_unit=g_unit, e_rev_exc=0.0, e_rev_inh=-80.0)

    rng = np.random.default_rng(SEED)
    pop_r = LIFPopulation(n_r, DT, tau_m=exp005.R_TAU, t_refrac=exp005.R_REF,
                          R_m=exp005.RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=exp005.L_TAU, t_refrac=exp005.L_REF,
                          R_m=exp005.RIN)
    pop_mid = LIFPopulation(n_mid, DT, tau_m=exp005.MID_TAU,
                            t_refrac=exp005.MID_REF, R_m=exp005.RIN)
    pop_t45 = LIFPopulation(n_t45, DT, tau_m=exp005.T45_TAU,
                            t_refrac=exp005.T45_REF, R_m=exp005.RIN)
    photo = exp005.PhotoCascadeVector(n_r, DT)
    noise_t45 = ColoredCurrentNoise(n_t45, DT, rng, tau_n=8.0,
                                    sigma=ou_sigma)
    l_base = np.full(n_l, exp009.I_L_BASE)
    mid_base = np.full(n_mid, i_mid_base)
    t45_base = np.zeros(n_t45)
    t45_base[is_t4] = i_t4_base

    n_steps = int(T_END / DT)
    n_out = n_steps // 2
    cnt = {k: np.zeros(n_out) for k in ("R", "L", "MID", "T4", "T5")}
    for k in range(n_steps):
        t = k * DT
        inc = np.full(n_r, LEVEL_PA) if T_DARK <= t < T_END \
            else np.zeros(n_r)
        inc_f = photo.step(inc)
        sp_r = pop_r.step(exp005.I_R_BASE + inc_f
                          + rng.normal(0, exp005.NOISE_SD["R"], n_r))
        sp_l = pop_l.step(l_base + syn["RL"].to_neuron_current()
                          + rng.normal(0, exp005.NOISE_SD["L"], n_l))
        sp_mid = pop_mid.step(mid_base + syn["LM"].to_neuron_current()
                              + rng.normal(0, exp005.NOISE_SD["MID"], n_mid))
        i_t45 = t45_base + noise_t45.step()
        g_t45 = np.zeros(n_t45)
        for mt in exp005.MID_TYPES:
            di, dg = syn[f"MT_{mt}"].to_neuron_drive()
            i_t45 += di
            g_t45 += dg
        sp_t45 = pop_t45.step(i_t45, g_t45)
        syn["RL"].step(r_ids[sp_r])
        syn["LM"].step(l_ids[sp_l])
        spiked_mid = mid_ids[sp_mid]
        for mt in exp005.MID_TYPES:
            syn[f"MT_{mt}"].step(spiked_mid)
        if k % 2 == 0:
            j = k // 2
            cnt["R"][j] = sp_r.sum() * 1000.0 / n_r
            cnt["L"][j] = sp_l.sum() * 1000.0 / n_l
            cnt["MID"][j] = sp_mid.sum() * 1000.0 / n_mid
            cnt["T4"][j] = sp_t45[is_t4].sum() * 1000.0 / is_t4.sum()
            cnt["T5"][j] = sp_t45[is_t5].sum() * 1000.0 / is_t5.sum()

    def w(t0, t1):
        m = slice(int(t0), int(t1))
        return {k: round(float(cnt[k][m].mean()), 1) for k in cnt}

    return {"dark": w(500.0, T_DARK), "flash": w(T_DARK + 300.0, T_END)}


def score(r):
    s = 0.0
    for key, (lo, hi) in TARGETS.items():
        for win, weight in (("dark", 1.0), ("flash", 0.35)):
            v = r[win][key]
            s += weight * (max(0.0, lo - v) + max(0.0, v - hi)) ** 2
    if r["flash"]["T4"] < 5.0:
        s += 400.0
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("building circuit ...")
    circuit = exp005.build_circuit()
    rows = []
    for params in COMBOS:
        print(f"combo I_MID={params[0]:.0f} T4base={params[1]:.0f} "
              f"OU={params[2]:.0f} g_unit={params[3]:.3f} ...", flush=True)
        r = run_combo(circuit, params)
        r["params"] = {"i_mid_base": params[0], "i_t4_base": params[1],
                       "ou_sigma_t45": params[2], "g_unit_mt": params[3]}
        r["score"] = round(score(r), 2)
        rows.append(r)
        print(f"   dark {r['dark']}  flash {r['flash']}  score {r['score']}")
    best = min(rows, key=lambda r: r["score"])
    print("\nBEST:", json.dumps(best, indent=1))
    (OUT / "working_point_calibration_r3.json").write_text(
        json.dumps({"combos": rows, "best": best}, indent=1))
    print(f"written to {OUT / 'working_point_calibration_r3.json'}")


if __name__ == "__main__":
    main()
