"""Working-point calibration (HANDOFF-ranked upgrade #1).

Current exp009 operating point vs fly literature (spontaneous rates):
    Mi/Tm dark 39 Hz   (literature ~5-20)
    T4     dark  0 Hz  (literature ~5-10)
    T5     dark 131 Hz (literature ~5-10)   <- worst offender
The T4/T5 asymmetry is structural (T5's Tm drivers are dark-active, T4's
Mi1/Tm3 are silenced by L1 in the dark) but the magnitude is a working-
point artifact: too much I_MID_BASE drive and too much GAIN_MT per spike.

This script sweeps (I_MID_BASE, GAIN_MT) plus optional T45 noise, runs a
short dark+flash simulation per combo (rates only, no field kernels), and
scores each combo against literature windows. NOTE: with NOISE_SD = 20pA
the LIF f-I curve is extremely steep (5Hz at ~200.3pA, 50Hz at ~250pA),
so rates are hypersensitive near threshold -- the winners land within a
few pA of the target and that fragility is reported, not hidden.

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

from ffbm.simulation import ExponentialSynapses, LIFPopulation

OUT = ROOT / "scripts" / "outputs"
DT = exp005.DT
T_DARK, T_FLASH = 2000.0, 2000.0
T_END = T_DARK + T_FLASH
LEVEL_PA = 350.0
SEED = 42

# literature spontaneous-rate windows (Hz) used for scoring
TARGETS = {"MID": (5.0, 20.0), "T4": (2.0, 12.0), "T5": (2.0, 15.0)}

# (I_MID_BASE, GAIN_MT, T45_NOISE_SD) — first entry = current values;
# round 2 adds per-population T4/T5 background (base, noise) standing in
# for inputs from partners outside the wired 5-layer subcircuit
COMBOS = [
    (220.0, 5.0, 20.0),     # exp009 baseline
    (90.0, 1.4, 20.0),      # round-1 winner (T5 dark 11.6)
    # round 2: T4 background (base pA, noise pA); T5 none
    (90.0, 1.4, 20.0, 175.0, 80.0),
    (90.0, 1.4, 20.0, 185.0, 80.0),
    (90.0, 1.4, 20.0, 195.0, 80.0),
    (90.0, 2.0, 20.0, 185.0, 80.0),
]


def run_combo(circuit, params):
    if len(params) == 3:
        i_mid_base, gain_mt, t45_noise = params
        t4_base, t4_noise = 0.0, None
    else:
        i_mid_base, gain_mt, t45_noise, t4_base, t4_noise = params
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

    syn = {
        "RL": ExponentialSynapses(
            e_rl["body_pre"].to_numpy(np.int64),
            e_rl["body_post"].to_numpy(np.int64),
            e_rl["weight"].to_numpy(np.float32), l_index, dt=DT,
            gain=exp009.GAIN_RL, tau_s=exp009.TAU_RL, n_post=n_l),
        "LM": ExponentialSynapses(
            e_lm["body_pre"].to_numpy(np.int64),
            e_lm["body_post"].to_numpy(np.int64),
            e_lm["weight"].to_numpy(np.float32), mid_index, dt=DT,
            gain=exp009.GAIN_LM, tau_s=exp009.TAU_LM, n_post=n_mid),
    }
    for mt in exp005.MID_TYPES:
        e_sub = e_mt[mt]
        syn[f"MT_{mt}"] = ExponentialSynapses(
            e_sub["body_pre"].to_numpy(np.int64),
            e_sub["body_post"].to_numpy(np.int64),
            (e_sub["weight"] * e_sub["sign"]).to_numpy(np.float32),
            t45_index, dt=DT, gain=gain_mt,
            tau_s=exp005.KINETICS["differentiated"][mt], n_post=n_t45)

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
    l_base = np.full(n_l, exp009.I_L_BASE)
    mid_base = np.full(n_mid, i_mid_base)

    n_steps = int(T_END / DT)
    n_out = n_steps // 2
    acc = {k: 0.0 for k in ("R", "L", "MID", "T4", "T5")}
    cnt = {k: np.zeros(n_out) for k in acc}
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
        if t4_noise is None:
            noise_t45 = rng.normal(0, t45_noise, n_t45)
        else:
            noise_t45 = rng.normal(0, t45_noise, n_t45)
            noise_t45[is_t4] = rng.normal(0, t4_noise, is_t4.sum())
        base_t45 = np.zeros(n_t45)
        base_t45[is_t4] = t4_base
        sp_t45 = pop_t45.step(
            base_t45 + sum(syn[f"MT_{mt}"].to_neuron_current()
                           for mt in exp005.MID_TYPES) + noise_t45)
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
    """Sum of squared excess outside the literature windows."""
    s = 0.0
    for key, (lo, hi) in TARGETS.items():
        for win, weight in (("dark", 1.0), ("flash", 0.35)):
            v = r[win][key]
            s += weight * (max(0.0, lo - v) + max(0.0, v - hi)) ** 2
    # evoked response must survive: T4 flash response present
    if r["flash"]["T4"] < 5.0:
        s += 400.0
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("building circuit ...")
    circuit = exp005.build_circuit()
    rows = []
    for params in COMBOS:
        extra = f" T4base={params[3]:.0f} T4noise={params[4]:.0f}" \
            if len(params) > 3 else ""
        print(f"combo I_MID={params[0]:.0f} GAIN_MT={params[1]:.2f} "
              f"NOISE45={params[2]:.0f}{extra} ...", flush=True)
        r = run_combo(circuit, params)
        r["params"] = {"i_mid_base": params[0], "gain_mt": params[1],
                       "t45_noise": params[2],
                       **({"t4_base": params[3], "t4_noise": params[4]}
                          if len(params) > 3 else {})}
        r["score"] = round(score(r), 2)
        rows.append(r)
        print(f"   dark {r['dark']}  flash {r['flash']}  score {r['score']}")
    best = min(rows, key=lambda r: r["score"])
    print("\nBEST:", json.dumps(best, indent=1))
    (OUT / "working_point_calibration.json").write_text(
        json.dumps({"combos": rows, "best": best}, indent=1))
    print(f"written to {OUT / 'working_point_calibration.json'}")


if __name__ == "__main__":
    main()
