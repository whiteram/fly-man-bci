"""Export simulation data for the 3D visualization page (viz/index.html).

Runs one flash-train simulation of the left-lobe cascade (exp009's
data-driven sign structure, exp011's protocol) with three internal fly-head
electrodes (eye / lamina / medulla, sealed-head kernel) and saves everything
the page needs:

  positions of every neuron per layer (centered, um)
  a sample of edges for the flow visualization
  electrode positions + head-sphere geometry
  1 kHz traces: stimulus luminance, per-layer rates, 3 phi channels
  flash cycle markers

Run from repository root:
    python viz/export_data.py        (~9 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import SealedHeadPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp009_run",
    ROOT / "experiments" / "exp009_lamina_polarity" / "run.py")
exp009 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exp009)

OUT = ROOT / "viz" / "data"
DT = exp005.DT
SEED = 42
LEVEL_PA = 350.0

T_LEAD = 2000.0
N_CYCLES = 8
T_DARK, T_LIGHT = 600.0, 600.0
T_END = T_LEAD + N_CYCLES * (T_DARK + T_LIGHT) + 600.0

N_FLOW_EDGES = 1400


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = exp005.build_circuit()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids = circuit["r_ids"]
    l_ids = circuit["l_ids"]
    mid_ids = circuit["mid_ids"]
    t45_ids = circuit["t45_ids"]
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
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)

    eye_elec = (r_pos.mean(axis=0)
                + (float(np.max((r_pos - r_pos.mean(axis=0)) @ u_eye)) + 20.0)
                * u_eye)
    lam_elec = l_pos.mean(axis=0) + 30.0 * u_eye
    med_elec = t45_pos.mean(axis=0) + 30.0 * (
        (t45_pos.mean(axis=0) - l_pos.mean(axis=0))
        / np.linalg.norm(t45_pos.mean(axis=0) - l_pos.mean(axis=0)))
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
    rhabd = r_pos + 23.5 * u_eye
    photo_pair = (r_pos, rhabd)

    ker = {}
    for name, (pr, po) in group_pairs.items():
        ker[name] = SealedHeadPairField(
            pr, po, electrodes, center=center, r1=r1, r2=1.3 * r1,
            sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA)
    ker["PHOTO"] = SealedHeadPairField(
        photo_pair[0], photo_pair[1], electrodes, center=center, r1=r1,
        r2=1.3 * r1, sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA)

    # ---- simulation ----
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
    mid_base = np.full(n_mid, exp009.I_MID_BASE)

    syn = {}
    for name, e_sub, tgt, n_t, gain, tau in (
            ("RL", e_rl, l_index, n_l, exp009.GAIN_RL, exp009.TAU_RL),
            ("LM", e_lm, mid_index, n_mid, exp009.GAIN_LM, exp009.TAU_LM)):
        syn[name] = ExponentialSynapses(
            e_sub["body_pre"].to_numpy(np.int64),
            e_sub["body_post"].to_numpy(np.int64),
            e_sub["weight"].to_numpy(np.float32), tgt, dt=DT, gain=gain,
            tau_s=tau, n_post=n_t)
    for mt in exp005.MID_TYPES:
        e_sub = e_mt[mt]
        syn[f"MT_{mt}"] = ExponentialSynapses(
            e_sub["body_pre"].to_numpy(np.int64),
            e_sub["body_post"].to_numpy(np.int64),
            (e_sub["weight"] * e_sub["sign"]).to_numpy(np.float32),
            t45_index, dt=DT, gain=exp009.GAIN_MT,
            tau_s=exp005.KINETICS["differentiated"][mt], n_post=n_t45)

    def in_light(t):
        if t < T_LEAD:
            return False
        ph = (t - T_LEAD) % (T_DARK + T_LIGHT)
        return ph >= T_DARK

    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi = np.zeros((n_field, 3))
    rate = {k: np.zeros(n_field) for k in
            ("R", "L", "MID", "T4", "T5")}
    stim = np.zeros(n_field)

    print(f"simulating {T_END / 1000:.1f} s ...")
    for k in range(n_steps):
        t = k * DT
        inc = np.full(n_r, LEVEL_PA) if in_light(t) else np.zeros(n_r)
        inc_f = photo.step(inc)
        i_r = exp005.I_R_BASE + inc_f + rng.normal(0, exp005.NOISE_SD["R"],
                                                   n_r)
        sp_r = pop_r.step(i_r)
        i_l = l_base + syn["RL"].to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["L"], n_l)
        sp_l = pop_l.step(i_l)
        i_mid = mid_base + syn["LM"].to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["MID"], n_mid)
        sp_mid = pop_mid.step(i_mid)
        i_t45 = sum(syn[f"MT_{mt}"].to_neuron_current()
                    for mt in exp005.MID_TYPES) \
            + rng.normal(0, exp005.NOISE_SD["T45"], n_t45)
        sp_t45 = pop_t45.step(i_t45)

        syn["RL"].step(r_ids[sp_r])
        syn["LM"].step(l_ids[sp_l])
        spiked_mid = mid_ids[sp_mid]
        for mt in exp005.MID_TYPES:
            syn[f"MT_{mt}"].step(spiked_mid)

        if k % 2 == 0:
            j = k // 2
            i_photo = exp005.I_R_BASE + inc_f
            for i in range(3):
                acc = 0.0
                for name in list(group_pairs) + ["PHOTO"]:
                    y = i_photo if name == "PHOTO" else syn[name].y
                    acc += ker[name].coef[i] @ y
                phi[j, i] = acc * 1e-12
            rate["R"][j] = sp_r.sum() * 1000.0 / n_r
            rate["L"][j] = sp_l.sum() * 1000.0 / n_l
            rate["MID"][j] = sp_mid.sum() * 1000.0 / n_mid
            rate["T4"][j] = sp_t45[is_t4].sum() * 1000.0 / is_t4.sum()
            rate["T5"][j] = sp_t45[is_t5].sum() * 1000.0 / is_t5.sum()
            stim[j] = 1.0 + LEVEL_PA / exp005.I_R_BASE if in_light(t) else 0.0

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
    data = {
        "meta": {
            "t_end_ms": T_END, "cycle_ms": [T_DARK, T_LIGHT],
            "t_lead_ms": T_LEAD, "n_cycles": N_CYCLES,
            "head_r_um": round(r1, 1),
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
        "positions": [pts(r_pos), pts(l_pos), pts(mid_pos),
                      pts(t45_pos[is_t4]), pts(t45_pos[is_t5])],
        "electrodes_pos": np.round(electrodes - center, 1).tolist(),
        "flow_edges": flow,
        "t_ms": np.arange(n_field).tolist(),
        "stim": np.round(stim, 2).tolist(),
        "rates": {k: np.round(v, 1).tolist() for k, v in rate.items()},
        "phi_uV": phi_uv.tolist(),
    }
    path = OUT / "viz_data.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
