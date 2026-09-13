"""Experiment 010 (thought experiment, computed): scale the fly visual
cascade to a human head and record human-style scalp EEG.

The question: magnify the fly network to human scale, put it in a human-like
3-layer head (brain 80 mm / skull 85 mm sigma=0.013 / scalp 92 mm), record
from a scalp electrode — what do you get?

One flash simulation (exp009's data-sign cascade, 1x flash), with two scalp
kernels evaluated on the same synaptic currents:
  S=1    the fly network at its real size (~200 um) at the brain center
  S=400  all source/sink positions magnified 400x about the network center
         (spans ~72 mm, fills the brain); currents unchanged

Also computed directly: the network's net dipole moment P(t) = sum_e
y_e * (post_e - pre_e), the quantity human-EEG source analysis works with
(typical cortical sources: 10-100 nA*m).

Anchors: EEG amplifier noise floor 1-2 uV; human alpha rhythm 20-100 uV;
visual evoked potentials 1-20 uV.

Run from repository root:
    python experiments/exp010_human_scale/run.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import ScalpPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp009_run",
    ROOT / "experiments" / "exp009_lamina_polarity" / "run.py")
exp009 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exp009)

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp005.DT
T_END = 4500.0
FLASH_ON, FLASH_OFF = 500.0, 3500.0
LEVEL_PA = 350.0
SEED = 42

SCALE = 400.0
# human 3-sphere head (um)
R_BRAIN, R_SKULL, R_SCALP = 8.0e4, 8.5e4, 9.2e4
SIGMA_BRAIN, SIGMA_SKULL, SIGMA_SCALP = 0.33, 0.013, 0.33

EPOCHS = (("dark", 100.0, FLASH_ON),
          ("on", FLASH_ON + 300, FLASH_OFF - 300),
          ("off", FLASH_OFF + 100, FLASH_OFF + 800))


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
    net_center = np.vstack([r_pos, l_pos, t45_pos]).mean(axis=0)
    u_axis = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_axis /= np.linalg.norm(u_axis)
    r_max = np.linalg.norm(np.vstack([r_pos, l_pos, t45_pos]) - net_center,
                           axis=1).max()
    print(f"network radius {r_max:.0f} um; magnified x{SCALE:.0f} = "
          f"{r_max * SCALE / 1000:.1f} mm (brain radius 80 mm)")
    assert r_max * SCALE < R_BRAIN * 0.95

    # scalp electrode above the network along its own axis
    elec = net_center[None, :] + 0.985 * R_SCALP * u_axis[None, :]

    def group_pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        pr = np.array([pp[b] for b in pre[order]])
        po = np.array([qq[b] for b in post[order]])
        return pre, post, pr, po

    groups = {"RL": group_pairs(e_rl), "LM": group_pairs(e_lm)}
    for mt in exp005.MID_TYPES:
        groups[f"MT_{mt}"] = group_pairs(e_mt[mt])
    rhabd = r_pos + 23.5 * u_axis
    groups["PHOTO"] = (r_ids, r_ids, r_pos, rhabd)

    def scalp_kernels(scale):
        out = {}
        for name, (pre, post, pr, po) in groups.items():
            c = net_center
            pr_s = c + (pr - c) * scale
            po_s = c + (po - c) * scale
            out[name] = ScalpPairField(pr_s, po_s, elec, center=c,
                                       r1=R_BRAIN, r2=R_SKULL,
                                       r3=R_SCALP, sigma1=SIGMA_BRAIN,
                                       sigma2=SIGMA_SKULL,
                                       sigma3=SIGMA_SCALP, n_terms=60)
        return out

    print("building scalp kernels (S=1 and S=400) ...")
    ker1 = scalp_kernels(1.0)
    ker4 = scalp_kernels(SCALE)
    # dipole-moment direction vectors per group (pA * um -> nA*m when summed)
    dvecs = {name: (po - pr) for name, (a, b, pr, po) in groups.items()}

    # ---- simulation (exp009 operating point, one flash level) ----
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

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi1 = np.zeros(n_field)      # scalp, S=1   (V)
    phi4 = np.zeros(n_field)      # scalp, S=400 (V)
    dip = np.zeros((n_field, 3))  # net dipole moment (pA*um)
    i_photo_trace = np.zeros((n_field, n_r))

    print(f"simulating flash {LEVEL_PA:.0f} pA ...")
    for k in range(n_steps):
        t = k * DT
        inc = np.full(n_r, LEVEL_PA) if FLASH_ON <= t < FLASH_OFF \
            else np.zeros(n_r)
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
            i_photo_trace[j] = i_photo
            for name in ("RL", "LM", *(f"MT_{mt}" for mt in
                                       exp005.MID_TYPES)):
                y = syn[name].y
                phi1[j] += ker1[name].field(y)[0]
                phi4[j] += ker4[name].field(y)[0]
                dip[j] += y @ dvecs[name]
            y_ph = i_photo
            phi1[j] += ker1["PHOTO"].field(y_ph)[0]
            phi4[j] += ker4["PHOTO"].field(y_ph)[0]
            dip[j] += y_ph @ (rhabd - r_pos)

    t_ms = np.arange(n_field) * 1.0
    dip_nm = dip * 1e-12 * 1e-6 * 1e9        # pA*um -> A*m -> nA*m
    dip_mag = np.linalg.norm(dip_nm, axis=1)

    # ---- analysis ----
    def seg(t0, t1):
        return (t_ms >= t0) & (t_ms < t1)

    res = {"config": {
        "scale": SCALE, "head_um": [R_BRAIN, R_SKULL, R_SCALP],
        "sigmas": [SIGMA_BRAIN, SIGMA_SKULL, SIGMA_SCALP],
        "level_pA": LEVEL_PA, "epochs": EPOCHS,
    }}
    for tag, sig in (("S1", phi1), ("S400", phi4)):
        res[tag] = {name: {"rms_uV": round(float(sig[seg(t0, t1)].std()
                                                 * 1e6), 4)}
                    for name, t0, t1 in EPOCHS}
    res["dipole_nAm"] = {name: {
        "mean": round(float(dip_mag[seg(t0, t1)].mean()), 4),
        "p95": round(float(np.percentile(dip_mag[seg(t0, t1)], 95)), 4)}
        for name, t0, t1 in EPOCHS}

    # replication multipliers (fields add in RMS: sqrt(N) independent, N
    # synchronized), relative to the S=400 "on" epoch
    base = res["S400"]["on"]["rms_uV"]
    res["replication_uV"] = {
        "1 copy (flash, maximal synchrony)": round(base, 3),
        "10^3 independent copies": round(base * np.sqrt(1e3), 3),
        "10^6 independent copies (human cell count)": round(base * 1e3, 3),
        "10^6 synchronized copies (unphysical)": round(base * 1e6, 1),
    }
    res["anchors_uV"] = {"EEG noise floor": 1.5, "human VEP": 10.0,
                         "human alpha": 50.0}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))

    # ---- figure ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_ms, phi1 * 1e6, color="tab:blue")
    axes[0].set_ylabel("S=1 scalp (uV)")
    axes[0].set_title("fly network in a human head, scalp electrode")
    axes[1].plot(t_ms, phi4 * 1e6, color="tab:red")
    axes[1].set_ylabel("S=400 scalp (uV)")
    axes[2].plot(t_ms, dip_mag, color="k")
    axes[2].set_ylabel("net dipole (nA*m)")
    axes[2].set_xlabel("time (ms)")
    for ax in axes:
        ax.axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_human_scale.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
