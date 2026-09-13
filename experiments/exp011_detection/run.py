"""Experiment 011: can ONE 400x-magnified fly network be recorded from the
human scalp with realistic measurement engineering?

exp010: single S=400 copy gives ~0.95 uV ongoing (dark) / ~0.24 uV flash
modulation — below the clinical EEG noise floor (1-2 uV) but NOT below the
physical hard floor (electrode/thermal Johnson noise ~0.03 uV over 50 Hz for
a 1 kOhm interface). Weak-but-nonzero means detectability is a measurement
problem, solvable by the standard tools of human ERP research: trial
averaging (SNR ~ sqrt(k)), bandwidth control, and low-impedance interfaces.

Protocol: one S=400 flash-train simulation (12 cycles of 600 ms dark /
600 ms light after a 2 s lead), scalp electrode as in exp010. The resulting
scalp trace is the "true signal + network self-noise". Synthetic white
measurement noise is added at three levels; responses are extracted by
epoch averaging (k = 1..12 epochs, extrapolated to k = 1000); detectability
d' = |mu_on - mu_base| / sqrt((var_on + var_base)/k) is reported, with
k* = trials needed for d' = 2 (standard detection criterion).

Run from repository root:
    python experiments/exp011_detection/run.py
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
    "exp009_run", ROOT / "experiments" / "exp009_lamina_polarity" / "run.py")
exp009 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exp009)

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp005.DT
SEED = 42
SCALE = 400.0
R_BRAIN, R_SKULL, R_SCALP = 8.0e4, 8.5e4, 9.2e4
LEVEL_PA = 350.0

T_LEAD = 2000.0
N_CYCLES = 12
T_DARK, T_LIGHT = 600.0, 600.0
T_END = T_LEAD + N_CYCLES * (T_DARK + T_LIGHT) + 500.0

NOISE_LEVELS_UV = (1.0, 0.3, 0.1)     # measurement noise, RMS
D_PRIME_TARGET = 2.0


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
    elec = net_center[None, :] + 0.985 * R_SCALP * u_axis[None, :]

    def group_pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        return (pre, post,
                np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]))

    groups = {"RL": group_pairs(e_rl), "LM": group_pairs(e_lm)}
    for mt in exp005.MID_TYPES:
        groups[f"MT_{mt}"] = group_pairs(e_mt[mt])
    rhabd = r_pos + 23.5 * u_axis
    groups["PHOTO"] = (r_ids, r_ids, r_pos, rhabd)

    ker = {}
    for name, (pre, post, pr, po) in groups.items():
        pr_s = net_center + (pr - net_center) * SCALE
        po_s = net_center + (po - net_center) * SCALE
        ker[name] = ScalpPairField(pr_s, po_s, elec, center=net_center,
                                   r1=R_BRAIN, r2=R_SKULL, r3=R_SCALP,
                                   sigma1=0.33, sigma2=0.013, sigma3=0.33,
                                   n_terms=60)
    print("S=400 scalp kernels built")

    # ---- flash-train simulation ----
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
        return int((t - T_LEAD) // (T_DARK + T_LIGHT)) >= 0 and \
            ((t - T_LEAD) % (T_DARK + T_LIGHT)) >= T_DARK

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi = np.zeros(n_field)          # scalp potential (V), 1 kHz
    print(f"simulating {T_END / 1000:.0f} s flash train "
          f"({N_CYCLES} cycles) ...")
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
            acc = 0.0
            for name in ("RL", "LM", *(f"MT_{mt}" for mt in
                                       exp005.MID_TYPES)):
                acc += ker[name].field(syn[name].y)[0]
            acc += ker["PHOTO"].field(i_photo)[0]
            phi[j] = acc

    t_ms = np.arange(n_field) * 1.0
    phi_uv = phi * 1e6

    # ---- epoch extraction and detection analysis ----
    onsets = [T_LEAD + i * (T_DARK + T_LIGHT) + T_DARK
              for i in range(N_CYCLES)]
    ep_t = np.arange(-200, 801)          # ms around light onset
    epochs = np.stack([phi_uv[int(o - 200):int(o + 801)] for o in onsets])
    mean_ep = epochs.mean(axis=0)
    base_m = (ep_t >= -200) & (ep_t < 0)
    on_m = (ep_t >= 300) & (ep_t < 600)

    base_samples = epochs[:, base_m].mean(axis=1)      # per-epoch baseline
    on_samples = epochs[:, on_m].mean(axis=1)          # per-epoch response
    dmu = float(on_samples.mean() - base_samples.mean())
    self_var = float(on_samples.var() + base_samples.var())
    print(f"flash modulation {dmu:+.3f} uV (per-epoch self-noise std "
          f"{np.sqrt(self_var / 2):.3f} uV)")

    rng2 = np.random.default_rng(7)
    table = {}
    for noise_uv in NOISE_LEVELS_UV:
        # per-epoch measurement noise adds var noise^2/n_samples_in_window
        win = int(base_m.sum())
        noise_var = 2.0 * noise_uv ** 2 / win
        total_var = self_var + noise_var
        dp_k = lambda k: abs(dmu) / np.sqrt(total_var / k)
        k_star = max(1, int(np.ceil((D_PRIME_TARGET ** 2) * total_var
                                    / dmu ** 2)))
        table[f"{noise_uv:g}uV"] = {
            "d_prime_1_trial": round(dp_k(1), 3),
            "d_prime_12": round(dp_k(12), 3),
            "d_prime_100": round(dp_k(100), 2),
            "trials_for_d2": k_star,
        }
    summary = {"config": {
        "scale": SCALE, "cycles": N_CYCLES, "t_dark_ms": T_DARK,
        "t_light_ms": T_LIGHT, "level_pA": LEVEL_PA,
        "noise_levels_uV": list(NOISE_LEVELS_UV),
        "d_prime_target": D_PRIME_TARGET,
    }, "signal": {
        "modulation_uV": round(dmu, 4),
        "self_noise_std_uV": round(float(np.sqrt(self_var / 2)), 4),
        "ongoing_dark_rms_uV": round(float(
            phi_uv[(t_ms > 500) & (t_ms < T_LEAD)].std()), 4),
    }, "detection": table,
        "hard_floor_note": "Johnson noise ~0.03 uV (1 kOhm, 50 Hz) is far "
                           "below the 0.2-1.0 uV signal -> no physical "
                           "obstruction; detectability is engineering"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["detection"], indent=1))

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t_ms, phi_uv, color="k", lw=0.5)
    for o in onsets:
        axes[0].axvspan(o, o + T_LIGHT, color="orange", alpha=0.15)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("scalp (uV)")
    axes[0].set_title("S=400 single copy, flash train (with self-noise only)")

    rng3 = np.random.default_rng(3)
    for noise_uv, color in ((1.0, "tab:red"), (0.3, "tab:orange"),
                            (0.1, "tab:green")):
        noisy = epochs + rng3.normal(0, noise_uv, epochs.shape)
        for k_use, alpha in ((1, 0.25), (12, 1.0)):
            avg = noisy[:k_use].mean(axis=0)
            axes[1].plot(ep_t, avg, color=color, alpha=alpha, lw=1,
                         label=f"noise {noise_uv} uV, avg {k_use}")
    axes[1].axvspan(0, T_LIGHT, color="orange", alpha=0.1)
    axes[1].set_xlabel("time from light onset (ms)")
    axes[1].set_ylabel("averaged scalp (uV)")
    axes[1].set_title("epoch-averaged response vs measurement noise")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_detection.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
