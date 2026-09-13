"""Experiment 007 (flagship): naturalistic stimulus -> full cascade ->
multi-electrode continuous field potentials — the "fly EEG" demonstration.

This is the end-to-end deliverable of the project's main line: a real
connectome circuit (R -> L1/L2/L3 -> Mi/Tm -> T4/T5, left lobe, ~19k neurons
/ 1.67M synapses) driven by a naturalistic stimulus, with the extracellular
field recorded simultaneously at three virtual electrodes (eye surface,
lamina, medulla). Every electrode trace is generated from the instantaneous
synaptic currents of the spiking network via the quasistatic forward kernels
built in exp001-006.

Stimulus (8 s):
  0.0-2.0 s  darkness (baseline)
  2.0-5.0 s  spatially uniform luminance flicker with 1/f temporal spectrum
  5.0-6.5 s  drifting 1/f spatial texture along the T4/T5 preferred axis
             (-135 deg, from exp006), wavelength content matched to lamina
             sampling
  6.5-8.0 s  same texture, opposite drift direction (direction contrast)

Readouts: per-layer population rates, three electrode potentials, epoch
statistics (band power, DS ratio of rates and of the medulla field).

Run from repository root:
    python experiments/exp007_fly_eeg/run.py
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

from ffbm import data as fdata
from ffbm.forward import StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp005.DT
T_END = 11000.0
SEED = 42

# stimulus epochs (ms)
T_DARK_END = 2000.0
T_FLICKER_END = 5000.0
T_DRIFT_A_END = 6500.0        # broadband texture, drift along +e_ds
T_DRIFT_B_END = 8000.0        # same movie time-reversed
T_BAND_A_END = 9500.0         # band-passed texture (DS-optimal scale), +e_ds
STIM_CONTRAST = 2.0           # luminance = 1 + contrast * noise, clipped >= 0.05
I_LUM = 150.0                 # pA of photocurrent per unit luminance
DRIFT_SPEED = 0.1             # um/ms (= 100 um/s) along the DS axis

EPOCHS = (("dark", 0.0, T_DARK_END),
          ("flicker", T_DARK_END, T_FLICKER_END),
          ("drift+", T_FLICKER_END, T_DRIFT_A_END),
          ("drift-", T_DRIFT_A_END, T_DRIFT_B_END),
          ("band+", T_DRIFT_B_END, T_BAND_A_END),
          ("band-", T_BAND_A_END, T_END))

KIN = exp005.KINETICS["differentiated"]
LAM_UM = 22.0                 # texture length scale (exp006 wavelength)


def one_over_f_noise(rng, n, exponent=1.0):
    """Zero-mean noise with power spectrum ~ 1/f^exponent."""
    x = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    X = np.fft.rfft(x) / f ** (exponent / 2.0)
    y = np.fft.irfft(X, n)
    return y / y.std()


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
    mid_type = circuit["mid_type"]
    t45_type = circuit["t45_type"]

    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)
    mid_index = np.full(int(mid_ids.max()) + 1, -1, dtype=np.int64)
    mid_index[mid_ids] = np.arange(n_mid)
    t45_index = np.full(int(t45_ids.max()) + 1, -1, dtype=np.int64)
    t45_index[t45_ids] = np.arange(n_t45)

    # ---- stimulus geometry: drift axis from exp006 (recompute u/v axes) ----
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]
    r_pos = np.array([pp[b] for b in r_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)
    a1 = np.array([1.0, 0.0, 0.0]) - u_eye * u_eye[0]
    a1 /= np.linalg.norm(a1)
    b1 = np.cross(u_eye, a1)
    mi1_ids = mid_ids[mid_type == "Mi1"]
    mi1_pos = np.array([pp[b] for b in mi1_ids])

    def hex_dir(hex_series):
        h = np.array([hex_series.get(b, np.nan) for b in mi1_ids])
        ok = np.isfinite(h)
        X = np.stack([mi1_pos[ok] @ a1, mi1_pos[ok] @ b1], axis=1)
        coef, *_ = np.linalg.lstsq(X, h[ok], rcond=None)
        d = coef[0] * a1 + coef[1] * b1
        return d / np.linalg.norm(d)

    u_dir = hex_dir(hex1)
    v_dir = hex_dir(hex2)
    e_ds = np.cos(np.radians(-135.0)) * u_dir + np.sin(np.radians(-135.0)) * v_dir
    e_ds /= np.linalg.norm(e_ds)
    x_r = r_pos @ e_ds
    print(f"drift axis e_ds={e_ds.round(3)}, R extent along it "
          f"{x_r.max() - x_r.min():.1f} um")

    # ---- stimulus signals ----
    rng = np.random.default_rng(SEED)
    n_ms = int(T_END)                       # 1 ms grid for stimulus
    flicker = one_over_f_noise(rng, n_ms, 1.0)
    travel = DRIFT_SPEED * T_END            # um of texture travel
    n_tex = 8192
    span = x_r.max() - x_r.min() + travel + 4 * LAM_UM
    texture = one_over_f_noise(rng, n_tex, 1.0)   # per-um samples
    tex_dx = span / n_tex

    # band-passed variant: 1/f only within wavelengths 15-45 um (the DS-
    # optimal scale from exp006), zero outside
    kx = np.fft.rfftfreq(n_tex, d=tex_dx)
    band = np.fft.rfft(rng.standard_normal(n_tex))
    lo, hi = 1.0 / 45.0, 1.0 / 15.0         # cycles/um
    mask = (kx >= lo) & (kx <= hi)
    amp = np.where(mask, 1.0 / np.maximum(kx, lo), 0.0)
    texture_bp = np.fft.irfft(band * amp, n_tex)
    texture_bp /= texture_bp.std()

    def texture_at(x_um):
        idx = np.clip(((x_um - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return texture[idx]

    def texture_bp_at(x_um):
        idx = np.clip(((x_um - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return texture_bp[idx]

    # ---- electrodes: eye surface, lamina, medulla ----
    eye_elec = (r_pos.mean(axis=0)
                + (float(np.max((r_pos - r_pos.mean(axis=0)) @ u_eye)) + 20.0)
                * u_eye)
    lam_elec = l_pos.mean(axis=0) + 30.0 * u_eye
    med_elec = t45_pos.mean(axis=0) + 30.0 * (
        (t45_pos.mean(axis=0) - l_pos.mean(axis=0))
        / np.linalg.norm(t45_pos.mean(axis=0) - l_pos.mean(axis=0)))
    electrodes = np.array([eye_elec, lam_elec, med_elec])
    print("electrodes (um): eye", eye_elec.round(1), "| lamina",
          lam_elec.round(1), "| medulla", med_elec.round(1))

    # ---- synapse groups + per-electrode field kernels ----
    pre = circuit["e_rl"]["body_pre"].to_numpy(np.int64)
    post = circuit["e_rl"]["body_post"].to_numpy(np.int64)
    syn_rl = ExponentialSynapses(
        pre, post, circuit["e_rl"]["weight"].to_numpy(np.float32),
        l_index, dt=DT, gain=exp005.SYN_GAIN_RL, tau_s=exp005.TAU_RL,
        n_post=n_l)
    order = np.lexsort((pre, post))
    field_rl = StaticPairField(np.array([pp[b] for b in pre[order]]),
                               np.array([qq[b] for b in post[order]]),
                               electrodes, sigma=exp005.SIGMA)
    pre = circuit["e_lm"]["body_pre"].to_numpy(np.int64)
    post = circuit["e_lm"]["body_post"].to_numpy(np.int64)
    syn_lm = ExponentialSynapses(
        pre, post, circuit["e_lm"]["weight"].to_numpy(np.float32),
        mid_index, dt=DT, gain=exp005.SYN_GAIN_LM, tau_s=exp005.TAU_LM,
        n_post=n_mid)
    order = np.lexsort((pre, post))
    field_lm = StaticPairField(np.array([pp[b] for b in pre[order]]),
                               np.array([qq[b] for b in post[order]]),
                               electrodes, sigma=exp005.SIGMA)
    syn_mt, field_mt = {}, {}
    for mt in exp005.MID_TYPES:
        pre = circuit["e_mt"][mt]["body_pre"].to_numpy(np.int64)
        post = circuit["e_mt"][mt]["body_post"].to_numpy(np.int64)
        weight = (circuit["e_mt"][mt]["weight"].to_numpy(np.float32)
                  * circuit["e_mt"][mt]["sign"].to_numpy(np.float32))
        syn_mt[mt] = ExponentialSynapses(pre, post, weight, t45_index,
                                         dt=DT, gain=exp005.SYN_GAIN_MT,
                                         tau_s=KIN[mt], n_post=n_t45)
        order = np.lexsort((pre, post))
        field_mt[mt] = StaticPairField(
            np.array([pp[b] for b in pre[order]]),
            np.array([qq[b] for b in post[order]]),
            electrodes, sigma=exp005.SIGMA)
    # photoreceptor dipole: terminal -> extrapolated rhabdomere (23.5 um out)
    rhabd = r_pos + 23.5 * u_eye
    field_photo = StaticPairField(r_pos, rhabd, electrodes,
                                  sigma=exp005.SIGMA)

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

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi = np.zeros((n_field, 3))            # volts, 1 kHz
    rate_r = np.zeros(n_field)
    rate_l = np.zeros(n_field)
    rate_mid = np.zeros(n_field)
    rate_t4 = np.zeros(n_field)
    rate_t5 = np.zeros(n_field)
    i_stim = np.zeros(n_field)              # mean drive (pA), for the record
    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])

    print(f"simulating {T_END / 1000:.0f} s naturalistic stimulus ...")
    for k in range(n_steps):
        t = k * DT
        tms = min(int(t), n_ms - 1)
        lum = np.ones(n_r)                     # luminance, 1 = dark base
        if T_DARK_END <= t < T_FLICKER_END:
            lum = lum + STIM_CONTRAST * flicker[tms]
        elif T_FLICKER_END <= t < T_DRIFT_A_END:
            s_rel = t - T_FLICKER_END
            lum = lum + STIM_CONTRAST * texture_at(x_r - DRIFT_SPEED * s_rel)
        elif T_DRIFT_A_END <= t < T_DRIFT_B_END:
            # exact time-reverse of the drift+ epoch: same luminance
            # distribution traversed backwards -> clean direction contrast
            s_rel = (T_DRIFT_A_END - T_FLICKER_END) - (t - T_DRIFT_A_END)
            lum = lum + STIM_CONTRAST * texture_at(x_r - DRIFT_SPEED * s_rel)
        elif T_DRIFT_B_END <= t < T_BAND_A_END:
            s_rel = t - T_DRIFT_B_END
            lum = lum + STIM_CONTRAST * texture_bp_at(
                x_r - DRIFT_SPEED * s_rel)
        elif t >= T_BAND_A_END:
            s_rel = (T_BAND_A_END - T_DRIFT_B_END) - (t - T_BAND_A_END)
            lum = lum + STIM_CONTRAST * texture_bp_at(
                x_r - DRIFT_SPEED * s_rel)
        lum = np.maximum(lum, 0.05)            # luminance is non-negative
        inc = I_LUM * (lum - 1.0)              # pA increment about dark base
        inc_f = photo.step(inc)
        i_r = exp005.I_R_BASE + inc_f + rng.normal(0, exp005.NOISE_SD["R"],
                                                   n_r)
        sp_r = pop_r.step(i_r)
        i_l = syn_rl.to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["L"], n_l)
        sp_l = pop_l.step(i_l)
        i_mid = syn_lm.to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["MID"], n_mid)
        sp_mid = pop_mid.step(i_mid)
        i_t45 = sum(s.to_neuron_current() for s in syn_mt.values()) \
            + rng.normal(0, exp005.NOISE_SD["T45"], n_t45)
        sp_t45 = pop_t45.step(i_t45)

        syn_rl.step(r_ids[sp_r])
        syn_lm.step(l_ids[sp_l])
        spiked_mid = mid_ids[sp_mid]
        for s in syn_mt.values():
            s.step(spiked_mid)

        if k % 2 == 0:
            j = k // 2
            i_photo = exp005.I_R_BASE + inc_f
            phi[j] = (field_rl.field_timeseries(syn_rl.y[None, :])[0]
                      + field_lm.field_timeseries(syn_lm.y[None, :])[0]
                      + sum(field_mt[mt].field_timeseries(
                          syn_mt[mt].y[None, :])[0]
                          for mt in exp005.MID_TYPES)
                      + field_photo.field_timeseries(i_photo[None, :])[0])
            rate_r[j] = sp_r.sum() * 1000.0 / n_r
            rate_l[j] = sp_l.sum() * 1000.0 / n_l
            rate_mid[j] = sp_mid.sum() * 1000.0 / n_mid
            rate_t4[j] = sp_t45[is_t4].sum() * 1000.0 / is_t4.sum()
            rate_t5[j] = sp_t45[is_t5].sum() * 1000.0 / is_t5.sum()
            i_stim[j] = float(np.mean(inc))

    t_ms = np.arange(n_field) * 1.0

    # ---- epoch statistics ----
    def epoch_stats(sig):
        out = {}
        for name, t0, t1 in EPOCHS:
            m = (t_ms >= t0 + 500) & (t_ms < t1)
            out[name] = {"mean": float(sig[m].mean()),
                         "rms": float(sig[m].std())}
        return out

    summary = {"config": {
        "t_end_ms": T_END, "epochs": EPOCHS,
        "stim_contrast": STIM_CONTRAST, "i_lum_pA": I_LUM,
        "drift_speed_um_ms": DRIFT_SPEED, "lambda_um": LAM_UM,
        "electrodes_um": electrodes.round(2).tolist(),
        "kinetics": KIN,
        "n_R": n_r, "n_L": n_l, "n_mid": n_mid, "n_t45": n_t45,
    }, "rates_hz": {g: epoch_stats(r) for g, r in
                    (("R", rate_r), ("L", rate_l), ("Mi/Tm", rate_mid),
                     ("T4", rate_t4), ("T5", rate_t5))},
        "phi_nV": {f"ch{i}_{name}": {k: {kk: round(vv, 2) for kk, vv
                                          in v.items()}
                                     for k, v in epoch_stats(
                                         phi[:, i] * 1e9).items()}
                   for i, name in enumerate(("eye", "lamina", "medulla"))}}

    # DS contrast in the paired drift epochs (std of rate / phi as response
    # size); the second epoch of each pair is the exact time-reverse of the
    # first, so ratio != 1 is pure direction selectivity
    def seg(t0, t1):
        return (t_ms >= t0 + 500) & (t_ms < t1)

    m_a, m_b = seg(T_FLICKER_END, T_DRIFT_A_END), seg(T_DRIFT_A_END,
                                                      T_DRIFT_B_END)
    m_c, m_d = seg(T_DRIFT_B_END, T_BAND_A_END), seg(T_BAND_A_END, T_END)
    summary["ds_contrast"] = {}
    for tag, (m1, m2) in (("broadband", (m_a, m_b)),
                          ("bandpass", (m_c, m_d))):
        summary["ds_contrast"][tag] = {
            g: round(float(r[m2].std() / (r[m1].std() + 1e-9)), 3)
            for g, r in (("R", rate_r), ("L", rate_l), ("T4", rate_t4),
                         ("T5", rate_t5))}
        summary["ds_contrast"][tag]["phi_medulla"] = round(
            float(phi[m2, 2].std() / phi[m1, 2].std()), 3)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["ds_contrast"], indent=1))
    print("epoch rates (Hz):")
    for g in ("R", "L", "Mi/Tm", "T4", "T5"):
        line = " ".join(f"{ep}:{summary['rates_hz'][g][ep]['mean']:6.1f}"
                        for ep, _, _ in EPOCHS)
        print(f"  {g:>5} {line}")

    # ---- figures ----
    fig, axes = plt.subplots(9, 1, figsize=(11, 13), sharex=True)
    ep_colors = ("orange", "tab:green", "tab:blue", "tab:red", "tab:purple")
    for i_ep, (name, t0, t1) in enumerate(EPOCHS[1:]):
        for ax in axes:
            ax.axvspan(t0, t1, color=ep_colors[i_ep], alpha=0.08)
    axes[0].plot(t_ms, i_stim, color="k", lw=0.6)
    axes[0].set_ylabel("stim (pA)")
    for ax, r, name, c in ((axes[1], rate_r, "R", "tab:purple"),
                           (axes[2], rate_l, "L1-3", "tab:brown"),
                           (axes[3], rate_mid, "Mi/Tm", "tab:olive"),
                           (axes[4], rate_t4, "T4", "tab:blue"),
                           (axes[5], rate_t5, "T5", "tab:cyan")):
        ax.plot(t_ms, r, color=c, lw=0.6)
        ax.set_ylabel(f"{name} (Hz)")
    for i, (name, c) in enumerate((("eye", "tab:red"),
                                   ("lamina", "tab:orange"),
                                   ("medulla", "tab:green"))):
        axes[6 + i].plot(t_ms, phi[:, i] * 1e9, color=c, lw=0.5)
        axes[6 + i].set_ylabel(f"phi {name}\n(nV)")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("exp007: fly EEG — naturalistic stimulus through the "
                 "connectome cascade -> multi-electrode field potentials")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_fly_eeg.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fs = 1000.0
    for i, (name, c) in enumerate((("eye", "tab:red"),
                                   ("lamina", "tab:orange"),
                                   ("medulla", "tab:green"))):
        m = t_ms >= T_DARK_END + 500
        freqs = np.fft.rfftfreq(int(m.sum()))
        spec = np.abs(np.fft.rfft(
            phi[m, i] * 1e9 - phi[m, i].mean())) ** 2
        axes[0].semilogy(freqs[1:], spec[1:] / spec[1:].sum(), color=c,
                         alpha=0.8, label=name)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD (rel.)")
    axes[0].set_title("field spectra (stimulus epochs pooled)")
    axes[0].legend(fontsize=8)

    labels = ["R", "L", "T4", "T5", "phi_med"]
    x_pos = np.arange(5)
    for j, (tag, c) in enumerate((("broadband", "tab:gray"),
                                  ("bandpass", "tab:green"))):
        vals = [summary["ds_contrast"][tag][g] for g in
                ("R", "L", "T4", "T5", "phi_medulla")]
        axes[1].bar(x_pos + (j - 0.5) * 0.38, vals, width=0.36, color=c,
                    label=tag)
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xticks(x_pos, labels)
    axes[1].set_ylabel("response(reverse) / response(forward)")
    axes[1].set_title("direction contrast (time-reversed pairs)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_spectra_ds.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
