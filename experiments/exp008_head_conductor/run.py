"""Experiment 008: sealed-head volume conductor — does layering change the
forward fields?

Upgrades the physics from the homogeneous infinite medium (all of
exp001-007) to a sealed two-layer spherical conductor (ffbm.forward.
SealedHeadPairField): inner sphere = CNS (sigma1), shell = cuticle-ish
(sigma2), zero current through the outer boundary. The kernel converges to
the homogeneous one in the far-boundary limit (validated to 6e-6) and
amplifies |coefficients| ~1.38x in a realistic geometry (image-charge gain).

Protocol: exactly one simulation — exp007's naturalistic cascade (dark / 1/f
flicker / drifting texture forward + reversed) with the SAME spiking and
synaptic currents, while four forward variants are evaluated in parallel on
the same y-streams every ms:

  hom         homogeneous infinite medium (sigma = 0.33)   [exp001-007]
  seal_s1     sealed, sigma2 = sigma1,        r2 = 1.3 r1
  seal_s01    sealed, sigma2 = 0.01 sigma1,   r2 = 1.3 r1
  seal_s01_t  sealed, sigma2 = 0.01 sigma1,   r2 = 1.1 r1

Reported: per-channel epoch amplitudes per variant, amplification ratios,
and trace correlations (topography invariance).

Run from repository root:
    python experiments/exp008_head_conductor/run.py
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

from ffbm.forward import SealedHeadPairField, StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp005.DT
T_END = 8000.0
SEED = 42

T_DARK_END = 2000.0
T_FLICKER_END = 5000.0
STIM_CONTRAST = 2.0
I_LUM = 150.0
DRIFT_SPEED = 0.1
LAM_UM = 22.0

EPOCHS = (("dark", 0.0, T_DARK_END),
          ("flicker", T_DARK_END, T_FLICKER_END),
          ("drift+", T_FLICKER_END, 6500.0),
          ("drift-", 6500.0, T_END))


def one_over_f_noise(rng, n, exponent=1.0):
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

    r_pos = np.array([pp[b] for b in r_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)

    # stimulus: drift along exp006's preferred axis
    from ffbm import data as fdata
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]
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

    u_dir, v_dir = hex_dir(hex1), hex_dir(hex2)
    e_ds = np.cos(np.radians(-135.0)) * u_dir + np.sin(np.radians(-135.0)) * v_dir
    e_ds /= np.linalg.norm(e_ds)
    x_r = r_pos @ e_ds

    rng = np.random.default_rng(SEED)
    n_ms = int(T_END)
    flicker = one_over_f_noise(rng, n_ms, 1.0)
    n_tex = 8192
    span = (x_r.max() - x_r.min()) + DRIFT_SPEED * T_END + 4 * LAM_UM
    texture = one_over_f_noise(rng, n_tex, 1.0)
    tex_dx = span / n_tex

    def texture_at(x_um):
        idx = np.clip(((x_um - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return texture[idx]

    # ---- electrodes and head sphere ----
    eye_elec = (r_pos.mean(axis=0)
                + (float(np.max((r_pos - r_pos.mean(axis=0)) @ u_eye)) + 20.0)
                * u_eye)
    lam_elec = l_pos.mean(axis=0) + 30.0 * u_eye
    med_elec = t45_pos.mean(axis=0) + 30.0 * (
        (t45_pos.mean(axis=0) - l_pos.mean(axis=0))
        / np.linalg.norm(t45_pos.mean(axis=0) - l_pos.mean(axis=0)))
    electrodes = np.array([eye_elec, lam_elec, med_elec])
    center = np.vstack([r_pos, l_pos, mi1_pos, t45_pos]).mean(axis=0)
    r_max = max(np.linalg.norm(np.vstack([r_pos, l_pos, mi1_pos, t45_pos,
                                          electrodes]) - center,
                               axis=1).max(), 1.0)
    r1 = 1.05 * r_max + 10.0
    print(f"head sphere: center={center.round(1)}, r1={r1:.1f} um "
          f"(max source/electrode radius {r_max:.1f})")

    # ---- synapse groups + FOUR parallel forward variants ----
    def build_groups():
        groups = {}
        pre = circuit["e_rl"]["body_pre"].to_numpy(np.int64)
        post = circuit["e_rl"]["body_post"].to_numpy(np.int64)
        groups["RL"] = (pre, post,
                        ExponentialSynapses(
                            pre, post,
                            circuit["e_rl"]["weight"].to_numpy(np.float32),
                            l_index, dt=DT, gain=exp005.SYN_GAIN_RL,
                            tau_s=exp005.TAU_RL, n_post=n_l))
        pre = circuit["e_lm"]["body_pre"].to_numpy(np.int64)
        post = circuit["e_lm"]["body_post"].to_numpy(np.int64)
        groups["LM"] = (pre, post,
                        ExponentialSynapses(
                            pre, post,
                            circuit["e_lm"]["weight"].to_numpy(np.float32),
                            mid_index, dt=DT, gain=exp005.SYN_GAIN_LM,
                            tau_s=exp005.TAU_LM, n_post=n_mid))
        for mt in exp005.MID_TYPES:
            pre = circuit["e_mt"][mt]["body_pre"].to_numpy(np.int64)
            post = circuit["e_mt"][mt]["body_post"].to_numpy(np.int64)
            weight = (circuit["e_mt"][mt]["weight"].to_numpy(np.float32)
                      * circuit["e_mt"][mt]["sign"].to_numpy(np.float32))
            groups[f"MT_{mt}"] = (pre, post, ExponentialSynapses(
                pre, post, weight, t45_index, dt=DT,
                gain=exp005.SYN_GAIN_MT, tau_s=exp005.KINETICS[
                    "differentiated"][mt], n_post=n_t45))
        return groups

    groups = build_groups()
    rhabd = r_pos + 23.5 * u_eye
    photo_pair = (r_pos, rhabd)
    pairs = {k: (np.array([pp[b] for b in g[0][np.lexsort((g[0], g[1]))]]),
                 np.array([qq[b] for b in g[1][np.lexsort((g[0], g[1]))]]))
             for k, g in groups.items()}
    pairs["PHOTO"] = photo_pair

    variants = {
        "hom": lambda pr, po: StaticPairField(pr, po, electrodes,
                                              sigma=exp005.SIGMA),
        "seal_s1": lambda pr, po: SealedHeadPairField(
            pr, po, electrodes, center=center, r1=r1, r2=1.3 * r1,
            sigma1=exp005.SIGMA, sigma2=exp005.SIGMA),
        "seal_s01": lambda pr, po: SealedHeadPairField(
            pr, po, electrodes, center=center, r1=r1, r2=1.3 * r1,
            sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA),
        "seal_s01_t": lambda pr, po: SealedHeadPairField(
            pr, po, electrodes, center=center, r1=r1, r2=1.1 * r1,
            sigma1=exp005.SIGMA, sigma2=0.01 * exp005.SIGMA),
    }
    kernels = {v: {k: f(pr, po) for k, (pr, po) in pairs.items()}
               for v, f in variants.items()}
    print("kernels built for", len(variants), "variants x", len(pairs),
          "groups")

    # ---- simulation (one pass; all variants evaluated on same y) ----
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
    phi = {v: np.zeros((n_field, 3)) for v in variants}
    rate_t4 = np.zeros(n_field)
    rate_t5 = np.zeros(n_field)
    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    group_names = list(groups)

    print(f"simulating {T_END / 1000:.0f} s ...")
    for k in range(n_steps):
        t = k * DT
        tms = min(int(t), n_ms - 1)
        lum = np.ones(n_r)
        if T_DARK_END <= t < T_FLICKER_END:
            lum = lum + STIM_CONTRAST * flicker[tms]
        elif T_FLICKER_END <= t < 6500.0:
            lum = lum + STIM_CONTRAST * texture_at(
                x_r - DRIFT_SPEED * (t - T_FLICKER_END))
        elif t >= 6500.0:
            s_rel = 1500.0 - (t - 6500.0)
            lum = lum + STIM_CONTRAST * texture_at(x_r - DRIFT_SPEED * s_rel)
        lum = np.maximum(lum, 0.05)
        inc = I_LUM * (lum - 1.0)
        inc_f = photo.step(inc)
        i_r = exp005.I_R_BASE + inc_f + rng.normal(0, exp005.NOISE_SD["R"],
                                                   n_r)
        sp_r = pop_r.step(i_r)
        i_l = groups["RL"][2].to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["L"], n_l)
        sp_l = pop_l.step(i_l)
        i_mid = groups["LM"][2].to_neuron_current() + rng.normal(
            0, exp005.NOISE_SD["MID"], n_mid)
        sp_mid = pop_mid.step(i_mid)
        i_t45 = sum(groups[f"MT_{mt}"][2].to_neuron_current()
                    for mt in exp005.MID_TYPES) \
            + rng.normal(0, exp005.NOISE_SD["T45"], n_t45)
        sp_t45 = pop_t45.step(i_t45)

        groups["RL"][2].step(r_ids[sp_r])
        groups["LM"][2].step(l_ids[sp_l])
        spiked_mid = mid_ids[sp_mid]
        for mt in exp005.MID_TYPES:
            groups[f"MT_{mt}"][2].step(spiked_mid)

        if k % 2 == 0:
            j = k // 2
            rate_t4[j] = sp_t45[is_t4].sum() * 1000.0 / is_t4.sum()
            rate_t5[j] = sp_t45[is_t5].sum() * 1000.0 / is_t5.sum()
            for v in variants:
                acc = np.zeros(3)
                for gname in group_names:
                    acc += kernels[v][gname].field_timeseries(
                        groups[gname][2].y[None, :])[0]
                acc += kernels[v]["PHOTO"].field_timeseries(
                    (exp005.I_R_BASE + inc_f)[None, :])[0]
                phi[v][j] = acc

    t_ms = np.arange(n_field) * 1.0

    # ---- analysis ----
    def seg(t0, t1, skip=500):
        return (t_ms >= t0 + skip) & (t_ms < t1)

    summary = {"config": {
        "t_end_ms": T_END, "r1_um": round(r1, 1), "epochs": EPOCHS,
        "center_um": center.round(1).tolist(),
        "n_R": n_r, "n_L": n_l, "n_mid": n_mid, "n_t45": n_t45,
    }, "variants": {}}
    ch_names = ("eye", "lamina", "medulla")
    for v in variants:
        summary["variants"][v] = {
            ch: {name: {"mean_nV": round(float(phi[v][m, i].mean() * 1e9), 1),
                        "rms_nV": round(float(phi[v][m, i].std() * 1e9), 1)}
                 for name, m in ((nm, seg(t0, t1)) for nm, t0, t1 in EPOCHS)}
            for i, ch in enumerate(ch_names)}

    # amplification ratios (std over the whole stimulated period) and trace
    # correlations vs homogeneous
    stim = t_ms >= T_DARK_END + 500
    ratios = {}
    corr = {}
    for v in variants:
        ratios[v] = [round(float(phi[v][stim, i].std()
                                / phi["hom"][stim, i].std()), 3)
                     for i in range(3)]
        corr[v] = [round(float(np.corrcoef(phi[v][stim, i],
                                           phi["hom"][stim, i])[0, 1]), 4)
                   for i in range(3)]
    summary["amplification_vs_hom"] = ratios
    summary["trace_corr_vs_hom"] = corr
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("amplification (std ratio vs homogeneous):",
          json.dumps(ratios, indent=1))
    print("trace correlation vs homogeneous:", json.dumps(corr, indent=1))

    # ---- figures ----
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, (v, color) in zip(axes, (("hom", "k"), ("seal_s1", "tab:blue"),
                                     ("seal_s01", "tab:green"),
                                     ("seal_s01_t", "tab:red"))):
        for i, ch in enumerate(ch_names):
            axes_i = ax
            axes_i.plot(t_ms, phi[v][:, i] * 1e9, color=color, lw=0.5,
                        alpha=0.9 if i == 0 else 0.5,
                        label=f"{ch}" if v == "hom" else None)
        ax.set_ylabel(f"{v}\nphi (nV)")
        if v == "hom":
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("exp008: same spikes, same synapses — four volume "
                 "conductors (eye/lamina/medulla channels)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_variants.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    for j, (v, color) in enumerate((("hom", "k"), ("seal_s1", "tab:blue"),
                                    ("seal_s01", "tab:green"),
                                    ("seal_s01_t", "tab:red"))):
        ax.bar(x + (j - 1.5) * 0.2, ratios[v], width=0.18, color=color,
               label=v)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, ch_names)
    ax.set_ylabel("response std / homogeneous")
    ax.set_title("sealed-head amplification per channel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_amplification.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
