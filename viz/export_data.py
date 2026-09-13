"""Export simulation data for the 3D visualization page (viz/index.html).

Runs one naturalistic simulation of the left-lobe cascade (exp009's
data-driven sign structure) with three internal fly-head electrodes
(eye / lamina / medulla, sealed-head kernel) plus a 17-electrode scalp
array in the x400 human-head thought experiment (exp010/011 geometry,
3-sphere ScalpPairField kernel), and saves everything the page needs:

  positions of every neuron per layer (centered, um)
  a sample of edges for the flow visualization
  electrode positions + head-sphere geometry
  1 kHz traces: stimulus luminance, per-layer rates, 3 phi channels,
  17 scalp channels (channel 0 on the eye axis, sorted by angle)

Run from repository root:
    python viz/export_data.py        (~20 min, scalp array dominates)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import FourSpherePairField, SealedHeadPairField
from ffbm.simulation import (ColoredCurrentNoise, ExponentialSynapses,
                             LIFPopulation)

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

# naturalistic protocol (exp007-style): continuous waveforms, no square
# flashes -- the resulting EEG traces are wavy, not blocky
T_EPOCHS = [
    ("dark", 0.0, 1500.0),            # 黑暗基线
    ("flicker", 1500.0, 4500.0),      # 空间均匀 1/f 闪烁
    ("drift+", 4500.0, 6000.0),       # 1/f 纹理沿偏好方向漂移
    ("drift-", 6000.0, 7500.0),       # 同一影片时间反演
    ("band+", 7500.0, 9000.0),        # DS 尺度带通纹理漂移
    ("band-", 9000.0, 10500.0),       # 其反演
]
T_END = T_EPOCHS[-1][2]
STIM_CONTRAST = 2.0
I_LUM = 150.0
DRIFT_SPEED = 0.1                     # um/ms
LAM_UM = 22.0
EPOCH_COLORS = {"dark": "#33465a", "flicker": "#f7c948", "drift+": "#34d399",
                "drift-": "#22d3ee", "band+": "#f472b6", "band-": "#c084fc"}

N_FLOW_EDGES = 1400

# working-point calibration (scripts/calibrate_working_point.py round 3,
# under OU background noise + conductance MT synapses + axon/syn delays):
# spontaneous rates matched to fly literature. The T4 background base
# stands in for inputs from partners outside the wired subcircuit; the OU
# noise (tau 8 ms, ~4 mV membrane fluctuation) makes the f-I curve graded
# instead of knife-edge. MT synapses are conductance-based (driving force
# depends on v_post; E_rev from dataset sign, exc 0 / inh -80 mV).
V_AXON_UM_PER_MS = 300.0     # 0.3 m/s small fly axons
SYN_DELAY_MS = 1.0
CAL = {"I_MID_BASE": 90.0, "I_T4_BASE": 175.0, "OU_SIGMA_T45": 60.0,
       "G_UNIT_MT": 0.02, "E_REV_INH": -80.0}

# background human EEG added to the scalp channels (thought experiment:
# the fly network's signal must be read out of a realistic recording):
# posterior-dominant 10 Hz alpha (eyes closed), 1/f aperiodic activity
# and sensor white noise
BG = {"ALPHA_HZ": 10.0, "ALPHA_AMP_UV": 30.0, "APERIODIC_UV": 3.0,
      "SENSOR_UV": 1.5}


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

    # thought-experiment channels (exp010/011): the same network magnified
    # 400x inside a human 4-layer head (brain/CSF/skull/scalp), electrode
    # ARRAY on the scalp (channel 0 on the eye axis + a quasi-uniform
    # Fibonacci cover, sorted by angle from the eye axis). The network is
    # shifted occipitally: the T4/T5 (output/"cortex") end of the cascade
    # is brought near the inner skull wall, like the real visual cortex at
    # the occipital pole; the retina end dips toward the head center.
    SCALE = 400.0
    R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
    SIGMAS = (0.33, 1.79, 0.013, 0.33)
    N_SCALP_ELEC = 17

    scaled_pairs = {name: (center + (pr - center) * SCALE,
                           center + (po - center) * SCALE)
                    for name, (pr, po) in {**group_pairs,
                                           "PHOTO": photo_pair}.items()}
    all_pts = np.vstack([p for pr, po in scaled_pairs.values()
                         for p in (pr, po)])
    proj = all_pts @ u_eye
    e_cortex = float(-proj.min())                    # T4/T5-side extent
    r_safe = 0.98 * R_BRAIN
    # per-point quadratic |s - d*u_eye| <= r_safe -> d <= root+
    root = proj + np.sqrt(proj ** 2 - (np.linalg.norm(all_pts - center,
                                                      axis=1) ** 2)
                          + r_safe ** 2)
    d_shift = float(min(0.90 * R_BRAIN - e_cortex, root.min()))
    d_shift = max(d_shift, 0.0)
    shift_vec = -u_eye * d_shift
    margin = r_safe - np.linalg.norm(all_pts + shift_vec - center, axis=1).max()
    print(f"occipital shift: {d_shift / 1000:.1f} mm toward the T4/T5 pole "
          f"(brain-margin {margin:.0f} um)")

    def scalp_electrode_dirs():
        n_ext = N_SCALP_ELEC          # generate N, drop the one nearest
        i = np.arange(n_ext) + 0.5    # the anchor, then prepend the anchor
        az = np.pi * (1.0 + 5.0 ** 0.5) * i   # -> N total, quasi-uniform
        z = 1.0 - 2.0 * i / n_ext
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        fib = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
        fib = np.delete(fib, int(np.argmax(fib @ u_eye)), axis=0)
        dirs = np.vstack([u_eye[None, :], fib])
        ang = np.degrees(np.arccos(np.clip(dirs @ u_eye, -1.0, 1.0)))
        return dirs[np.argsort(ang, kind="stable")]

    scalp_dirs = scalp_electrode_dirs()
    scalp_ang = np.degrees(np.arccos(np.clip(scalp_dirs @ u_eye, -1, 1)))
    coef_scalp = {name: [] for name in
                  list(group_pairs) + ["PHOTO"]}
    import time as _time
    _t0 = _time.time()
    for k, d in enumerate(scalp_dirs):
        elec_k = center + 0.985 * R_SCALP * d
        for name, (pr_s, po_s) in scaled_pairs.items():
            coef_scalp[name].append(FourSpherePairField(
                pr_s + shift_vec, po_s + shift_vec, elec_k[None, :],
                center=center, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL,
                r4=R_SCALP, sigma1=SIGMAS[0], sigma2=SIGMAS[1],
                sigma3=SIGMAS[2], sigma4=SIGMAS[3]).coef[0])
        print(f"scalp kernel {k + 1}/{N_SCALP_ELEC} "
              f"({_time.time() - _t0:.0f} s)", flush=True)
    coef_scalp = {k: np.vstack(v) for k, v in coef_scalp.items()}
    print(f"scalp kernels built (S=400, 4-sphere, {N_SCALP_ELEC} electrodes, "
          f"{_time.time() - _t0:.0f} s)")

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
    mid_base = np.full(n_mid, CAL["I_MID_BASE"])
    noise_t45 = ColoredCurrentNoise(n_t45, DT, rng, tau_n=8.0,
                                    sigma=CAL["OU_SIGMA_T45"])

    def edge_delays(pre_ids, post_ids):
        d = np.array([np.linalg.norm(pp[a] - qq[b])
                      for a, b in zip(pre_ids, post_ids)])
        return SYN_DELAY_MS + d / V_AXON_UM_PER_MS

    syn = {}
    for name, e_sub, tgt, n_t, gain, tau in (
            ("RL", e_rl, l_index, n_l, exp009.GAIN_RL, exp009.TAU_RL),
            ("LM", e_lm, mid_index, n_mid, exp009.GAIN_LM, exp009.TAU_LM)):
        syn[name] = ExponentialSynapses(
            e_sub["body_pre"].to_numpy(np.int64),
            e_sub["body_post"].to_numpy(np.int64),
            e_sub["weight"].to_numpy(np.float32), tgt, dt=DT, gain=gain,
            tau_s=tau, n_post=n_t,
            delay_ms=edge_delays(e_sub["body_pre"].to_numpy(np.int64),
                                 e_sub["body_post"].to_numpy(np.int64)))
    for mt in exp005.MID_TYPES:
        e_sub = e_mt[mt]
        syn[f"MT_{mt}"] = ExponentialSynapses(
            e_sub["body_pre"].to_numpy(np.int64),
            e_sub["body_post"].to_numpy(np.int64),
            e_sub["weight"].to_numpy(np.float32),
            t45_index, dt=DT, gain=1.0,          # unitless gating
            tau_s=exp005.KINETICS["differentiated"][mt], n_post=n_t45,
            sign=e_sub["sign"].to_numpy(np.float32),
            delay_ms=edge_delays(e_sub["body_pre"].to_numpy(np.int64),
                                 e_sub["body_post"].to_numpy(np.int64)),
            conductance=True, g_unit=CAL["G_UNIT_MT"], e_rev_exc=0.0,
            e_rev_inh=CAL["E_REV_INH"])

    # naturalistic stimulus: 1/f flicker, drifting 1/f texture along the
    # T4/T5 preferred axis (hex regression), DS-scale bandpassed texture;
    # reverse epochs replay the forward movie exactly backwards
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
    e_ds = (np.cos(np.radians(-135.0)) * u_dir
            + np.sin(np.radians(-135.0)) * v_dir)
    e_ds /= np.linalg.norm(e_ds)
    x_r = r_pos @ e_ds

    rng_s = np.random.default_rng(SEED)

    def one_over_f(n, exponent=1.0):
        x = rng_s.standard_normal(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        y = np.fft.irfft(np.fft.rfft(x) / f ** (exponent / 2.0), n)
        return y / y.std()

    n_ms = int(T_END)
    flicker = one_over_f(n_ms)
    n_tex = 8192
    span = (x_r.max() - x_r.min()) + DRIFT_SPEED * T_END + 4 * LAM_UM
    texture = one_over_f(n_tex)
    kx = np.fft.rfftfreq(n_tex, d=span / n_tex)
    band = np.fft.rfft(rng_s.standard_normal(n_tex))
    lo, hi = 1.0 / 45.0, 1.0 / 15.0
    amp = np.where((kx >= lo) & (kx <= hi), 1.0 / np.maximum(kx, lo), 0.0)
    texture_bp = np.fft.irfft(band * amp, n_tex)
    texture_bp /= texture_bp.std()
    tex_dx = span / n_tex

    def tex_at(x, table):
        idx = np.clip(((x - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return table[idx]

    def epoch_of(t):
        for name, t0, t1 in T_EPOCHS:
            if t0 <= t < t1:
                return name
        return "dark"

    def luminance(t):
        """Per-R luminance (1 = dark base); non-negative by construction."""
        ep = epoch_of(t)
        one = np.ones(n_r)
        if ep == "flicker":
            return np.maximum(one + STIM_CONTRAST * flicker[min(int(t),
                                                                n_ms - 1)],
                              0.05)
        if ep in ("drift+", "drift-"):
            t0 = 4500.0
            s = t - t0 if ep == "drift+" else 1500.0 - (t - 6000.0)
            return np.maximum(one + STIM_CONTRAST * tex_at(
                x_r - DRIFT_SPEED * s, texture), 0.05)
        if ep in ("band+", "band-"):
            t0 = 7500.0
            s = t - t0 if ep == "band+" else 1500.0 - (t - 9000.0)
            return np.maximum(one + STIM_CONTRAST * tex_at(
                x_r - DRIFT_SPEED * s, texture_bp), 0.05)
        return one

    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    t45_base = np.zeros(n_t45)
    t45_base[is_t4] = CAL["I_T4_BASE"]
    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi = np.zeros((n_field, 3))
    phi_scalp = np.zeros((n_field, N_SCALP_ELEC))
    rate = {k: np.zeros(n_field) for k in
            ("R", "L", "MID", "T4", "T5")}
    stim = np.zeros(n_field)

    print(f"simulating {T_END / 1000:.1f} s ...")
    for k in range(n_steps):
        t = k * DT
        lum = luminance(t)
        inc = I_LUM * (lum - 1.0)
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
            i_photo = exp005.I_R_BASE + inc_f

            def y_of(name):
                if name == "PHOTO":
                    return i_photo
                if name.startswith("MT_"):
                    return syn[name].edge_currents(pop_t45.v)
                return syn[name].y

            for i in range(3):
                acc = 0.0
                for name in list(group_pairs) + ["PHOTO"]:
                    acc += ker[name].coef[i] @ y_of(name)
                phi[j, i] = acc * 1e-12
            acc_s = np.zeros(N_SCALP_ELEC)
            for name in list(group_pairs) + ["PHOTO"]:
                acc_s += coef_scalp[name] @ y_of(name)
            phi_scalp[j] = acc_s * 1e-12
            rate["R"][j] = sp_r.sum() * 1000.0 / n_r
            rate["L"][j] = sp_l.sum() * 1000.0 / n_l
            rate["MID"][j] = sp_mid.sum() * 1000.0 / n_mid
            rate["T4"][j] = sp_t45[is_t4].sum() * 1000.0 / is_t4.sum()
            rate["T5"][j] = sp_t45[is_t5].sum() * 1000.0 / is_t5.sum()
            stim[j] = float(np.mean(lum))

    # ---- background human EEG added to the scalp channels ----
    # eyes-closed alpha (10 Hz, waxing envelope, occipital-dominant like
    # the real posterior rhythm) + 1/f aperiodic activity + sensor noise
    rng_bg = np.random.default_rng(2026)
    bg = np.zeros((N_SCALP_ELEC, n_field))
    tt = np.arange(n_field, dtype=np.float64)
    cos_occ = scalp_dirs @ (-u_eye)              # cos(angle to occipital)
    w_occ = np.clip(cos_occ, 0.0, None) ** 2     # posterior weight

    def bg_one_over_f(n, exponent=1.0):
        x = rng_bg.standard_normal(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        y = np.fft.irfft(np.fft.rfft(x) / f ** (exponent / 2.0), n)
        return y / y.std()

    for e in range(N_SCALP_ELEC):
        env = 0.55 + 0.45 * np.sin(
            2 * np.pi * tt / 700.0 + 0.7 * e)   # ~1.4 s waxing cycles
        alpha = (BG["ALPHA_AMP_UV"] * w_occ[e] * env
                 * np.sin(2 * np.pi * BG["ALPHA_HZ"] * tt / 1000.0
                          + 0.15 * e))
        aper = BG["APERIODIC_UV"] * bg_one_over_f(n_field)
        sens = rng_bg.normal(0, BG["SENSOR_UV"], n_field)
        bg[e] = alpha + aper + sens
    # SNR of the fly signal against this background (best electrode)
    flicker_win = slice(1500, 4500)
    sig_std = phi_scalp[1500:4500, :].std(axis=0) * 1e6   # (n_elec,)
    bg_std = bg[:, flicker_win].std(axis=1)               # (n_elec,)
    best = int(np.argmax(sig_std))
    snr = {"best_elec_deg": round(float(scalp_ang[best]), 1),
           "sig_uv": round(float(sig_std[best]), 2),
           "bg_uv": round(float(bg_std[best]), 2),
           "amp_ratio": round(float(sig_std[best] / bg_std[best]), 3),
           "k_for_dprime2": int(np.ceil(
               (2.0 * bg_std[best] / sig_std[best]) ** 2))}
    print(f"background EEG: fly signal {snr['sig_uv']:.2f} uV vs bg "
          f"{snr['bg_uv']:.2f} uV at {snr['best_elec_deg']:.0f} deg -> "
          f"d'=2 needs ~{snr['k_for_dprime2']} trials")

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
            "t_end_ms": T_END,
            "epochs": [
                {"name": name, "t0": t0, "t1": t1,
                 "color": EPOCH_COLORS[name]}
                for name, t0, t1 in T_EPOCHS],
            "epoch_labels": {"dark": "黑暗", "flicker": "1/f 闪烁",
                             "drift+": "纹理漂移 →", "drift-": "纹理漂移 ←",
                             "band+": "带通纹理 →", "band-": "带通纹理 ←"},
            "head_r_um": round(r1, 1),
            "scalp": {"model": "4sphere", "place": "occipital",
                      "scale": 400,
                      "radii_um": [7.8e4, 8.0e4, 8.5e4, 9.2e4],
                      "sigmas": list(SIGMAS),
                      "shift_um": [round(float(x), 1) for x in shift_vec],
                      "n_elec": N_SCALP_ELEC,
                      "elec_dist_um": 0.985 * 9.2e4,
                      "elec_dir": [round(float(x), 4) for x in u_eye],
                      "elec_dirs": [[round(float(x), 4) for x in d]
                                    for d in scalp_dirs],
                      "elec_deg": [round(float(a), 1)
                                   for a in scalp_ang]},
            "calibration": {**CAL, "source":
                            "scripts/calibrate_working_point.py round 3 "
                            "(OU bg + conductance MT + delays)"},
            "bg_eeg": {**BG, "snr": snr},
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
        "phi_scalp_all_uV": np.round(phi_scalp.T * 1e6, 3).tolist(),
        "phi_scalp_bg_uV": np.round(bg, 2).tolist(),
    }
    path = OUT / "viz_data.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
