"""Experiment 006: 2D direction sweep — does direction selectivity emerge
along the structurally predicted axes?

exp005 ruled out DS along the u-axis (offset vectors are mostly off-axis) and
produced the 2D input-offset blueprint. Here gratings drift along all six hex
grid axes (0..300 deg, both drift signs) at a wavelength matched to the
measured offsets (6 hex units instead of exp005's ~15), with differentiated
input kinetics. Per T4/T5 subtype we measure the DS vector

    D = sum_axis DSI(axis) * e_axis

and compare it with the linearized Hassenstein-Reichardt prediction from the
structure blueprint:

    P = sum_t  w_t * sign_t * (tau_t - tau_ref) * offset_t

with per-type synapse fraction w_t, dataset neurotransmitter sign, kinetic
delay contrast, and the 2D offset vectors (hex units) of structure.py.

Circuit, layers, gains and kinetics are exactly exp005's (left lobe).

Run from repository root:
    python experiments/exp006_ds_directions/run.py
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

# ----------------------------- configuration -----------------------------
FREQ_HZ = 5.0
N_HEX_UNITS = 6.0          # grating wavelength, matched to 1-2 unit offsets
AXES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
DRIFTS = (+1, -1)
KIN = exp005.KINETICS["differentiated"]
INPUT_T4 = ("Mi1", "Tm3", "Mi4", "Mi9")
INPUT_T5 = ("Tm1", "Tm2", "Tm4", "Tm9")
STRUCT_JSON = ROOT / "experiments" / "exp005_medulla_ds" / "outputs" \
    / "structure_offsets.json"
TAU_REF = float(np.mean(list(KIN.values())))   # kinetic delay reference


def lockin(signal, t_ms, freq_hz, t0, t1):
    m = (t_ms >= t0) & (t_ms < t1)
    z = signal[m] * np.exp(-2j * np.pi * freq_hz * t_ms[m] * 1e-3)
    return z.mean() * 2.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = exp005.build_circuit()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    ann = fdata.load_annotations()
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]

    r_ids = circuit["r_ids"]
    l_ids = circuit["l_ids"]
    mid_ids = circuit["mid_ids"]
    t45_ids = circuit["t45_ids"]
    n_r, n_l, n_mid, n_t45 = (len(r_ids), len(l_ids), len(mid_ids),
                              len(t45_ids))
    mid_type = circuit["mid_type"]
    t45_type = circuit["t45_type"]

    # ---- in-plane axes from BOTH hex regressions + hex-unit scale ----
    r_pos = np.array([pp[b] for b in r_ids])
    u_eye = r_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)
    a1 = np.array([1.0, 0.0, 0.0]) - u_eye * u_eye[0]
    a1 /= np.linalg.norm(a1)
    b1 = np.cross(u_eye, a1)
    mi1_ids = mid_ids[mid_type == "Mi1"]
    mi1_pos = np.array([pp[b] for b in mi1_ids])

    def hex_axis(hex_series):
        h = np.array([hex_series.get(b, np.nan) for b in mi1_ids])
        ok = np.isfinite(h)
        X = np.stack([mi1_pos[ok] @ a1, mi1_pos[ok] @ b1], axis=1)
        coef, *_ = np.linalg.lstsq(X, h[ok], rcond=None)
        d = coef[0] * a1 + coef[1] * b1
        scale = np.linalg.norm(d)          # hex units per um
        r2 = 1.0 - np.sum((h[ok] - X @ coef) ** 2) / np.sum(
            (h[ok] - h[ok].mean()) ** 2)
        return d / scale, scale, r2

    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    u_dir, scale_u, r2u = hex_axis(hex1)
    v_dir, scale_v, r2v = hex_axis(hex2)
    um_per_hex = 1.0 / scale_u
    lam_um = N_HEX_UNITS * um_per_hex
    print(f"hex axes: u R2={r2u:.3f} ({um_per_hex:.1f} um/hex), "
          f"v R2={r2v:.3f}; lambda = {lam_um:.0f} um")

    # ---- synapse groups + field kernels (as exp005) ----
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)
    mid_index = np.full(int(mid_ids.max()) + 1, -1, dtype=np.int64)
    mid_index[mid_ids] = np.arange(n_mid)
    t45_index = np.full(int(t45_ids.max()) + 1, -1, dtype=np.int64)
    t45_index[t45_ids] = np.arange(n_t45)
    type_idx = {**{f"mid_{mt}": np.where(mid_type == mt)[0]
                   for mt in exp005.MID_TYPES},
                **{st: np.where(t45_type == st)[0]
                   for st in exp005.T45_SUBTYPES}}

    elec = circuit["elec"][None, :]
    t_ms = np.arange(int(exp005.T_END / exp005.DT) // 2) * 2 * exp005.DT

    pre = circuit["e_rl"]["body_pre"].to_numpy(np.int64)
    post = circuit["e_rl"]["body_post"].to_numpy(np.int64)
    syn_rl = ExponentialSynapses(
        pre, post, circuit["e_rl"]["weight"].to_numpy(np.float32),
        l_index, dt=exp005.DT, gain=exp005.SYN_GAIN_RL,
        tau_s=exp005.TAU_RL, n_post=n_l)
    order = np.lexsort((pre, post))
    field_rl = StaticPairField(np.array([pp[b] for b in pre[order]]),
                               np.array([qq[b] for b in post[order]]),
                               elec, sigma=exp005.SIGMA)
    pre = circuit["e_lm"]["body_pre"].to_numpy(np.int64)
    post = circuit["e_lm"]["body_post"].to_numpy(np.int64)
    syn_lm = ExponentialSynapses(
        pre, post, circuit["e_lm"]["weight"].to_numpy(np.float32),
        mid_index, dt=exp005.DT, gain=exp005.SYN_GAIN_LM,
        tau_s=exp005.TAU_LM, n_post=n_mid)
    order = np.lexsort((pre, post))
    field_lm = StaticPairField(np.array([pp[b] for b in pre[order]]),
                               np.array([qq[b] for b in post[order]]),
                               elec, sigma=exp005.SIGMA)
    syn_mt, field_mt = {}, {}
    for mt in exp005.MID_TYPES:
        pre = circuit["e_mt"][mt]["body_pre"].to_numpy(np.int64)
        post = circuit["e_mt"][mt]["body_post"].to_numpy(np.int64)
        weight = (circuit["e_mt"][mt]["weight"].to_numpy(np.float32)
                  * circuit["e_mt"][mt]["sign"].to_numpy(np.float32))
        syn_mt[mt] = ExponentialSynapses(pre, post, weight, t45_index,
                                         dt=exp005.DT,
                                         gain=exp005.SYN_GAIN_MT,
                                         tau_s=KIN[mt], n_post=n_t45)
        order = np.lexsort((pre, post))
        field_mt[mt] = StaticPairField(
            np.array([pp[b] for b in pre[order]]),
            np.array([qq[b] for b in post[order]]),
            elec, sigma=exp005.SIGMA)

    # ---- direction sweep ----
    def phase_for(theta_deg):
        e = np.cos(np.radians(theta_deg)) * u_dir \
            + np.sin(np.radians(theta_deg)) * v_dir
        x = r_pos @ e
        return 2.0 * np.pi * (x - x.mean()) / lam_um

    amps = {}     # (theta_deg, drift, group) -> modulation amplitude
    for theta in AXES_DEG:
        phase0 = phase_for(theta)
        for d in DRIFTS:
            tag = f"{theta:.0f}deg/dir{d:+d}"
            print(f"simulating {tag} ...")
            res = exp005.simulate(
                FREQ_HZ, d, phase0, r_ids, l_ids, mid_ids, syn_rl, syn_lm,
                syn_mt, type_idx, field_rl, field_lm, field_mt)
            for g in list(exp005.RATE_GROUPS) + ["phi"]:
                sig = res["phi_nV"] if g == "phi" else res["rates"][g]
                amps[(theta, d, g)] = float(
                    abs(lockin(sig, t_ms, FREQ_HZ, exp005.MEAS_ON,
                               exp005.T_END)))
            print(f"  T5a {amps[(theta, d, 'T5a')]:.2f} Hz | "
                  f"T4c {amps[(theta, d, 'T4c')]:.2f} Hz")

    # ---- DS vectors per group ----
    axes_dirs = {th: np.cos(np.radians(th)) * u_dir
                 + np.sin(np.radians(th)) * v_dir for th in AXES_DEG}
    ds_vectors = {}
    for g in exp005.T45_SUBTYPES:
        vec = np.zeros(3)
        dsi_by_axis = {}
        for th in AXES_DEG:
            ap, am = amps[(th, +1, g)], amps[(th, -1, g)]
            dsi = (ap - am) / (ap + am + 1e-12)
            dsi_by_axis[th] = dsi
            vec = vec + dsi * axes_dirs[th]
        ds_vectors[g] = {"vector_um": vec.tolist(),
                         "mag": float(np.linalg.norm(vec)),
                         "dsi_by_axis": {f"{k:.0f}": round(v, 4)
                                         for k, v in dsi_by_axis.items()},
                         "amp_mean": float(np.mean(
                             [amps[(th, d, g)] for th in AXES_DEG
                              for d in DRIFTS]))}
    phi_vec = np.zeros(3)
    for th in AXES_DEG:
        ap, am = amps[(th, +1, "phi")], amps[(th, -1, "phi")]
        phi_vec = phi_vec + (ap - am) / (ap + am + 1e-12) * axes_dirs[th]

    # ---- structural HR prediction from exp005 blueprint ----
    struct = json.loads(STRUCT_JSON.read_text())
    o2d = struct["offsets_2d_lobeA"]
    pred_vectors = {}
    edges_mt_all = circuit["e_mt"]
    tmap = dict(zip(t45_ids.tolist(), t45_type.tolist()))
    for st in exp005.T45_SUBTYPES:
        pg = "T4" if st.startswith("T4") else "T5"
        pv = np.zeros(2)
        for mt in (INPUT_T4 if pg == "T4" else INPUT_T5):
            sub = edges_mt_all[mt]
            wsub = float(sub.loc[[tmap.get(b) == st
                                  for b in sub["body_post"]], "weight"].sum())
            off = o2d.get(f"{mt}->{pg}", {}).get(st)
            if off is None:
                continue
            sign = sub["sign"].mode().iloc[0]
            pv = pv + (wsub * sign * (KIN[mt] - TAU_REF)
                       * np.array([off["du"], off["dv"]]))
        # hex (du,dv) -> physical direction
        pred_vectors[st] = (pv[0] * u_dir + pv[1] * v_dir)

    # comparison table: measured vs predicted direction (deg), |DSI|
    def angle_of(v):
        c = np.dot(v, u_dir) / (np.linalg.norm(v) * np.linalg.norm(u_dir))
        s = np.dot(v, v_dir) / (np.linalg.norm(v) * np.linalg.norm(v_dir))
        return float(np.degrees(np.arctan2(s, c)))

    comparison = {}
    for st in exp005.T45_SUBTYPES:
        mv = np.array(ds_vectors[st]["vector_um"])
        comparison[st] = {
            "measured_dir_deg": round(angle_of(mv), 1),
            "measured_mag": round(float(np.linalg.norm(mv)), 4),
            "predicted_dir_deg": round(angle_of(pred_vectors[st]), 1),
            "predicted_mag": round(float(np.linalg.norm(pred_vectors[st])), 1),
            "amp_mean_hz": round(ds_vectors[st]["amp_mean"], 3),
        }
    summary = {"config": {
        "freq_hz": FREQ_HZ, "n_hex_units": N_HEX_UNITS,
        "lambda_um": round(lam_um, 1), "axes_deg": list(AXES_DEG),
        "kinetics": KIN, "tau_ref_ms": TAU_REF,
        "hex_axis_r2": {"u": round(r2u, 3), "v": round(r2v, 3)},
    }, "ds_vectors": ds_vectors,
        "phi_ds_vector_mag": round(float(np.linalg.norm(phi_vec)), 4),
        "comparison": comparison}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nmeasured vs predicted DS direction (deg, u-axis = 0):")
    print(json.dumps(comparison, indent=1))

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             subplot_kw={"projection": "polar"})
    for ax, key, title in ((axes[0], "measured_dir_deg", "measured"),
                           (axes[1], "predicted_dir_deg", "predicted (HR)")):
        for st in exp005.T45_SUBTYPES:
            ang = np.radians(comparison[st][key])
            mag = (comparison[st]["measured_mag"] if key.startswith("measured")
                   else comparison[st]["predicted_mag"])
            norm = max(v["measured_mag"] for v in comparison.values()) + 1e-9
            ax.plot([ang, ang], [0, mag / norm], "o-",
                    label=f"{st} ({comparison[st][key]:.0f} deg)")
        ax.set_theta_zero_location("E")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=7)
    fig.suptitle("exp006: T4/T5 DS directions vs structure prediction")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ds_polar.png", dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [comparison[st]["predicted_dir_deg"] for st in exp005.T45_SUBTYPES]
    ys = [comparison[st]["measured_dir_deg"] for st in exp005.T45_SUBTYPES]
    for st, x, y in zip(exp005.T45_SUBTYPES, xs, ys):
        ax.scatter(x, y, s=60)
        ax.annotate(st, (x, y), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.plot([-180, 180], [-180, 180], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("predicted direction (deg)")
    ax.set_ylabel("measured direction (deg)")
    ax.set_title("DS direction: measured vs HR prediction")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pred_vs_meas.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
