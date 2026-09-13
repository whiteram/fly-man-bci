"""Experiment 004: retinotopic drifting gratings — temporal frequency tuning
and direction symmetry at the lamina level.

Design (HANDOFF §8): each photoreceptor terminal is projected onto the plane
perpendicular to its eye axis. By neural superposition the lamina cartridge
mosaic is a visual-angle map, so the projected coordinates are a valid
retinotopic proxy. A drifting grating I(x, t) = I_g * (0.5 + 0.5 cos(k x - s w t))
drives every R with its own phase; the drive passes through exp002's
phototransduction cascade (per-R state) into the same R -> L1/L2/L3 circuit
and the three forward kernels.

Testable literature expectations at the lamina level:
- NO direction selectivity: responses to s = +1 and s = -1 are identical in
  modulation amplitude (direction selectivity first appears at T4/T5 in the
  medulla); the measured DSI quantifies the sampling noise floor.
- Modulation amplitude falls with temporal frequency (flicker fusion); with
  tau = 10 ms x2 the analytic cascade attenuation is
  1 / (1 + (2 pi f tau)^2): ~0.99 / 0.97 / 0.91 / 0.72 / 0.39 for 1..20 Hz.

Run from repository root:
    python experiments/exp004_motion_grating/run.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as ft
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata
from ffbm.forward import StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

OUT = Path(__file__).resolve().parent / "outputs"

# ----------------------------- configuration -----------------------------
DT = 0.5            # ms
T_END = 6000.0      # ms
GRAT_ON = 500.0     # grating starts
MEAS_ON = 2000.0    # lock-in measurement window start (adaptation settled)
I_GRATING = 700.0   # pA peak increment (mean 350 = exp002 1x)

FREQS_HZ = (1.0, 2.0, 5.0, 10.0, 20.0)
DIRECTIONS = (+1, -1)

LAMINA_TYPES = ("L1", "L2", "L3")
R_TAU, R_RIN, R_REF = 10.0, 0.1, 3.0
I_R_BASE = 150.0
R_NOISE_SD = 50.0
L_TAU, L_RIN, L_REF = 20.0, 0.1, 2.0
L_NOISE_SD = 20.0
SYN_GAIN = 12.0
SYN_TAU = 5.0

PHOTO_TAU = 10.0        # ms, both cascade stages (symmetric, as exp002)
PHOTO_TAU_ADAPT = 800.0
PHOTO_SAG = 0.30

RHABD_OFFSET_UM = 23.5
ELECTRODE_MARGIN_UM = 20.0
MIN_ELEC_DIST_UM = 30.0
SIGMA = 0.33
SEED = 42

RL_SITES_CACHE = fdata.DERIVED / "rl_sites.parquet"


class PhotoCascadeVector:
    """Per-photoreceptor two-stage cascade + adaptation (exp002 kinetics,
    vectorized over the population because each R has its own grating phase).
    """

    def __init__(self, n, dt, tau=PHOTO_TAU, tau_adapt=PHOTO_TAU_ADAPT,
                 sag=PHOTO_SAG):
        self.k = 1.0 - np.exp(-dt / tau)
        self.ka = 1.0 - np.exp(-dt / tau_adapt)
        self.sag = sag
        self.y1 = np.zeros(n)
        self.y2 = np.zeros(n)
        self.g = np.ones(n)

    def step(self, inc_pa):
        self.y1 += self.k * (inc_pa - self.y1)
        self.y2 += self.k * (self.y1 - self.y2)
        target = np.where(inc_pa > 0.0, 1.0 - self.sag, 1.0)
        self.g += self.ka * (target - self.g)
        return self.g * self.y2


# ----------------------------- geometry / circuit (as exp002/003) ----------
def eye_axis_and_soma_depth(r_pos_map, soma_map, r_ids):
    v = np.array([soma_map[b] - r_pos_map[b] for b in r_ids
                  if b in soma_map and b in r_pos_map])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    axis = u.mean(axis=0)
    axis /= np.linalg.norm(axis)
    keep = (u @ axis) > 0.3
    axis = u[keep].mean(axis=0)
    axis /= np.linalg.norm(axis)
    return axis


def split_sides(x):
    c = np.percentile(x, [25.0, 75.0]).astype(float)
    for _ in range(20):
        lab = np.abs(x - c[0]) < np.abs(x - c[1])
        c = np.array([x[lab].mean(), x[~lab].mean()])
    return float(c.mean()), lab


def build_circuit():
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    sites = fdata.load_neuron_sites()
    r_mean = fdata.site_positions(sites, "PreSyn")
    l_mean = fdata.site_positions(sites, "PostSyn")
    types = nodes.set_index("bodyId")["type"]
    r_ids = np.array(sorted(b for b in nodes.loc[nodes["type"] == "R1-R6", "bodyId"]
                            if b in r_mean), dtype=np.int64)
    l_ids = np.array(sorted(b for b in nodes.loc[nodes["type"].isin(LAMINA_TYPES),
                                                 "bodyId"] if b in l_mean),
                     dtype=np.int64)
    e = edges.assign(type_pre=edges["body_pre"].map(types),
                     type_post=edges["body_post"].map(types))
    e = e[(e["type_pre"] == "R1-R6") & (e["type_post"].isin(LAMINA_TYPES))]
    e = e[["body_pre", "body_post", "weight"]].copy()
    e["pos_pre"] = e["body_pre"].map(r_mean)
    e["pos_post"] = e["body_post"].map(l_mean)
    e = e.dropna(subset=["pos_pre", "pos_post"]).reset_index(drop=True)
    print(f"R->lamina edges: {len(e):,} | R: {len(r_ids):,} | "
          f"L: {len(l_ids):,} | synapses: {int(e['weight'].sum()):,}")
    return e, r_ids, l_ids, r_mean, l_mean


def rl_site_points(r_ids, l_ids):
    if RL_SITES_CACHE.exists():
        df = pd.read_parquet(RL_SITES_CACHE)
    else:
        t = ft.read_table(fdata.RAW / "syn-points.feather", memory_map=True,
                          columns=["x", "y", "z", "kind", "body"])
        sub = t.filter(pc.is_in(t["body"], value_set=pa.array(
            np.concatenate([r_ids, l_ids]))))
        df = sub.to_pandas()
        del t, sub
        for c, cu in (("x", "x_um"), ("y", "y_um"), ("z", "z_um")):
            df[cu] = df[c] * fdata.VOXEL_UM
        df = df.drop(columns=["x", "y", "z"])
        df["body"] = df["body"].astype("int64")
        df.to_parquet(RL_SITES_CACHE)
    return {kind: {int(b): g[["x_um", "y_um", "z_um"]].to_numpy(np.float64)
                   for b, g in s.groupby("body")}
            for kind, s in df.groupby("kind")}


def match_edge_pairs(r_pre_sites, l_post_sites, pre, post, weight):
    pair_pre, pair_post, pair_edge, pair_dist = [], [], [], []
    for e in range(len(pre)):
        P = r_pre_sites[int(pre[e])]
        Q = l_post_sites[int(post[e])]
        w = int(weight[e])
        D = cdist(P, Q)
        d = D.min(axis=0)
        nearest = D.argmin(axis=0)
        n = min(w, len(Q))
        sel = np.argpartition(d, n - 1)[:n]
        pair_pre.append(P[nearest[sel]])
        pair_post.append(Q[sel])
        pair_edge.append(np.full(n, e, dtype=np.int64))
        pair_dist.append(d[sel])
    return (np.concatenate(pair_pre), np.concatenate(pair_post),
            np.concatenate(pair_edge), np.concatenate(pair_dist))


# ----------------------------- simulation -----------------------------
def run_grating(freq_hz, direction, phase0, r_ids, l_ids, r_side_masks,
                l_side_masks, syn_args, edge_coef, edge_coef_neu, mean_field,
                photo_field, n_r, n_l):
    """One grating condition. phase0: per-R static grating phase k_i * x_i."""
    rng = np.random.default_rng(SEED)
    pop_r = LIFPopulation(n_r, DT, tau_m=R_TAU, t_refrac=R_REF, R_m=R_RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=L_TAU, t_refrac=L_REF, R_m=L_RIN)
    syn = ExponentialSynapses(*syn_args, dt=DT, gain=SYN_GAIN, tau_s=SYN_TAU,
                              n_post=n_l)
    photo = PhotoCascadeVector(n_r, DT)

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi_geo = np.zeros((n_field, 2))
    phi_neu = np.zeros((n_field, 2))
    phi_mean = np.zeros((n_field, 2))
    rate_r = np.zeros((n_field, 2))       # per-eye spikes per 1 ms bin
    rate_l = np.zeros((n_field, 2))

    w_t = 2.0 * np.pi * freq_hz * 1e-3  # rad/ms
    for k in range(n_steps):
        t = k * DT
        inc = I_GRATING * (0.5 + 0.5 * np.cos(phase0 - direction * w_t * t)) \
            if t >= GRAT_ON else np.zeros(n_r)
        inc_f = photo.step(inc)
        i_r = I_R_BASE + inc_f + rng.normal(0, R_NOISE_SD, n_r)
        sp_r = pop_r.step(i_r)
        i_l = syn.to_neuron_current() + rng.normal(0, L_NOISE_SD, n_l)
        sp_l = pop_l.step(i_l)
        y = syn.step(r_ids[sp_r])
        if k % 2 == 0:
            i_photo = I_R_BASE + inc_f
            phi_geo[k // 2] = edge_coef @ (y * 1e-12) \
                + photo_field.field(i_photo)
            phi_neu[k // 2] = edge_coef_neu @ (y * 1e-12) \
                + photo_field.field(i_photo)
            phi_mean[k // 2] = mean_field.field(y) \
                + photo_field.field(i_photo)
            for i, m in enumerate(r_side_masks):
                rate_r[k // 2, i] = sp_r[m].sum()
                rate_l[k // 2, i] = sp_l[l_side_masks[i]].sum()

    return {"phi_geo_nV": phi_geo * 1e9, "phi_neu_nV": phi_neu * 1e9,
            "phi_mean_nV": phi_mean * 1e9,
            # per-eye mean population rate (Hz per neuron)
            "rate_r_hz": rate_r * 1000.0 / np.array(
                [m.sum() for m in r_side_masks]),
            "rate_l_hz": rate_l * 1000.0 / np.array(
                [m.sum() for m in l_side_masks])}


def lockin(signal, t_ms, freq_hz, t0, t1):
    """Complex modulation amplitude of signal at freq over [t0, t1] (ms)."""
    m = (t_ms >= t0) & (t_ms < t1)
    z = signal[m] * np.exp(-2j * np.pi * freq_hz * t_ms[m] * 1e-3)
    return z.mean() * 2.0        # x(t) = A cos -> A at the driving freq


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    edges_df, r_ids, l_ids, r_mean, l_mean = build_circuit()
    ann = fdata.load_annotations()
    soma_map = fdata.neuron_positions(ann)
    u_out = eye_axis_and_soma_depth(r_mean, soma_map, r_ids.tolist())

    n_r, n_l = len(r_ids), len(l_ids)
    r_pos = np.array([r_mean[b] for b in r_ids])

    _, r_left = split_sides(r_pos[:, 0])
    side_names = ["L", "R"]
    r_side = np.where(r_left, "L", "R")
    if u_out[0] > 0:
        u_out = u_out * np.array([-1.0, 1.0, 1.0])
    axis_side = {"L": u_out, "R": u_out * np.array([-1.0, 1.0, 1.0])}

    electrodes = []
    for s in side_names:
        cloud = r_pos[r_side == s]
        centroid = cloud.mean(axis=0)
        beyond = float(np.max((cloud - centroid) @ axis_side[s]))
        elec = centroid + (beyond + ELECTRODE_MARGIN_UM) * axis_side[s]
        d = np.linalg.norm(cloud - elec, axis=1)
        if d.min() < MIN_ELEC_DIST_UM:
            elec += (MIN_ELEC_DIST_UM - d.min()) * axis_side[s]
        electrodes.append(elec)
    electrodes = np.array(electrodes)

    # retinotopy: project terminals onto the plane perpendicular to the eye
    # axis; grating axis e1 is perpendicular to both eye axes (zero x-comp)
    e1 = np.cross(u_out, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    phase0 = np.zeros(n_r)              # per-R static phase k_i * x_i
    for s in side_names:
        m = r_side == s
        x = (r_pos[m] - r_pos[m].mean(axis=0)) @ e1
        extent = x.max() - x.min()
        lam = extent / 2.0              # 2 periods across the eye
        phase0[m] = 2.0 * np.pi * x / lam
        print(f"eye {s}: n={m.sum()}, grating-axis extent = {extent:.1f} um, "
              f"lambda = {lam:.1f} um")

    sites = rl_site_points(r_ids, l_ids)
    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float64)
    order = np.lexsort((pre, post))
    n_edges = len(pre)
    pair_pre, pair_post, pair_edge, pair_dist = match_edge_pairs(
        sites["PreSyn"], sites["PostSyn"], pre[order], post[order],
        weight[order])
    assert len(pair_dist) == 234_561
    print(f"pairing: {len(pair_dist):,} pairs, median "
          f"{np.median(pair_dist):.3f} um")

    w_sorted = weight[order]
    pair_field = StaticPairField(pair_pre, pair_post, electrodes, sigma=SIGMA)
    edge_coef = np.stack([
        np.bincount(pair_edge, weights=pair_field.coef[k], minlength=n_edges)
        / w_sorted for k in range(2)])
    mean_field = StaticPairField(
        np.array([r_mean[b] for b in pre[order]]),
        np.array([l_mean[b] for b in post[order]]),
        electrodes, sigma=SIGMA)
    l_out_mean = {b: (sites["PreSyn"][b].mean(axis=0)
                      if b in sites["PreSyn"] else l_mean[b])
                  for b in l_ids.tolist()}
    pair_src = np.array([l_out_mean[int(b)] for b in post[order]])[pair_edge]
    neurite_field = StaticPairField(pair_src, pair_post, electrodes,
                                    sigma=SIGMA)
    edge_coef_neu = np.stack([
        np.bincount(pair_edge, weights=neurite_field.coef[k], minlength=n_edges)
        / w_sorted for k in range(2)])

    rhabd_axis = np.array([axis_side[s] for s in r_side])
    photo_field = StaticPairField(r_pos, r_pos + RHABD_OFFSET_UM * rhabd_axis,
                                  electrodes, sigma=SIGMA)

    syn_args = (pre, post, weight.astype(np.float32),
                np.full(int(l_ids.max()) + 1, -1, dtype=np.int64))
    syn_args[3][l_ids] = np.arange(n_l)
    r_side_masks = [r_side == s for s in side_names]
    l_pos_arr = np.array([l_mean[b] for b in l_ids])
    _, l_left = split_sides(l_pos_arr[:, 0])
    l_side_masks = [l_left, ~l_left]

    t_ms = np.arange(int(T_END / DT) // 2) * 2 * DT
    kern_arrays = {"per_synapse_tbar": "phi_geo_nV",
                   "per_synapse_neurite": "phi_neu_nV",
                   "mean_position": "phi_mean_nV"}

    results = {}
    for f in FREQS_HZ:
        for s_dir in DIRECTIONS:
            print(f"simulating f={f:g} Hz, dir={s_dir:+d} ...")
            results[(f, s_dir)] = run_grating(
                f, s_dir, phase0, r_ids, l_ids, r_side_masks, l_side_masks,
                syn_args, edge_coef, edge_coef_neu, mean_field, photo_field,
                n_r, n_l)
            res = results[(f, s_dir)]
            a0 = abs(lockin(res["phi_geo_nV"][:, 0], t_ms, f, MEAS_ON, T_END))
            print(f"  L eye tbar |phi_mod| = {a0 / 1000:.1f} uV")

    # ---- analysis: modulation amplitude, DSI, phase lag ----
    signals = {"rate_r": None, "phi_tbar": "per_synapse_tbar",
               "phi_mean": "mean_position"}   # sig_label -> kernel (or rate)

    def amp(cond, sig_label, eye_i):
        res = results[cond]
        k = signals[sig_label]
        x = res["rate_r_hz"][:, eye_i] if k is None \
            else res[kern_arrays[k]][:, eye_i]
        return abs(lockin(x, t_ms, cond[0], MEAS_ON, T_END))

    def phase(cond, kern, eye_i):
        return float(np.angle(lockin(
            results[cond][kern_arrays[kern]][:, eye_i], t_ms, cond[0],
            MEAS_ON, T_END)))

    summary = {"config": {
        "dt_ms": DT, "t_end_ms": T_END, "grating_on_ms": GRAT_ON,
        "meas_on_ms": MEAS_ON, "i_grating_pA": I_GRATING,
        "freqs_hz": list(FREQS_HZ), "directions": list(DIRECTIONS),
        "photo_tau_ms": PHOTO_TAU, "photo_tau_adapt_ms": PHOTO_TAU_ADAPT,
        "photo_sag": PHOTO_SAG, "syn_gain_pA": SYN_GAIN,
        "n_photoreceptors": n_r, "n_lamina": n_l,
        "n_edges": n_edges, "n_synapses": int(weight.sum()),
    }, "tuning": {}, "phase_lag_rad": {}, "dsi": {}}

    for sig_label in signals:
        summary["tuning"][sig_label] = {
            s: {f"{f:g}": {f"dir{d:+d}": round(amp((f, d), sig_label, i), 4)
                           for d in DIRECTIONS}
                for f in FREQS_HZ}
            for i, s in enumerate(side_names)}
    for kern in ("per_synapse_tbar", "mean_position"):
        summary["phase_lag_rad"][kern] = {
            s: {f"{f:g}": round(phase((f, +1), kern, i), 3)
                for f in FREQS_HZ}
            for i, s in enumerate(side_names)}
    for sig_label in signals:
        summary["dsi"][sig_label] = {
            s: {f"{f:g}": round(
                (summary["tuning"][sig_label][s][f"{f:g}"]["dir+1"]
                 - summary["tuning"][sig_label][s][f"{f:g}"]["dir-1"])
                / (summary["tuning"][sig_label][s][f"{f:g}"]["dir+1"]
                   + summary["tuning"][sig_label][s][f"{f:g}"]["dir-1"]
                   + 1e-12), 4)
                for f in FREQS_HZ}
            for s in side_names}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("tuning (rate_r, L eye): "
          + json.dumps(summary["tuning"]["rate_r"]["L"]))
    print("DSI (expect ~0): " + json.dumps(summary["dsi"]))

    # ----------------------------- figures -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    zoom = (t_ms >= 3000) & (t_ms < 4000)
    for d, ls in zip(DIRECTIONS, ("-", "--")):
        res = results[(2.0, d)]
        axes[0].plot(t_ms[zoom], res["phi_geo_nV"][zoom, 0] / 1000.0, ls,
                     alpha=0.8, label=f"2 Hz dir{d:+d}")
    res = results[(20.0, +1)]
    axes[0].plot(t_ms[zoom], res["phi_geo_nV"][zoom, 0] / 1000.0, ":",
                 color="tab:red", label="20 Hz dir+1")
    axes[0].set_ylabel("phi (uV), L eye, tbar kernel")
    axes[0].legend(fontsize=8)
    axes[0].set_title("grating-driven modulation: 1 s zoom")
    for d, ls in zip(DIRECTIONS, ("-", "--")):
        res = results[(2.0, d)]
        axes[1].plot(t_ms[zoom], res["rate_r_hz"][zoom, 0], ls, alpha=0.8,
                     label=f"2 Hz dir{d:+d}")
    res = results[(20.0, +1)]
    axes[1].plot(t_ms[zoom], res["rate_r_hz"][zoom, 0], ":", color="tab:red",
                 label="20 Hz dir+1")
    axes[1].set_ylabel("R population rate (Hz)")
    axes[1].set_xlabel("time (ms)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_traces.png", dpi=130)

    x = np.array(FREQS_HZ)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("R rate modulation (Hz)", "rate_r"),
              ("phi tbar (uV)", "phi_tbar"), ("phi mean (uV)", "phi_mean")]
    analytic = 1.0 / (1.0 + (2 * np.pi * x * PHOTO_TAU * 1e-3) ** 2)
    for ax, (title, key) in zip(axes, panels):
        for i, s in enumerate(side_names):
            for d, style in ((+1, "o-"), (-1, "s--")):
                ys = [summary["tuning"][key][s][f"{f:g}"][f"dir{d:+d}"]
                      for f in FREQS_HZ]
                ax.plot(x, ys, style, color=f"C{i}", label=f"{s} dir{d:+d}")
        if key != "rate_r":
            ref = summary["tuning"][key]["L"]["1"]["dir+1"]
            ax.plot(x, ref * analytic / analytic[0], color="k", alpha=0.4,
                    lw=1, label="cascade filter x const")
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{f:g}" for f in x])
        ax.set_xlabel("temporal frequency (Hz)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("exp004: modulation amplitude vs temporal frequency "
                 "(solid = dir +, dashed = dir -)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tuning.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, s in enumerate(side_names):
        for key, style in (("rate_r", "o-"), ("phi_tbar", "s--"),
                           ("phi_mean", "^:")):
            axes[0].plot(x, [abs(summary["dsi"][key][s][f"{f:g}"])
                             for f in FREQS_HZ], style, label=f"{key} {s}")
    axes[0].axhline(0.05, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel("temporal frequency (Hz)")
    axes[0].set_ylabel("|DSI|")
    axes[0].set_title("direction selectivity (expect ~0 at lamina)")
    axes[0].set_xscale("log")
    axes[0].set_xticks(x, [f"{f:g}" for f in x])
    axes[0].legend(fontsize=7)
    for s, style in (("L", "o-"), ("R", "s--")):
        axes[1].plot(x, [summary["phase_lag_rad"]["per_synapse_tbar"][s][f"{f:g}"]
                         for f in FREQS_HZ], style, label=f"{s} tbar")
    axes[1].set_xlabel("temporal frequency (Hz)")
    axes[1].set_ylabel("phi phase (rad)")
    axes[1].set_title("response phase vs driving frequency")
    axes[1].set_xscale("log")
    axes[1].set_xticks(x, [f"{f:g}" for f in x])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_dsi_phase.png", dpi=130)

    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
