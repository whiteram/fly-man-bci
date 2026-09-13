"""Experiment 003: asymmetric phototransduction — can the positive Off
transient be regenerated?

exp002 found that under symmetric cascade kinetics (tau_on = tau_off = 10 ms)
no return-current hypothesis produces a positive Off overshoot: the photocurrent
and the synaptic currents decay on similar timescales, so phi simply returns to
the dark baseline. Real phototransduction is asymmetric — channel deactivation
after light-off is faster than the multi-step activation cascade (Hardie &
Raghu 2001) — which separates the timescales: the photoreceptor dipole
collapses first, while the lamina synaptic component (R membrane tau = 10 ms +
synapse tau_s = 5 ms of extra lag) is still present, transiently unopposed.

Design: exp002's circuit, pairing and three forward kernels, unchanged
(SEED=42; the tau_fall = 10 column must reproduce exp002 bit-for-bit as a
regression check). Sweep tau_fall in {10, 5, 2.5} ms x flash intensity in
{0.25x, 1x, 2x}. Primary metric: Off overshoot = peak_off - dark baseline,
per kernel.

Run from repository root:
    python experiments/exp003_asymmetric_off/run.py
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
DT = 0.5
T_END = 4500.0
FLASH_ON, FLASH_OFF = 500.0, 3500.0     # 3 s flash

TAU_RISE = 10.0                          # ms, both cascade stages
TAU_FALLS = (10.0, 5.0, 2.5)             # ms, both cascade stages
LEVEL_FRACTIONS = (0.25, 1.0, 2.0)
I_R_FLASH = 350.0                        # pA at fraction 1.0

LAMINA_TYPES = ("L1", "L2", "L3")

R_TAU, R_RIN, R_REF = 10.0, 0.1, 3.0     # ms, GOhm, ms
I_R_BASE = 150.0                         # pA
R_NOISE_SD = 50.0
L_TAU, L_RIN, L_REF = 20.0, 0.1, 2.0
L_NOISE_SD = 20.0
SYN_GAIN = 12.0
SYN_TAU = 5.0

PHOTO_TAU_ADAPT = 800.0
PHOTO_SAG = 0.30

RHABD_OFFSET_UM = 23.5
ELECTRODE_MARGIN_UM = 20.0
MIN_ELEC_DIST_UM = 30.0
SIGMA = 0.33

SEED = 42

RL_SITES_CACHE = fdata.DERIVED / "rl_sites.parquet"

# exp002 1x t-bar plateau (L eye) — the tau_fall=10 column must reproduce this
EXP002_REGRESSION_NV = -556_370.1


# ----------------------------- phototransduction -----------------------------
class AsymPhototransduction:
    """Two-stage cascade with direction-dependent time constants.

    Each low-pass stage follows its input with tau_rise while the input is
    above the stage state and with tau_fall while below, so the flash-OFF
    edge propagates through the cascade faster than the ON edge. With
    tau_fall == tau_rise this reproduces exp002's symmetric filter exactly.
    Adaptation as in exp002: multiplicative gain towards (1 - sag) in light.
    """

    def __init__(self, dt, tau_rise=TAU_RISE, tau_fall=TAU_RISE,
                 tau_adapt=PHOTO_TAU_ADAPT, sag=PHOTO_SAG):
        self.k1r = 1.0 - np.exp(-dt / tau_rise)
        self.k1f = 1.0 - np.exp(-dt / tau_fall)
        self.k2r, self.k2f = self.k1r, self.k1f
        self.ka = 1.0 - np.exp(-dt / tau_adapt)
        self.sag = sag
        self.reset()

    def reset(self):
        self.y1 = self.y2 = 0.0
        self.g = 1.0

    def step(self, inc_pa: float) -> float:
        k1 = self.k1r if inc_pa > self.y1 else self.k1f
        self.y1 += k1 * (inc_pa - self.y1)
        k2 = self.k2r if self.y1 > self.y2 else self.k2f
        self.y2 += k2 * (self.y1 - self.y2)
        target = 1.0 - self.sag if inc_pa > 0.0 else 1.0
        self.g += self.ka * (target - self.g)
        return self.g * self.y2


# ----------------------------- geometry / circuit (as exp002) -----------------------------
def eye_axis_and_soma_depth(r_pos_map, soma_map, r_ids):
    v = np.array([soma_map[b] - r_pos_map[b] for b in r_ids
                  if b in soma_map and b in r_pos_map])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    axis = u.mean(axis=0)
    axis /= np.linalg.norm(axis)
    keep = (u @ axis) > 0.3
    axis = u[keep].mean(axis=0)
    axis /= np.linalg.norm(axis)
    depth = float(np.linalg.norm(v, axis=1).mean())
    print(f"eye axis calibrated from {keep.sum()} photoreceptors: "
          f"u_out={axis.round(3)}, soma depth={depth:.1f} um")
    return axis, depth


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
    l_type = np.array([types.get(b) for b in l_ids])

    e = edges.assign(type_pre=edges["body_pre"].map(types),
                     type_post=edges["body_post"].map(types))
    e = e[(e["type_pre"] == "R1-R6") & (e["type_post"].isin(LAMINA_TYPES))]
    e = e[["body_pre", "body_post", "weight"]].copy()
    e["pos_pre"] = e["body_pre"].map(r_mean)
    e["pos_post"] = e["body_post"].map(l_mean)
    e = e.dropna(subset=["pos_pre", "pos_post"]).reset_index(drop=True)
    print(f"R->lamina edges with site positions: {len(e):,} | "
          f"photoreceptors: {len(r_ids):,} | lamina: {len(l_ids):,} | "
          f"synapses: {int(e['weight'].sum()):,}")
    return e, r_ids, l_ids, l_type, r_mean, l_mean


def rl_site_points(r_ids, l_ids):
    if RL_SITES_CACHE.exists():
        df = pd.read_parquet(RL_SITES_CACHE)
        print(f"site points loaded from cache {RL_SITES_CACHE.name} "
              f"({len(df):,} rows)")
    else:
        src = fdata.RAW / "syn-points.feather"
        print(f"extracting circuit site points from {src.name} "
              f"(memory-mapped) ...")
        t = ft.read_table(src, memory_map=True,
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
        print(f"cached {len(df):,} site rows -> {RL_SITES_CACHE}")
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
    print(f"pairing: {sum(map(len, pair_dist)):,} t-bar/PSD pairs")
    return (np.concatenate(pair_pre), np.concatenate(pair_post),
            np.concatenate(pair_edge), np.concatenate(pair_dist))


# ----------------------------- simulation -----------------------------
def run_condition(level_pa, tau_fall, r_ids, l_ids, l_type, syn_args,
                  edge_coef, edge_coef_neu, mean_field, photo_field, n_r, n_l):
    rng = np.random.default_rng(SEED)
    pop_r = LIFPopulation(n_r, DT, tau_m=R_TAU, t_refrac=R_REF, R_m=R_RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=L_TAU, t_refrac=L_REF, R_m=L_RIN)
    syn = ExponentialSynapses(*syn_args, dt=DT, gain=SYN_GAIN, tau_s=SYN_TAU,
                              n_post=n_l)
    photo = AsymPhototransduction(DT, TAU_RISE, tau_fall)

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi_geo = np.zeros((n_field, 2))
    phi_neu = np.zeros((n_field, 2))
    phi_mean = np.zeros((n_field, 2))

    for k in range(n_steps):
        t = k * DT
        flash = level_pa if FLASH_ON <= t < FLASH_OFF else 0.0
        inc = photo.step(flash)
        i_r = I_R_BASE + inc + rng.normal(0, R_NOISE_SD, n_r)
        sp_r = pop_r.step(i_r)
        i_l = syn.to_neuron_current() + rng.normal(0, L_NOISE_SD, n_l)
        sp_l = pop_l.step(i_l)
        y = syn.step(r_ids[sp_r])
        if k % 2 == 0:
            i_photo = I_R_BASE + inc
            phi_geo[k // 2] = edge_coef @ (y * 1e-12) \
                + photo_field.field(np.full(n_r, i_photo))
            phi_neu[k // 2] = edge_coef_neu @ (y * 1e-12) \
                + photo_field.field(np.full(n_r, i_photo))
            phi_mean[k // 2] = mean_field.field(y) \
                + photo_field.field(np.full(n_r, i_photo))

    return {"phi_geo_nV": phi_geo * 1e9, "phi_neu_nV": phi_neu * 1e9,
            "phi_mean_nV": phi_mean * 1e9}


def window_metrics(phi_nV, t_ms, smooth_ms=5.0):
    w = max(1, int(smooth_ms))
    s = np.convolve(phi_nV, np.ones(w) / w, mode="same")
    base = float(np.median(s[(t_ms > 100.0) & (t_ms < FLASH_ON - 100.0)]))
    on = (t_ms >= FLASH_ON) & (t_ms < FLASH_ON + 150.0)
    early = (t_ms > FLASH_ON + 300.0) & (t_ms < FLASH_ON + 800.0)
    late = (t_ms > FLASH_OFF - 500.0) & (t_ms < FLASH_OFF)
    off = (t_ms >= FLASH_OFF) & (t_ms < FLASH_OFF + 300.0)
    i_on = int(np.argmin(s[on]))
    i_off = int(np.argmax(s[off]))
    plateau_early = float(np.median(s[early]))
    plateau_late = float(np.median(s[late]))
    peak_off = float(s[off][i_off])

    depth = plateau_early - base
    onset = (t_ms >= FLASH_ON) & (t_ms < FLASH_ON + 300.0)
    crossed_on = (s[onset] <= base + 0.5 * depth if depth < 0
                  else s[onset] >= base + 0.5 * depth)
    t50_on = (float(t_ms[onset][crossed_on][0] - FLASH_ON)
              if crossed_on.any() else None)
    decay = plateau_late - base
    offset = (t_ms >= FLASH_OFF) & (t_ms < FLASH_OFF + 300.0)
    crossed_off = (s[offset] >= base + 0.5 * decay if decay < 0
                   else s[offset] <= base + 0.5 * decay)
    t50_off = (float(t_ms[offset][crossed_off][0] - FLASH_OFF)
               if crossed_off.any() else None)

    return {
        "baseline": round(base, 1),
        "peak_on": round(float(s[on][i_on]), 1),
        "t50_on_ms": None if t50_on is None else round(t50_on, 1),
        "plateau_early": round(plateau_early, 1),
        "plateau_late": round(plateau_late, 1),
        "peak_off": round(peak_off, 1),
        "off_overshoot": round(peak_off - base, 1),
        "t50_off_ms": None if t50_off is None else round(t50_off, 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    edges_df, r_ids, l_ids, l_type, r_mean, l_mean = build_circuit()
    ann = fdata.load_annotations()
    soma_map = fdata.neuron_positions(ann)
    u_out, _ = eye_axis_and_soma_depth(r_mean, soma_map, r_ids.tolist())

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
        print(f"electrode {s}: n_R={len(cloud)}, min dist to terminal = "
              f"{np.linalg.norm(cloud - elec, axis=1).min():.1f} um")
    electrodes = np.array(electrodes)

    sites = rl_site_points(r_ids, l_ids)
    r_pre_sites = sites["PreSyn"]
    l_post_sites = sites["PostSyn"]

    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float64)
    order = np.lexsort((pre, post))
    n_edges = len(pre)

    pair_pre, pair_post, pair_edge, pair_dist = match_edge_pairs(
        r_pre_sites, l_post_sites, pre[order], post[order], weight[order])

    pair_field = StaticPairField(pair_pre, pair_post, electrodes, sigma=SIGMA)
    w_sorted = weight[order]
    edge_coef = np.stack([
        np.bincount(pair_edge, weights=pair_field.coef[k], minlength=n_edges)
        / w_sorted
        for k in range(len(electrodes))])

    mean_field = StaticPairField(
        np.array([r_mean[b] for b in pre[order]]),
        np.array([l_mean[b] for b in post[order]]),
        electrodes, sigma=SIGMA)

    l_out_mean = {}
    for b in l_ids.tolist():
        arr = sites["PreSyn"].get(b)
        l_out_mean[b] = arr.mean(axis=0) if arr is not None and len(arr) \
            else l_mean[b]
    pair_src = np.array([l_out_mean[int(b)] for b in post[order]])[pair_edge]
    neurite_field = StaticPairField(pair_src, pair_post, electrodes, sigma=SIGMA)
    edge_coef_neu = np.stack([
        np.bincount(pair_edge, weights=neurite_field.coef[k], minlength=n_edges)
        / w_sorted
        for k in range(len(electrodes))])

    print(f"pair distances: median {np.median(pair_dist):.3f} um, "
          f"n_pairs {len(pair_dist):,} (regression: expect 234,561)")
    assert len(pair_dist) == 234_561

    rhabd_axis = np.array([axis_side[s] for s in r_side])
    rhabd_pos = r_pos + RHABD_OFFSET_UM * rhabd_axis
    photo_field = StaticPairField(r_pos, rhabd_pos, electrodes, sigma=SIGMA)

    syn_args = (pre, post, weight.astype(np.float32),
                np.full(int(l_ids.max()) + 1, -1, dtype=np.int64))
    syn_args[3][l_ids] = np.arange(n_l)

    t_ms = np.arange(int(T_END / DT) // 2) * 2 * DT
    levels = {f"{fr:g}x": I_R_FLASH * fr for fr in LEVEL_FRACTIONS}
    kern_arrays = {"per_synapse_tbar": "phi_geo_nV",
                   "per_synapse_neurite": "phi_neu_nV",
                   "mean_position": "phi_mean_nV"}

    results = {}
    for tf in TAU_FALLS:
        for lname, level_pa in levels.items():
            key = (tf, lname)
            print(f"simulating tau_fall={tf:g} ms, level {lname} ...")
            res = run_condition(level_pa, tf, r_ids, l_ids, l_type, syn_args,
                                edge_coef, edge_coef_neu, mean_field,
                                photo_field, n_r, n_l)
            results[key] = res
            m = window_metrics(res["phi_geo_nV"][:, 0], t_ms)
            print(f"  L eye tbar: plateau_late {m['plateau_late']:.0f} nV, "
                  f"off_overshoot {m['off_overshoot']:+.0f} nV")

    # regression: tau_fall=10 at 1x must reproduce exp002 bit-for-bit
    reg = window_metrics(results[(10.0, "1x")]["phi_geo_nV"][:, 0], t_ms)
    print(f"regression vs exp002: plateau_late {reg['plateau_late']} "
          f"(expect {EXP002_REGRESSION_NV})")
    assert abs(reg["plateau_late"] - EXP002_REGRESSION_NV) < 1.0, \
        "tau_fall=tau_rise failed to reproduce exp002"

    # ---- summary ----
    summary = {
        "config": {
            "dt_ms": DT, "t_end_ms": T_END,
            "flash_ms": [FLASH_ON, FLASH_OFF],
            "tau_rise_ms": TAU_RISE, "tau_falls_ms": list(TAU_FALLS),
            "level_fractions": list(LEVEL_FRACTIONS),
            "i_r_flash_pA": I_R_FLASH,
            "phototransduction": {"tau_adapt_ms": PHOTO_TAU_ADAPT,
                                  "sag": PHOTO_SAG},
            "syn_gain_pA": SYN_GAIN, "syn_tau_ms": SYN_TAU,
            "i_r_base_pA": I_R_BASE,
            "n_photoreceptors": n_r, "n_lamina": n_l,
            "n_edges": n_edges, "n_synapses": int(weight.sum()),
        },
        "pairing": {
            "n_pairs": int(len(pair_dist)),
            "pair_dist_um_median": round(float(np.median(pair_dist)), 3),
        },
        "conditions": {},
    }
    for (tf, lname), res in results.items():
        summary["conditions"][f"tfall={tf:g}/{lname}"] = {
            "phi_nV": {
                s: {kern: window_metrics(res[arr][:, i], t_ms)
                    for kern, arr in kern_arrays.items()}
                for i, s in enumerate(side_names)
            }
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----------------------------- figures -----------------------------
    # fig1: waveforms per condition (L eye, 1x): 3 kernels x 3 tau_fall
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, tf in zip(axes, TAU_FALLS):
        res = results[(tf, "1x")]
        for kern, arr, style in (("per_synapse_tbar", "phi_geo_nV", "-"),
                                 ("per_synapse_neurite", "phi_neu_nV", "-"),
                                 ("mean_position", "phi_mean_nV", "--")):
            ax.plot(t_ms, res[arr][:, 0] / 1000.0, style, alpha=0.8,
                    label=kern)
        ax.axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
        ax.set_title(f"tau_fall = {tf:g} ms")
        ax.set_xlabel("time (ms)")
    axes[0].set_ylabel("phi (uV), L eye")
    axes[0].legend(fontsize=8)
    fig.suptitle("exp003: asymmetric phototransduction vs return-current "
                 "hypothesis (1x flash)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_waveforms.png", dpi=130)

    # fig2: Off overshoot vs tau_fall — one panel per kernel, color = eye,
    # alpha = intensity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, (kern, arr) in zip(axes, kern_arrays.items()):
        for i, s in enumerate(side_names):
            for fr, alpha in zip(LEVEL_FRACTIONS, (0.45, 1.0, 0.7)):
                ys = [window_metrics(results[(tf, f"{fr:g}x")][arr][:, i],
                                     t_ms)["off_overshoot"] / 1000.0
                      for tf in TAU_FALLS]
                ax.plot(TAU_FALLS, ys, "o-", color=f"C{i}", alpha=alpha,
                        label=f"{s} {fr:g}x")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("tau_fall (ms)")
        ax.set_ylabel("Off overshoot (uV)")
        ax.set_title(kern)
        ax.set_xticks(TAU_FALLS)
        ax.legend(fontsize=7)
    fig.suptitle("exp003: Off overshoot vs deactivation speed "
                 "(>0 = positive Off transient)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_off_overshoot.png", dpi=130)

    # fig3: Off transition zoom (L eye, 1x)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    zoom = (t_ms >= 3400) & (t_ms <= 3700)
    for tf, color in zip(TAU_FALLS, ("tab:blue", "tab:orange", "tab:green")):
        res = results[(tf, "1x")]
        ax.plot(t_ms[zoom], res["phi_neu_nV"][zoom, 0] / 1000.0, color=color,
                label=f"neurite, tau_fall={tf:g}")
        ax.plot(t_ms[zoom], res["phi_geo_nV"][zoom, 0] / 1000.0, color=color,
                ls="--", alpha=0.6, label=f"tbar, tau_fall={tf:g}")
    ax.axvline(FLASH_OFF, color="k", ls=":", label="flash off")
    ax.axhline(-209.6, color="gray", lw=0.8, ls="-.", label="dark baseline")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("phi (uV), L eye")
    ax.set_title("Off transition zoom (1x flash)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_off_zoom.png", dpi=130)

    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
