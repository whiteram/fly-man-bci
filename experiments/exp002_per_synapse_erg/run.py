"""Experiment 002: per-synapse dipoles + phototransduction + intensity series.

Upgrades over exp001 (design in HANDOFF.md §8):
1. Per-synapse geometry: every R->lamina synapse becomes its own t-bar->PSD
   current dipole instead of one dipole between per-neuron mean positions.
   Within each edge (R_i, L_j, w) PSDs are greedily matched to their nearest
   t-bar of R_i (PSDs are the constrained resource; a t-bar may serve several
   PSDs of the same edge -- tetrad architecture). The per-edge coefficient is
   the mean over its w pair coefficients: sum_pairs (y/w) c = y mean(c), so the
   runtime cost is unchanged.
2. Phototransduction dynamics: rectangular flash -> two-stage first-order
   low-pass (tau = 10 ms each, ~20 ms receptor delay) and slow light adaptation
   (tau = 800 ms, 30% plateau sag).
3. Intensity series: 4 flash strengths x 3 s flash; plateau and On/Off
   transient amplitudes vs intensity, onset latency and adaptation sag.

Run from repository root:
    python experiments/exp002_per_synapse_erg/run.py
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
T_END = 4500.0      # ms
FLASH_ON, FLASH_OFF = 500.0, 3500.0     # 3 s flash

LEVEL_FRACTIONS = (0.25, 0.5, 1.0, 2.0)  # flash strengths, x I_R_FLASH
I_R_FLASH = 350.0                        # pA at fraction 1.0

LAMINA_TYPES = ("L1", "L2", "L3")

# photoreceptor (R) parameters (as exp001: only geometry/dynamics change)
R_TAU, R_RIN, R_REF = 10.0, 0.1, 3.0     # ms, GOhm, ms
I_R_BASE = 150.0                         # pA
R_NOISE_SD = 50.0                        # pA

# lamina (L) parameters
L_TAU, L_RIN, L_REF = 20.0, 0.1, 2.0
L_NOISE_SD = 20.0                        # pA

SYN_GAIN = 12.0     # pA per synapse (weight unit)
SYN_TAU = 5.0       # ms

# phototransduction cascade (Hardie & Raghu 2001: ~20 ms receptor delay)
PHOTO_TAU1, PHOTO_TAU2 = 10.0, 10.0      # ms, two-stage low-pass
PHOTO_TAU_ADAPT = 800.0                  # ms, light adaptation
PHOTO_SAG = 0.30                         # plateau sag fraction

RHABD_OFFSET_UM = 23.5   # terminal -> rhabdomere depth along u_out (calibrated)
ELECTRODE_MARGIN_UM = 20.0
MIN_ELEC_DIST_UM = 30.0
SIGMA = 0.33             # S/m

SEED = 42

RL_SITES_CACHE = fdata.DERIVED / "rl_sites.parquet"


# ----------------------------- phototransduction -----------------------------
class Phototransduction:
    """Flash increment -> two-stage low-pass, then multiplicative adaptation.

    step(inc_pA) returns the adapted increment; the caller adds its own dark
    baseline. The adaptation gain g decays towards (1 - sag) while light is on
    and recovers towards 1 in the dark, both with tau_adapt.
    """

    def __init__(self, dt, tau1=PHOTO_TAU1, tau2=PHOTO_TAU2,
                 tau_adapt=PHOTO_TAU_ADAPT, sag=PHOTO_SAG):
        self.k1 = 1.0 - np.exp(-dt / tau1)
        self.k2 = 1.0 - np.exp(-dt / tau2)
        self.ka = 1.0 - np.exp(-dt / tau_adapt)
        self.sag = sag
        self.reset()

    def reset(self):
        self.y1 = self.y2 = 0.0
        self.g = 1.0

    def step(self, inc_pa: float) -> float:
        self.y1 += self.k1 * (inc_pa - self.y1)
        self.y2 += self.k2 * (self.y1 - self.y2)
        target = 1.0 - self.sag if inc_pa > 0.0 else 1.0
        self.g += self.ka * (target - self.g)
        return self.g * self.y2


# ----------------------------- geometry -----------------------------
def eye_axis_and_soma_depth(r_pos_map, soma_map, r_ids):
    """u_out (towards the eye surface) and terminal->soma depth, from the few
    photoreceptors that have annotated soma positions (as exp001)."""
    v = np.array([soma_map[b] - r_pos_map[b] for b in r_ids
                  if b in soma_map and b in r_pos_map])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    axis = u.mean(axis=0)
    axis /= np.linalg.norm(axis)
    keep = (u @ axis) > 0.3                       # drop inconsistent cells
    axis = u[keep].mean(axis=0)
    axis /= np.linalg.norm(axis)
    depth = float(np.linalg.norm(v, axis=1).mean())
    print(f"eye axis calibrated from {keep.sum()} photoreceptors: "
          f"u_out={axis.round(3)}, soma depth={depth:.1f} um")
    return axis, depth


def split_sides(x):
    """1D 2-means split of a coordinate array -> (midline, mask_left)."""
    c = np.percentile(x, [25.0, 75.0]).astype(float)
    for _ in range(20):
        lab = np.abs(x - c[0]) < np.abs(x - c[1])
        c = np.array([x[lab].mean(), x[~lab].mean()])
    return float(c.mean()), lab


# ----------------------------- circuit & sites -----------------------------
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
    """Individual synapse-site coordinates (um) for the R/L circuit bodies.

    Cached to data/derived/rl_sites.parquet; rebuilt from the 12.7 GB
    syn-points feather (memory-mapped, Arrow-native filter) when absent.
    Returns {kind: {bodyId: (n_sites, 3) array}}.
    """
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
    """Greedy PSD->nearest-t-bar matching per edge, in the given edge order.

    One pair per PSD (the constrained resource); t-bars are reusable because a
    single t-bar serves several PSDs of the same edge (tetrad architecture).
    For "w closest pairs with distinct PSDs" the greedy optimum is the nearest
    t-bar per PSD, keeping the w PSDs with the smallest distance.

    Returns (pair_pre, pair_post, pair_edge, pair_dist), where pair_edge maps
    each pair onto its edge index.
    """
    pair_pre, pair_post, pair_edge, pair_dist = [], [], [], []
    fewer = 0
    for e in range(len(pre)):
        P = r_pre_sites[int(pre[e])]
        Q = l_post_sites[int(post[e])]
        w = int(weight[e])
        if len(Q) < w:
            fewer += 1
        D = cdist(P, Q)                    # (n_tbar, n_psd)
        d = D.min(axis=0)                  # nearest t-bar per PSD
        nearest = D.argmin(axis=0)
        n = min(w, len(Q))
        sel = np.argpartition(d, n - 1)[:n]  # n closest PSDs
        pair_pre.append(P[nearest[sel]])
        pair_post.append(Q[sel])
        pair_edge.append(np.full(n, e, dtype=np.int64))
        pair_dist.append(d[sel])
    print(f"pairing: {sum(map(len, pair_dist)):,} t-bar/PSD pairs | "
          f"{fewer:,} edges with fewer PSDs than synapses")
    return (np.concatenate(pair_pre), np.concatenate(pair_post),
            np.concatenate(pair_edge), np.concatenate(pair_dist))


# ----------------------------- simulation -----------------------------
def run_level(level_pa, r_ids, l_ids, l_type, syn_args, edge_coef, edge_coef_neu,
              mean_field, photo_field, n_r, n_l):
    """One intensity level: fresh populations/synapses, identical noise stream.

    Returns per-ms traces for all three forward kernels plus spike rates.
    """
    rng = np.random.default_rng(SEED)
    pop_r = LIFPopulation(n_r, DT, tau_m=R_TAU, t_refrac=R_REF, R_m=R_RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=L_TAU, t_refrac=L_REF, R_m=L_RIN)
    syn = ExponentialSynapses(*syn_args, dt=DT, gain=SYN_GAIN, tau_s=SYN_TAU,
                              n_post=n_l)
    photo = Phototransduction(DT)

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    phi_geo = np.zeros((n_field, 2))
    phi_neu = np.zeros((n_field, 2))
    phi_mean = np.zeros((n_field, 2))
    i_photo_trace = np.zeros(n_field)
    spike_r = np.zeros((n_steps, n_r), dtype=bool)
    spike_l = np.zeros((n_steps, n_l), dtype=bool)

    for k in range(n_steps):
        t = k * DT
        flash = level_pa if FLASH_ON <= t < FLASH_OFF else 0.0
        inc = photo.step(flash)
        i_r = I_R_BASE + inc + rng.normal(0, R_NOISE_SD, n_r)
        sp_r = pop_r.step(i_r)
        i_l = syn.to_neuron_current() + rng.normal(0, L_NOISE_SD, n_l)
        sp_l = pop_l.step(i_l)
        y = syn.step(r_ids[sp_r])
        spike_r[k] = sp_r
        spike_l[k] = sp_l
        if k % 2 == 0:
            i_photo = I_R_BASE + inc
            i_photo_trace[k // 2] = i_photo
            phi_geo[k // 2] = edge_coef @ (y * 1e-12) \
                + photo_field.field(np.full(n_r, i_photo))
            phi_neu[k // 2] = edge_coef_neu @ (y * 1e-12) \
                + photo_field.field(np.full(n_r, i_photo))
            phi_mean[k // 2] = mean_field.field(y) \
                + photo_field.field(np.full(n_r, i_photo))

    def w_rate(spikes, t0, t1):
        return float(spikes[int(t0 / DT):int(t1 / DT)].mean() / (DT * 1e-3))

    rates = {
        "R": {"baseline": w_rate(spike_r, 0, FLASH_ON),
              "flash": w_rate(spike_r, FLASH_ON + 100, FLASH_OFF)},
        **{ty: {"flash": w_rate(spike_l[:, l_type == ty],
                                FLASH_ON + 100, FLASH_OFF)}
           for ty in LAMINA_TYPES},
    }
    return {"phi_geo_nV": phi_geo * 1e9, "phi_neu_nV": phi_neu * 1e9,
            "phi_mean_nV": phi_mean * 1e9,
            "i_photo_pA": i_photo_trace, "rates_hz": rates}


# ----------------------------- metrics -----------------------------
def window_metrics(phi_nV, t_ms, smooth_ms=5.0):
    """Peak/plateau/latency metrics on a boxcar-smoothed trace (per eye).

    t50_on/off are half-depth crossing delays relative to flash edges -- the
    standard ERG latency measure, robust against the flat peak of an
    adapting trace. Depth is measured from the dark baseline.
    """
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

    depth = plateau_early - base                     # response polarity
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
        "trough_on_ms": round(float(t_ms[on][i_on] - FLASH_ON), 1),
        "plateau_early": round(plateau_early, 1),
        "plateau_late": round(plateau_late, 1),
        "sag_ratio": round(plateau_late / plateau_early, 3),
        "peak_off": round(float(s[off][i_off]), 1),
        "t50_off_ms": None if t50_off is None else round(t50_off, 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- circuit, eye axes, electrodes (identical construction to exp001) ----
    edges_df, r_ids, l_ids, l_type, r_mean, l_mean = build_circuit()
    ann = fdata.load_annotations()
    soma_map = fdata.neuron_positions(ann)
    u_out, _ = eye_axis_and_soma_depth(r_mean, soma_map, r_ids.tolist())

    n_r, n_l = len(r_ids), len(l_ids)
    r_pos = np.array([r_mean[b] for b in r_ids])

    _, r_left = split_sides(r_pos[:, 0])
    side_names = ["L", "R"]
    r_side = np.where(r_left, "L", "R")
    if u_out[0] > 0:  # calibration axis must point towards the left cluster
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

    # ---- per-site extraction + greedy pairing (edge order = syn sort) ----
    sites = rl_site_points(r_ids, l_ids)
    r_pre_sites = sites["PreSyn"]
    l_post_sites = sites["PostSyn"]

    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float64)
    order = np.lexsort((pre, post))     # ExponentialSynapses internal order
    n_edges = len(pre)

    pair_pre, pair_post, pair_edge, pair_dist = match_edge_pairs(
        r_pre_sites, l_post_sites, pre[order], post[order], weight[order])

    # per-synapse kernel: pair coefficients -> per-edge mean (divide by w:
    # each of the w synapse currents of the edge carries y/w)
    pair_field = StaticPairField(pair_pre, pair_post, electrodes, sigma=SIGMA)
    w_sorted = weight[order]
    edge_coef = np.stack([
        np.bincount(pair_edge, weights=pair_field.coef[k], minlength=n_edges)
        / w_sorted
        for k in range(len(electrodes))])

    # mean-position kernel (exp001 replica) for comparison
    mean_field = StaticPairField(
        np.array([r_mean[b] for b in pre[order]]),
        np.array([l_mean[b] for b in post[order]]),
        electrodes, sigma=SIGMA)

    # neurite-return kernel: sink at the matched PSD (lamina), source at the
    # same L cell's PreSyn cloud mean (medulla) -- the hypothesis that the
    # postsynaptic return current exits along the neurite's distal span
    l_out_mean = {}
    degenerate = 0
    for b in l_ids.tolist():
        arr = sites["PreSyn"].get(b)
        if arr is not None and len(arr):
            l_out_mean[b] = arr.mean(axis=0)
        else:
            l_out_mean[b] = l_mean[b]     # no PreSyn cloud: return at PSD mean
            degenerate += 1
    print(f"neurite-return sources: {len(l_ids) - degenerate:,} lamina cells "
          f"with PreSyn clouds, {degenerate} degenerated to PSD mean")
    pair_src = np.array([l_out_mean[int(b)] for b in post[order]])[pair_edge]
    neurite_field = StaticPairField(pair_src, pair_post, electrodes, sigma=SIGMA)
    edge_coef_neu = np.stack([
        np.bincount(pair_edge, weights=neurite_field.coef[k], minlength=n_edges)
        / w_sorted
        for k in range(len(electrodes))])

    # photoreceptor dipole (as exp001: sink at the extrapolated rhabdomere)
    rhabd_axis = np.array([axis_side[s] for s in r_side])
    rhabd_pos = r_pos + RHABD_OFFSET_UM * rhabd_axis
    photo_field = StaticPairField(r_pos, rhabd_pos, electrodes, sigma=SIGMA)

    pairing_stats = {
        "n_pairs": int(len(pair_dist)),
        "pair_dist_um": {
            "median": round(float(np.median(pair_dist)), 3),
            "p90": round(float(np.percentile(pair_dist, 90)), 3),
            "p99": round(float(np.percentile(pair_dist, 99)), 3),
            "max": round(float(pair_dist.max()), 3),
        },
        "coef_norm_vs_mean_position": {
            "per_synapse_tbar": {
                s: round(float(np.linalg.norm(edge_coef[i]) /
                              np.linalg.norm(mean_field.coef[i])), 5)
                for i, s in enumerate(side_names)
            },
            "per_synapse_neurite": {
                s: round(float(np.linalg.norm(edge_coef_neu[i]) /
                              np.linalg.norm(mean_field.coef[i])), 5)
                for i, s in enumerate(side_names)
            },
        },
    }
    print("pairing stats:", json.dumps(pairing_stats))

    syn_args = (pre, post, weight.astype(np.float32),
                np.full(int(l_ids.max()) + 1, -1, dtype=np.int64))
    syn_args[3][l_ids] = np.arange(n_l)

    # ---- intensity series ----
    t_ms = np.arange(int(T_END / DT) // 2) * 2 * DT
    levels = {f"{fr:g}x": I_R_FLASH * fr for fr in LEVEL_FRACTIONS}
    results = {}
    for name, level_pa in levels.items():
        print(f"simulating level {name} ({level_pa:.0f} pA) ...")
        results[name] = run_level(level_pa, r_ids, l_ids, l_type, syn_args,
                                  edge_coef, edge_coef_neu, mean_field,
                                  photo_field, n_r, n_l)
        r_flash = results[name]["rates_hz"]["R"]["flash"]
        print(f"  R flash rate {r_flash:.0f} Hz | "
              f"plateau(tbar, L eye) {np.median(results[name]['phi_geo_nV'][(t_ms > FLASH_OFF - 500) & (t_ms < FLASH_OFF), 0]):.0f} nV")

    # ---- summary ----
    summary = {
        "config": {
            "dt_ms": DT, "t_end_ms": T_END,
            "flash_ms": [FLASH_ON, FLASH_OFF],
            "level_fractions": list(LEVEL_FRACTIONS),
            "i_r_flash_pA": I_R_FLASH,
            "phototransduction": {"tau1_ms": PHOTO_TAU1, "tau2_ms": PHOTO_TAU2,
                                  "tau_adapt_ms": PHOTO_TAU_ADAPT,
                                  "sag": PHOTO_SAG},
            "syn_gain_pA": SYN_GAIN, "syn_tau_ms": SYN_TAU,
            "i_r_base_pA": I_R_BASE,
            "n_photoreceptors": n_r, "n_lamina": n_l,
            "n_edges": n_edges, "n_synapses": int(weight.sum()),
        },
        "pairing": pairing_stats,
        "levels": {},
    }
    for name, res in results.items():
        summary["levels"][name] = {
            "rates_hz": {k: {kk: round(vv, 1) for kk, vv in v.items()}
                         for k, v in res["rates_hz"].items()},
            "phi_nV": {
                s: {
                    "per_synapse_tbar": window_metrics(res["phi_geo_nV"][:, i],
                                                       t_ms),
                    "per_synapse_neurite": window_metrics(res["phi_neu_nV"][:, i],
                                                          t_ms),
                    "mean_position": window_metrics(res["phi_mean_nV"][:, i],
                                                    t_ms),
                }
                for i, s in enumerate(side_names)
            },
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["levels"]["1x"]["phi_nV"]["L"], indent=2))

    # ----------------------------- figures -----------------------------
    # fig1: ERG waveform per intensity level (per-synapse kernel)
    fig, axes = plt.subplots(len(levels), 1, figsize=(9, 3 * len(levels)),
                             sharex=True)
    for ax, (name, res) in zip(axes, results.items()):
        for i, s in enumerate(side_names):
            ax.plot(t_ms, res["phi_geo_nV"][:, i] / 1000.0,
                    label=f"{s} electrode", alpha=0.85)
        ax.axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
        ax.set_ylabel(f"{name}\nphi (uV)")
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("exp002: ERG waveform vs flash intensity "
                 "(per-synapse dipoles, t-bar return)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_levels.png", dpi=130)

    # fig2: amplitude & latency metrics vs intensity
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = np.array(LEVEL_FRACTIONS)
    for i, s in enumerate(side_names):
        for key, style in (("plateau_late", "o-"), ("peak_on", "s--"),
                           ("peak_off", "^:")):
            axes[0, i].plot(x, [abs(summary["levels"][f"{fr:g}x"]["phi_nV"][s]
                                    ["per_synapse_tbar"][key]) / 1000.0
                                for fr in x], style, label=key)
        axes[0, i].set_xscale("log")
        axes[0, i].set_xticks(x, [f"{fr:g}x" for fr in x])
        axes[0, i].set_xlabel("flash intensity")
        axes[0, i].set_ylabel("|amplitude| (uV)")
        axes[0, i].set_title(f"{s} eye")
        axes[0, i].legend(fontsize=8)
        for key, style in (("t50_on_ms", "o-"), ("t50_off_ms", "s--")):
            axes[1, i].plot(x, [summary["levels"][f"{fr:g}x"]["phi_nV"][s]
                                ["per_synapse_tbar"][key] for fr in x],
                            style, label=key)
        axes[1, i].set_xscale("log")
        axes[1, i].set_xticks(x, [f"{fr:g}x" for fr in x])
        axes[1, i].set_xlabel("flash intensity")
        axes[1, i].set_ylabel("latency (ms)")
        axes[1, i].legend(fontsize=8)
    fig.suptitle("exp002: amplitudes & latencies vs intensity (per-synapse)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_intensity.png", dpi=130)

    # fig3: forward-kernel comparison (three return-current hypotheses) + zoom
    ref = results["1x"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, s in enumerate(side_names):
        axes[0].plot(t_ms, ref["phi_mean_nV"][:, i] / 1000.0, alpha=0.5,
                     ls="--", label=f"{s} mean-position")
        axes[0].plot(t_ms, ref["phi_geo_nV"][:, i] / 1000.0, alpha=0.85,
                     label=f"{s} t-bar return")
        axes[0].plot(t_ms, ref["phi_neu_nV"][:, i] / 1000.0, alpha=0.85,
                     label=f"{s} neurite return")
    axes[0].axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("phi (uV)")
    axes[0].set_title("1x flash: synaptic return-current hypotheses")
    axes[0].legend(fontsize=7)

    zoom = (t_ms >= 400) & (t_ms <= 750)
    axes[1].plot(t_ms[zoom], ref["i_photo_pA"][zoom], color="k", alpha=0.6,
                 label="photocurrent (pA)")
    for i, s in enumerate(side_names):
        axes[1].plot(t_ms[zoom], ref["phi_geo_nV"][zoom, i] / 1000.0,
                     alpha=0.85, label=f"{s} phi (uV)")
    axes[1].axvline(FLASH_ON, color="orange", ls=":", label="flash on")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_title("onset zoom: phototransduction delay")
    axes[1].legend(fontsize=8)

    axes[2].hist(np.clip(pair_dist, 0, 2.0), bins=80, color="tab:blue")
    axes[2].set_xlabel("matched t-bar <-> PSD distance (um)")
    axes[2].set_ylabel("pairs")
    axes[2].set_title(f"pair distances (median "
                      f"{pairing_stats['pair_dist_um']['median']} um)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_geometry.png", dpi=130)

    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
