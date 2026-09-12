"""Experiment 001: flash -> R1-R6 + lamina LIF population -> eye-surface potential.

Positions come from aggregated synapse sites (scripts/build_neuron_sites.py):
photoreceptor terminals (PreSyn) and lamina PSDs (PostSyn). The eye-surface axis
is calibrated from the 13 photoreceptors that also have annotated somas.

Run from repository root:
    python experiments/exp001_erg_forward/run.py
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

from ffbm import data as fdata
from ffbm.forward import StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

OUT = Path(__file__).resolve().parent / "outputs"

# ----------------------------- configuration -----------------------------
DT = 0.5            # ms
T_END = 3000.0      # ms
FLASH_ON, FLASH_OFF = 500.0, 1500.0

LAMINA_TYPES = ("L1", "L2", "L3")

# photoreceptor (R) parameters
R_TAU, R_RIN, R_REF = 10.0, 0.1, 3.0        # ms, GOhm, ms
I_R_BASE, I_R_FLASH = 150.0, 350.0          # pA
R_NOISE_SD = 50.0                           # pA

# lamina (L) parameters
L_TAU, L_RIN, L_REF = 20.0, 0.1, 2.0
L_NOISE_SD = 20.0                           # pA

SYN_GAIN = 12.0     # pA per synapse (weight unit)
SYN_TAU = 5.0       # ms

RHABD_OFFSET_UM = 23.5  # terminal -> soma/rhabdomere depth along u_out (calibrated)
ELECTRODE_MARGIN_UM = 20.0  # electrode distance beyond the outermost terminal
MIN_ELEC_DIST_UM = 30.0     # enforce equal minimum source-electrode distance
SIGMA = 0.33            # S/m

SEED = 42

# ----------------------------- geometry -----------------------------
def eye_axis_and_soma_depth(r_pos_map, soma_map, r_ids):
    """u_out (towards the eye surface) and terminal->soma depth, from the few
    photoreceptors that have annotated soma positions."""
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


# ----------------------------- circuit -----------------------------
def build_circuit():
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    sites = fdata.load_neuron_sites()

    r_pos = fdata.site_positions(sites, "PreSyn")   # photoreceptor terminals
    l_pos = fdata.site_positions(sites, "PostSyn")  # lamina PSDs

    types = nodes.set_index("bodyId")["type"]
    r_ids = np.array(sorted(b for b in nodes.loc[nodes["type"] == "R1-R6", "bodyId"]
                            if b in r_pos), dtype=np.int64)
    l_ids = np.array(sorted(b for b in nodes.loc[nodes["type"].isin(LAMINA_TYPES),
                                                 "bodyId"] if b in l_pos),
                     dtype=np.int64)
    l_type = np.array([types.get(b) for b in l_ids])

    e = edges.assign(type_pre=edges["body_pre"].map(types),
                     type_post=edges["body_post"].map(types))
    e = e[(e["type_pre"] == "R1-R6") & (e["type_post"].isin(LAMINA_TYPES))]
    e = e[["body_pre", "body_post", "weight"]].copy()
    e["pos_pre"] = e["body_pre"].map(r_pos)
    e["pos_post"] = e["body_post"].map(l_pos)
    e = e.dropna(subset=["pos_pre", "pos_post"]).reset_index(drop=True)
    print(f"R->lamina edges with site positions: {len(e):,} | "
          f"photoreceptors: {len(r_ids):,} | lamina: {len(l_ids):,} | "
          f"synapses: {int(e['weight'].sum()):,}")
    return e, r_ids, l_ids, l_type, r_pos, l_pos


# ----------------------------- simulation -----------------------------
def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    edges_df, r_ids, l_ids, l_type, r_pos_map, l_pos_map = build_circuit()
    ann = fdata.load_annotations()
    soma_map = fdata.neuron_positions(ann)
    u_out, soma_depth = eye_axis_and_soma_depth(r_pos_map, soma_map, r_ids.tolist())

    n_r, n_l = len(r_ids), len(l_ids)
    r_pos = np.array([r_pos_map[b] for b in r_ids])
    l_pos = np.array([l_pos_map[b] for b in l_ids])
    n_steps = int(T_END / DT)

    # ---- per-side geometry: mirror the calibrated eye axis across the midline ----
    _, r_left = split_sides(r_pos[:, 0])
    midline, l_left = split_sides(l_pos[:, 0])
    side_names = ["L", "R"]
    r_side = np.where(r_left, "L", "R")
    l_side = np.where(l_left, "L", "R")
    if u_out[0] > 0:  # calibration axis must point towards the left cluster
        u_out = u_out * np.array([-1.0, 1.0, 1.0])
    axis_side = {"L": u_out, "R": u_out * np.array([-1.0, 1.0, 1.0])}

    electrodes = []
    for s in side_names:
        cloud = r_pos[r_side == s]
        centroid = cloud.mean(axis=0)
        beyond = float(np.max((cloud - centroid) @ axis_side[s]))
        elec = centroid + (beyond + ELECTRODE_MARGIN_UM) * axis_side[s]
        # normalize minimum electrode-to-source distance across sides (the two
        # reconstructed retinas differ in thickness and cell count)
        d = np.linalg.norm(cloud - elec, axis=1)
        if d.min() < MIN_ELEC_DIST_UM:
            elec += (MIN_ELEC_DIST_UM - d.min()) * axis_side[s]
        electrodes.append(elec)
        print(f"electrode {s}: n_R={len(cloud)}, min dist to terminal = "
              f"{np.linalg.norm(cloud - elec, axis=1).min():.1f} um")
    electrodes = np.array(electrodes)

    # ---- synapses (edge order must match ExponentialSynapses' internal sort) ----
    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float32)
    post_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    post_index[l_ids] = np.arange(n_l)
    syn = ExponentialSynapses(pre, post, weight, post_index,
                              dt=DT, gain=SYN_GAIN, tau_s=SYN_TAU, n_post=n_l)
    order = np.lexsort((pre, post))
    pre_pos = np.array([r_pos_map[b] for b in pre[order]])
    post_pos = np.array([l_pos_map[b] for b in post[order]])

    # ---- forward kernels ----
    syn_field = StaticPairField(pre_pos, post_pos, electrodes, sigma=SIGMA)
    rhabd_axis = np.array([axis_side[s] for s in r_side])
    rhabd_pos = r_pos + RHABD_OFFSET_UM * rhabd_axis  # distal sink per eye
    photo_field = StaticPairField(r_pos, rhabd_pos, electrodes, sigma=SIGMA)

    # ---- populations ----
    pop_r = LIFPopulation(n_r, DT, tau_m=R_TAU, t_refrac=R_REF, R_m=R_RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=L_TAU, t_refrac=L_REF, R_m=L_RIN)

    n_field = int(n_steps // 2)
    phi = np.zeros((n_field, len(side_names)))
    i_photo = np.zeros(n_steps)
    spike_r = np.zeros((n_steps, n_r), dtype=bool)
    spike_l = np.zeros((n_steps, n_l), dtype=bool)
    r_body_row = {b: i for i, b in enumerate(r_ids.tolist())}

    print(f"simulating {n_steps} steps ({T_END} ms), "
          f"{syn.n_edges:,} synapse edges, {n_r + n_l:,} neurons ...")
    for k in range(n_steps):
        t = k * DT
        flash = I_R_FLASH if FLASH_ON <= t < FLASH_OFF else 0.0
        i_r = I_R_BASE + flash + rng.normal(0, R_NOISE_SD, n_r)
        i_photo[k] = I_R_BASE + flash
        sp_r = pop_r.step(i_r)
        i_l = syn.to_neuron_current() + rng.normal(0, L_NOISE_SD, n_l)
        sp_l = pop_l.step(i_l)
        y = syn.step(r_ids[sp_r])
        spike_r[k] = sp_r
        spike_l[k] = sp_l
        if k % 2 == 0:
            phi[k // 2] = syn_field.field(y) + photo_field.field(
                np.full(n_r, i_photo[k]))

    # ----------------------------- analysis -----------------------------
    t_ms = np.arange(n_field) * 2 * DT

    def rate(spikes):
        counts = spikes.reshape(n_field, 2, -1).sum(axis=1)   # 1 ms bins
        kernel = np.ones(20) / 20.0
        return np.apply_along_axis(lambda s: np.convolve(s, kernel, "same"),
                                   1, counts * 1000.0)

    rate_r = rate(spike_r).mean(axis=1)
    rate_l_by_type = {ty: rate(spike_l[:, l_type == ty]).mean(axis=1)
                      for ty in LAMINA_TYPES}

    def w_rate(spikes, t0, t1):
        return float(spikes[int(t0 / DT):int(t1 / DT)].mean() / (DT * 1e-3))

    summary = {
        "config": {
            "dt_ms": DT, "t_end_ms": T_END, "flash_ms": [FLASH_ON, FLASH_OFF],
            "syn_gain_pA": SYN_GAIN, "syn_tau_ms": SYN_TAU,
            "i_r_base_pA": I_R_BASE, "i_r_flash_pA": I_R_FLASH,
            "n_photoreceptors": n_r, "n_lamina": n_l, "n_edges": int(syn.n_edges),
            "n_synapses": int(weight.sum()),
        },
        "rate_r_hz": {
            "baseline": round(w_rate(spike_r, 0, FLASH_ON), 1),
            "flash": round(w_rate(spike_r, FLASH_ON + 100, FLASH_OFF), 1),
            "after": round(w_rate(spike_r, FLASH_OFF + 100, T_END), 1),
        },
        "rate_l_hz": {
            ty: {
                "baseline": round(w_rate(spike_l[:, l_type == ty], 0, FLASH_ON), 1),
                "flash": round(
                    w_rate(spike_l[:, l_type == ty], FLASH_ON + 100, FLASH_OFF), 1),
            }
            for ty in LAMINA_TYPES
        },
        "phi_nV": {
            s: {
                "peak_on": round(float(phi[
                    (t_ms > FLASH_ON) & (t_ms < FLASH_ON + 100), i].min() * 1e9), 1),
                "plateau": round(float(np.median(
                    phi[(t_ms > FLASH_ON + 300) & (t_ms < FLASH_OFF), i]) * 1e9), 1),
                "peak_off": round(float(
                    phi[(t_ms > FLASH_OFF) & (t_ms < FLASH_OFF + 200), i].max() * 1e9),
                    1),
            }
            for i, s in enumerate(side_names)
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ----------------------------- figures -----------------------------
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t_ms, (t_ms >= FLASH_ON) & (t_ms < FLASH_OFF), color="k")
    axes[0].set_ylabel("flash")
    axes[0].set_yticks([0, 1])
    axes[1].plot(t_ms, rate_r, color="tab:purple", label="R1-R6")
    for ty, ls in zip(LAMINA_TYPES, ("-", "--", ":")):
        axes[2].plot(t_ms, rate_l_by_type[ty], ls=ls, label=ty)
    for i, s in enumerate(side_names):
        axes[3].plot(t_ms, phi[:, i] * 1e9, label=f"{s} electrode", alpha=0.85)
    axes[1].set_ylabel("R rate (Hz)")
    axes[2].set_ylabel("lamina rate (Hz)")
    axes[3].set_ylabel("phi (nV)")
    axes[3].set_xlabel("time (ms)")
    for ax in axes:
        ax.axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
        ax.legend(loc="upper right")
    fig.suptitle("exp001: population firing rates")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_rates.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, s in enumerate(side_names):
        ax.plot(t_ms, phi[:, i] * 1e9, label=f"{s} electrode")
    ax.axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12, label="flash")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("eye-surface potential (nV)")
    ax.set_title("simulated ERG-like waveform (forward model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_erg.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
