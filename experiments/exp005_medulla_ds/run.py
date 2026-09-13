"""Experiment 005: does direction selectivity emerge in T4/T5 from the
connectome + differentiated input kinetics?

Structure Part A (structure.py) found: slow inputs (Mi1/Tm1) sit 0.1-0.7
columns upstream of fast inputs (Tm3/Tm4) on the hex1 axis, with the
differential offset varying by subtype (T5a/T5d strongest, T4d ~ zero) — the
anatomical delay-line blueprint. This script tests the functional
consequence: drive the left-lobe cascade R -> L1/L2/L3 -> Mi/Tm -> T4a-d/
T5a-d with drifting gratings along the hex axis and measure direction
selectivity of every layer.

Two kinetics conditions (only MID->T4/T5 synapses differ):
- uniform:      tau_s = 5 ms for all input types
- differentiated: Mi1=25, Tm3=8, Mi4=10, Mi9=15 (T4 side, slow vs fast per
  Behnia et al. 2014 / Arenz et al. 2017); Tm1=20, Tm2=8, Tm4=10, Tm9=20
  (T5 side)
Prediction (falsifiable): uniform -> DSI ~ 0 everywhere; differentiated ->
DSI > 0 (preferred +u drift) ordered by |dhex(slow) - dhex(fast)| per subtype.

Neurotransmitter signs come from the dataset's sign column (Mi4 GABA, Mi9
glutamate -> inhibitory; the rest ACh -> excitatory).

Run from repository root:
    python experiments/exp005_medulla_ds/run.py
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
DT = 0.5
T_END = 6000.0
GRAT_ON, MEAS_ON = 500.0, 2000.0
I_GRATING = 700.0        # pA peak increment (mean 350, as exp002 1x / exp004)

FREQS_HZ = (2.0, 5.0)
DIRECTIONS = (+1, -1)
KINETICS = {
    "uniform": {mt: 5.0 for mt in ("Mi1", "Tm3", "Mi4", "Mi9",
                                   "Tm1", "Tm2", "Tm4", "Tm9")},
    "differentiated": {"Mi1": 25.0, "Tm3": 8.0, "Mi4": 10.0, "Mi9": 15.0,
                       "Tm1": 20.0, "Tm2": 8.0, "Tm4": 10.0, "Tm9": 20.0},
}

LAMINA_TYPES = ("L1", "L2", "L3")
MID_TYPES = ("Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9")
T45_SUBTYPES = tuple(f"T{x}{c}" for x in "45" for c in "abcd")

# LIF parameters per layer (ms / GOhm)
R_TAU, R_REF = 10.0, 3.0
L_TAU, L_REF = 20.0, 2.0
MID_TAU, MID_REF = 10.0, 2.0
T45_TAU, T45_REF = 10.0, 2.0
RIN = 0.1
NOISE_SD = {"R": 50.0, "L": 20.0, "MID": 20.0, "T45": 20.0}
I_R_BASE = 150.0

SYN_GAIN_RL = 12.0     # pA per synapse, R -> L (as exp001-004)
SYN_GAIN_LM = 8.0      # L -> Mi/Tm
SYN_GAIN_MT = 5.0      # Mi/Tm -> T4/T5
TAU_RL, TAU_LM = 5.0, 5.0

PHOTO_TAU = 10.0
PHOTO_TAU_ADAPT = 800.0
PHOTO_SAG = 0.30

SIGMA = 0.33
ELECTRODE_OFFSET_UM = 50.0
SEED = 42


class PhotoCascadeVector:
    """Per-R two-stage cascade + adaptation (exp002 kinetics, exp004 shape)."""

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


def split_sides(x):
    c = np.percentile(x, [25.0, 75.0]).astype(float)
    for _ in range(20):
        lab = np.abs(x - c[0]) < np.abs(x - c[1])
        c = np.array([x[lab].mean(), x[~lab].mean()])
    return float(c.mean()), lab


def build_circuit():
    """Left-lobe cascade with per-group edge tables and positions."""
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    sites = fdata.load_neuron_sites()
    pre_pos = fdata.site_positions(sites, "PreSyn")
    post_pos = fdata.site_positions(sites, "PostSyn")
    types = nodes.set_index("bodyId")["type"]

    def ids_of(mask_types, pos_map, kind="pre"):
        src = pre_pos if kind == "pre" else post_pos
        return np.array(sorted(b for b in nodes.loc[
            nodes["type"].isin(mask_types), "bodyId"] if b in src),
            dtype=np.int64)

    r_ids = ids_of(("R1-R6",), pre_pos)
    l_ids = ids_of(LAMINA_TYPES, post_pos, "post")
    mid_ids = ids_of(MID_TYPES, pre_pos)
    t45_ids = ids_of(T45_SUBTYPES, pre_pos)

    # one-lobe selection: a single midline from the pooled x 2-means over all
    # four layers, applied to every layer (hex1 is the hex-grid u axis, NOT a
    # lobe tag — the two lobes' u axes are mirrored in physical x, so hex
    # coverage cannot be used to pick a side)
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    all_x = np.concatenate([
        [pre_pos[b][0] for b in r_ids],
        [post_pos[b][0] for b in l_ids],
        [pre_pos[b][0] for b in mid_ids],
        [pre_pos[b][0] for b in t45_ids]])
    midline, _ = split_sides(all_x)

    def left_of(ids, pos_map):
        return np.array([b for b in ids if pos_map[b][0] < midline],
                        dtype=np.int64)

    r_ids = left_of(r_ids, pre_pos)
    l_ids = left_of(l_ids, post_pos)
    mid_ids = left_of(mid_ids, pre_pos)
    t45_ids = left_of(t45_ids, pre_pos)
    r_type = np.array([types.get(b) for b in r_ids])
    l_type = np.array([types.get(b) for b in l_ids])
    mid_type = np.array([types.get(b) for b in mid_ids])
    t45_type = np.array([types.get(b) for b in t45_ids])
    print(f"circuit (left lobe): R {len(r_ids):,} | L {len(l_ids):,} | "
          f"Mi/Tm {len(mid_ids):,} | T4/T5 {len(t45_ids):,}")

    e = edges.assign(tp=edges["body_pre"].map(types),
                     tq=edges["body_post"].map(types))

    def pair_edges(pre_types, post_types):
        s = e[e["tp"].isin(pre_types) & e["tq"].isin(post_types)]
        s = s[s["body_pre"].isin(set(r_ids) | set(l_ids) | set(mid_ids))
              & s["body_post"].isin(set(l_ids) | set(mid_ids) | set(t45_ids))]
        return s[["body_pre", "body_post", "weight", "sign"]].copy()

    e_rl = pair_edges(("R1-R6",), LAMINA_TYPES)
    e_lm = pair_edges(LAMINA_TYPES, MID_TYPES)
    e_mt = {mt: pair_edges((mt,), T45_SUBTYPES) for mt in MID_TYPES}
    n_syn = int(e_rl["weight"].sum() + e_lm["weight"].sum()
                + sum(v["weight"].sum() for v in e_mt.values()))
    print(f"edges: R->L {len(e_rl):,} | L->Mi/Tm {len(e_lm):,} | "
          f"Mi/Tm->T4/T5 {sum(len(v) for v in e_mt.values()):,} "
          f"| synapses {n_syn:,}")

    # grating axis: in-plane direction along which Mi1 hex1 increases
    r_pos = np.array([pre_pos[b] for b in r_ids])
    u_eye = r_pos.mean(axis=0)
    u_eye = u_eye / np.linalg.norm(u_eye)      # rough eye-axis proxy
    mi1_ids = mid_ids[mid_type == "Mi1"]
    mi1_hex_vals = np.array([hex1.get(b, np.nan) for b in mi1_ids])
    mi1_pos = np.array([pre_pos[b] for b in mi1_ids])
    ok = np.isfinite(mi1_hex_vals)
    # in-plane basis: two axes orthogonal to u_eye
    a = np.array([1.0, 0.0, 0.0]) - u_eye * u_eye[0]
    a /= np.linalg.norm(a)
    bvec = np.cross(u_eye, a)
    X = np.stack([mi1_pos[ok] @ a, mi1_pos[ok] @ bvec], axis=1)
    coef, res_, rank, sv = np.linalg.lstsq(X, mi1_hex_vals[ok], rcond=None)
    u_dir = coef[0] * a + coef[1] * bvec
    u_dir /= np.linalg.norm(u_dir)
    pred = X @ coef
    r2 = 1.0 - np.sum((mi1_hex_vals[ok] - pred) ** 2) / np.sum(
        (mi1_hex_vals[ok] - mi1_hex_vals[ok].mean()) ** 2)
    print(f"grating axis from Mi1 hex regression: R^2={r2:.3f}, "
          f"u_dir={u_dir.round(3)}")
    phase0_x = r_pos @ u_dir
    lam = (phase0_x.max() - phase0_x.min()) / 2.0
    phase0 = 2.0 * np.pi * (phase0_x - phase0_x.mean()) / lam
    print(f"R projection extent {phase0_x.max() - phase0_x.min():.1f} um, "
          f"lambda {lam:.1f} um")

    # medulla electrode near the T4/T5 output cloud
    t45_pos = np.array([pre_pos[b] for b in t45_ids])
    elec = t45_pos.mean(axis=0) + ELECTRODE_OFFSET_UM * u_dir

    return dict(r_ids=r_ids, l_ids=l_ids, mid_ids=mid_ids, t45_ids=t45_ids,
                r_type=r_type, l_type=l_type, mid_type=mid_type,
                t45_type=t45_type, e_rl=e_rl, e_lm=e_lm, e_mt=e_mt,
                pre_pos=pre_pos, post_pos=post_pos, phase0=phase0,
                elec=elec, hex_r2=r2)



def lockin(signal, t_ms, freq_hz, t0, t1):
    m = (t_ms >= t0) & (t_ms < t1)
    z = signal[m] * np.exp(-2j * np.pi * freq_hz * t_ms[m] * 1e-3)
    return z.mean() * 2.0


RATE_GROUPS = (["R", "L"] + [f"mid_{mt}" for mt in MID_TYPES]
               + list(T45_SUBTYPES))


def simulate(freq_hz, direction, phase0, r_ids, l_ids, mid_ids,
             syn_rl, syn_lm, syn_mt, type_idx, field_rl, field_lm, field_mt):
    """One grating condition; returns per-group rate traces (Hz) and the
    medulla-electrode potential trace (nV)."""
    n_r, n_l = len(r_ids), len(l_ids)
    n_mid = len(mid_ids)
    n_t45 = len(type_idx["T4a"]) + len(type_idx["T4b"]) \
        + len(type_idx["T4c"]) + len(type_idx["T4d"]) \
        + len(type_idx["T5a"]) + len(type_idx["T5b"]) \
        + len(type_idx["T5c"]) + len(type_idx["T5d"])
    rng = np.random.default_rng(SEED)
    pop_r = LIFPopulation(n_r, DT, tau_m=R_TAU, t_refrac=R_REF, R_m=RIN)
    pop_l = LIFPopulation(n_l, DT, tau_m=L_TAU, t_refrac=L_REF, R_m=RIN)
    pop_mid = LIFPopulation(n_mid, DT, tau_m=MID_TAU, t_refrac=MID_REF,
                            R_m=RIN)
    pop_t45 = LIFPopulation(n_t45, DT, tau_m=T45_TAU, t_refrac=T45_REF,
                            R_m=RIN)
    photo = PhotoCascadeVector(n_r, DT)

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    w_t = 2.0 * np.pi * freq_hz * 1e-3
    counts = {g: np.zeros(n_field) for g in RATE_GROUPS}
    phi_med = np.zeros(n_field)

    for k in range(n_steps):
        t = k * DT
        inc = I_GRATING * (0.5 + 0.5 * np.cos(phase0 - direction * w_t * t)) \
            if t >= GRAT_ON else np.zeros(n_r)
        inc_f = photo.step(inc)
        i_r = I_R_BASE + inc_f + rng.normal(0, NOISE_SD["R"], n_r)
        sp_r = pop_r.step(i_r)
        i_l = syn_rl.to_neuron_current() + rng.normal(0, NOISE_SD["L"], n_l)
        sp_l = pop_l.step(i_l)
        i_mid = syn_lm.to_neuron_current() \
            + rng.normal(0, NOISE_SD["MID"], n_mid)
        sp_mid = pop_mid.step(i_mid)
        i_t45 = sum(s.to_neuron_current() for s in syn_mt.values()) \
            + rng.normal(0, NOISE_SD["T45"], n_t45)
        sp_t45 = pop_t45.step(i_t45)

        syn_rl.step(r_ids[sp_r])
        syn_lm.step(l_ids[sp_l])
        spiked_mid = mid_ids[sp_mid]
        for s in syn_mt.values():
            s.step(spiked_mid)

        if k % 2 == 0:
            j = k // 2
            counts["R"][j] = sp_r.sum()
            counts["L"][j] = sp_l.sum()
            for g, m in type_idx.items():
                counts[g][j] = (sp_mid[m].sum() if g.startswith("mid_")
                                else sp_t45[m].sum())
            phi_med[j] = (field_rl.coef[0] @ syn_rl.y
                          + field_lm.coef[0] @ syn_lm.y
                          + sum(field_mt[mt].coef[0] @ syn_mt[mt].y
                                for mt in MID_TYPES)) * 1e-12

    sizes = {g: (n_r if g == "R" else n_l if g == "L"
                 else n_mid if g.startswith("mid_") else n_t45)
             for g in RATE_GROUPS}
    rates = {g: counts[g] * 1000.0 / sizes[g] for g in RATE_GROUPS}
    return {"rates": rates, "phi_nV": phi_med * 1e9}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = build_circuit()
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
    type_idx = {**{f"mid_{mt}": np.where(mid_type == mt)[0]
                   for mt in MID_TYPES},
                **{st: np.where(t45_type == st)[0] for st in T45_SUBTYPES}}

    elec = circuit["elec"][None, :]
    t_ms = np.arange(int(T_END / DT) // 2) * 2 * DT

    results = {}
    for kin_name, kin in KINETICS.items():
        # synapse groups (fresh state per condition) + medulla field kernels
        pre = circuit["e_rl"]["body_pre"].to_numpy(np.int64)
        post = circuit["e_rl"]["body_post"].to_numpy(np.int64)
        syn_rl = ExponentialSynapses(
            pre, post, circuit["e_rl"]["weight"].to_numpy(np.float32),
            l_index, dt=DT, gain=SYN_GAIN_RL, tau_s=TAU_RL, n_post=n_l)
        order = np.lexsort((pre, post))
        field_rl = StaticPairField(np.array([pp[b] for b in pre[order]]),
                                   np.array([qq[b] for b in post[order]]),
                                   elec, sigma=SIGMA)

        pre = circuit["e_lm"]["body_pre"].to_numpy(np.int64)
        post = circuit["e_lm"]["body_post"].to_numpy(np.int64)
        syn_lm = ExponentialSynapses(
            pre, post, circuit["e_lm"]["weight"].to_numpy(np.float32),
            mid_index, dt=DT, gain=SYN_GAIN_LM, tau_s=TAU_LM, n_post=n_mid)
        order = np.lexsort((pre, post))
        field_lm = StaticPairField(np.array([pp[b] for b in pre[order]]),
                                   np.array([qq[b] for b in post[order]]),
                                   elec, sigma=SIGMA)

        syn_mt, field_mt = {}, {}
        for mt in MID_TYPES:
            pre = circuit["e_mt"][mt]["body_pre"].to_numpy(np.int64)
            post = circuit["e_mt"][mt]["body_post"].to_numpy(np.int64)
            weight = (circuit["e_mt"][mt]["weight"].to_numpy(np.float32)
                      * circuit["e_mt"][mt]["sign"].to_numpy(np.float32))
            syn_mt[mt] = ExponentialSynapses(pre, post, weight, t45_index,
                                             dt=DT, gain=SYN_GAIN_MT,
                                             tau_s=kin[mt], n_post=n_t45)
            order = np.lexsort((pre, post))
            field_mt[mt] = StaticPairField(
                np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]),
                elec, sigma=SIGMA)

        for f in FREQS_HZ:
            for d in DIRECTIONS:
                print(f"simulating {kin_name} f={f:g} Hz dir={d:+d} ...")
                results[(kin_name, f, d)] = simulate(
                    f, d, circuit["phase0"], r_ids, l_ids, mid_ids,
                    syn_rl, syn_lm, syn_mt, type_idx, field_rl, field_lm,
                    field_mt)
                res = results[(kin_name, f, d)]
                a5 = abs(lockin(res["rates"]["T5a"], t_ms, f, MEAS_ON, T_END))
                ap = abs(lockin(res["phi_nV"], t_ms, f, MEAS_ON, T_END))
                print(f"  T5a mod {a5:.2f} Hz | phi mod {ap / 1000:.1f} uV")

    # ---- analysis ----
    def amp(g, cond):
        res = results[cond]
        sig = res["phi_nV"] if g == "phi" else res["rates"][g]
        return abs(lockin(sig, t_ms, cond[1], MEAS_ON, T_END))

    def dsi(g, kin_name, f):
        a_p, a_m = amp(g, (kin_name, f, +1)), amp(g, (kin_name, f, -1))
        return (a_p - a_m) / (a_p + a_m + 1e-12)

    summary = {"config": {
        "dt_ms": DT, "t_end_ms": T_END, "freqs_hz": list(FREQS_HZ),
        "kinetics": KINETICS, "i_grating_pA": I_GRATING,
        "gains_pA": {"RL": SYN_GAIN_RL, "LM": SYN_GAIN_LM, "MT": SYN_GAIN_MT},
        "n_R": n_r, "n_L": n_l, "n_mid": n_mid, "n_t45": n_t45,
        "hex_regression_r2": circuit["hex_r2"],
    }, "mean_rate_hz": {}, "amp_hz": {}, "dsi": {}}

    meas = (t_ms >= MEAS_ON) & (t_ms < T_END)
    for cond, res in results.items():
        tag = f"{cond[0]}/{cond[1]:g}Hz/dir{cond[2]:+d}"
        summary["mean_rate_hz"][tag] = {
            g: round(float(res["rates"][g][meas].mean()), 2)
            for g in RATE_GROUPS}
        summary["amp_hz"][tag] = {
            g: round(float(amp(g, cond)), 4)
            for g in list(RATE_GROUPS) + ["phi"]}
    for kin_name in KINETICS:
        summary["dsi"][kin_name] = {
            f"{f:g}Hz": {g: round(float(dsi(g, kin_name, f)), 4)
                         for g in list(RATE_GROUPS) + ["phi"]}
            for f in FREQS_HZ}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDSI:")
    print(json.dumps(summary["dsi"], indent=1))

    # ---- figures ----
    x = np.arange(len(T45_SUBTYPES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, f in zip(axes, FREQS_HZ):
        for kin_name, style in (("uniform", "o--"), ("differentiated", "o-")):
            ys = [summary["dsi"][kin_name][f"{f:g}Hz"][g]
                  for g in T45_SUBTYPES]
            ax.plot(x, ys, style, label=kin_name)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x, T45_SUBTYPES)
        ax.set_xlabel("cell type")
        ax.set_title(f"{f:g} Hz")
        ax.legend()
    axes[0].set_ylabel("DSI (+ = preferred +u drift)")
    fig.suptitle("exp005: direction selectivity of T4/T5 — structure + "
                 "differentiated kinetics")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dsi.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    zoom = (t_ms >= 3000) & (t_ms < 4000)
    for d, ls in zip(DIRECTIONS, ("-", "--")):
        res = results[("differentiated", 5.0, d)]
        for g, color in (("T5a", "tab:blue"), ("T4d", "tab:red")):
            ax.plot(t_ms[zoom], res["rates"][g][zoom], ls, color=color,
                    alpha=0.85, label=f"{g} dir{d:+d}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("differentiated kinetics, 5 Hz grating (1 s zoom)")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_traces.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
