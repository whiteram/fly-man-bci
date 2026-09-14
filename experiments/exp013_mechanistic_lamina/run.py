"""Experiment 013: the mechanistic lamina -- real biophysics replaces the
exp009 base-current proxy, and the return-current discrimination is
retried on top.

Literature constraints (see README):
  L1/2 dark rest -38.4 +- 3.2 mV (Rusanen & Weckstrom 2016), light
  response = graded hyperpolarization ~10-25 mV (Laughlin/Hardie era),
  photoreceptors depolarize to light and INCREASE histamine release.
  Physics: with release increasing yet v_L hyperpolarizing, the
  histamine-gated chloride reversal must sit BELOW the dark potential:
  solving the two-conductance steady state (leak v_K, Cl E_cl) for the
  two known potentials at dark/light release ratio ~4 gives
  E_cl ~ -70 mV, v_K ~ -25 mV -- a CONVENTIONAL chloride reversal plus
  a depolarized LMC leak (the "inverted chloride" folklore applies to
  other insect interneurons, not the L1/2 lamina).

Model: R1-6 and L1/L2 are graded (non-spiking) leaky integrators
(LIF with v_th = inf); R release rate r in [0,1] maps its membrane
potential; GradedSynapsePool carries the histamine-gated Cl conductance
on per-synapse t-bar/PSD geometry (exp002 pairing). g_unit_hist is
auto-calibrated (1-D scan) to place the L1/2 dark rest at -38 mV.

Then: the SAME three return-current geometries and ERG anchors as
exp012 -- does the mechanistic lamina close the off/on anchor and
discriminate the geometry?

Run from repository root:
    python experiments/exp013_mechanistic_lamina/run.py   (~8 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import StaticPairField
from ffbm.simulation import GradedSynapsePool, LIFPopulation

import run as exp005
import importlib.util as _ilu


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp002 = _load("exp002_run",
               ROOT / "experiments" / "exp002_per_synapse_erg" / "run.py")

OUT = Path(__file__).resolve().parent / "outputs"
DT = exp002.DT
T_END = exp002.T_END
FLASH_ON, FLASH_OFF = exp002.FLASH_ON, exp002.FLASH_OFF
LEVEL_PA = 350.0
SEED = 42

# literature-derived lamina constants (README for provenance)
E_CL_MV = -70.0          # histamine-gated chloride reversal
V_K_L_MC_MV = -25.0      # LMC leak reversal (depolarized)
V_L_DARK_TARGET_MV = -38.4
# 2-D sweep: histamine conductance x release contrast. v_R dark ~ -55 mV,
# bright ~ -30 mV; the map (lo, hi) sets the tonic dark release and the
# release ratio (r_bright / r_dark) -- the two levers that traded the
# off-amplitude against the dark clamp in the single runs.
SWEEP_G_UNIT = (1.6, 2.5, 4.0)
SWEEP_RELEASE = ((-66.0, -26.0), (-62.0, -22.0), (-58.0, -18.0))
DARK_WINDOW_MV = (V_L_DARK_TARGET_MV - 3.2, V_L_DARK_TARGET_MV + 3.2)
LIGHT_DEPTH_WINDOW_MV = (10.0, 25.0)

ANCHORS = {
    "on_transient_over_plateau": (1.0, 3.0),
    "off_over_on_transient": (0.3, 1.0),
    "lmc_fraction_of_on_transient": (0.5, 1.0),
    "lmc_fraction_of_plateau": (0.0, 0.5),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    edges_df, r_ids, l_ids, l_type, r_mean, l_mean = exp002.build_circuit()
    ann = exp002.fdata.load_annotations()
    soma_map = exp002.fdata.neuron_positions(ann)
    u_out, _ = exp002.eye_axis_and_soma_depth(r_mean, soma_map,
                                              r_ids.tolist())
    n_r, n_l = len(r_ids), len(l_ids)
    r_pos = np.array([r_mean[b] for b in r_ids])
    _, r_left = exp002.split_sides(r_pos[:, 0])
    r_side = np.where(r_left, "L", "R")
    if u_out[0] > 0:
        u_out = u_out * np.array([-1.0, 1.0, 1.0])
    axis_side = {"L": u_out, "R": u_out * np.array([-1.0, 1.0, 1.0])}
    electrodes = []
    for s in ("L", "R"):
        cloud = r_pos[r_side == s]
        centroid = cloud.mean(axis=0)
        beyond = float(np.max((cloud - centroid) @ axis_side[s]))
        elec = centroid + (beyond + exp002.ELECTRODE_MARGIN_UM) * axis_side[s]
        electrodes.append(elec)
    electrodes = np.array(electrodes)

    sites = exp002.rl_site_points(r_ids, l_ids)
    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float64)
    order = np.lexsort((pre, post))
    pair_pre, pair_post, pair_edge, pair_dist = exp002.match_edge_pairs(
        sites["PreSyn"], sites["PostSyn"], pre[order], post[order],
        weight[order])
    w_sorted = weight[order]

    def per_edge(field):
        return np.stack([
            np.bincount(pair_edge, weights=field.coef[k],
                        minlength=len(pre)) / w_sorted
            for k in range(len(electrodes))])

    pair_field = StaticPairField(pair_pre, pair_post, electrodes,
                                 sigma=exp002.SIGMA)
    edge_coef_tbar = per_edge(pair_field)
    l_out_mean = {}
    degenerate = 0
    for b in l_ids.tolist():
        arr = sites["PreSyn"].get(b)
        if arr is not None and len(arr):
            l_out_mean[b] = arr.mean(axis=0)
        else:
            l_out_mean[b] = l_mean[b]
            degenerate += 1
    pair_src = np.array([l_out_mean[int(b)] for b in post[order]])[pair_edge]
    edge_coef_neurite = per_edge(StaticPairField(
        pair_src, pair_post, electrodes, sigma=exp002.SIGMA))
    mean_field = StaticPairField(
        np.array([r_mean[b] for b in pre[order]]),
        np.array([l_mean[b] for b in post[order]]),
        electrodes, sigma=exp002.SIGMA)
    rhabd_axis = np.array([axis_side[s] for s in r_side])
    photo_field = StaticPairField(r_pos, r_pos + exp002.RHABD_OFFSET_UM
                                  * rhabd_axis, electrodes,
                                  sigma=exp002.SIGMA)

    # row lookups: release rates are indexed by R row; the pool's
    # post_index maps L body ids to rows. All pool inputs are passed in
    # the same lexsorted edge order as the per-edge kernels above, so
    # the pool's internal sort is the identity and pool.y stays aligned
    # with edge_coef / pair_edge.
    r_row = {int(b): i for i, b in enumerate(r_ids)}
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)
    pre_rows_sorted = np.array([r_row[int(b)] for b in pre[order]])

    def run_sim(g_unit, r_lo, r_hi):
        rng = np.random.default_rng(SEED)
        # graded (non-spiking) R and L1/L2: leaky integrators, v_th = inf
        pop_r = LIFPopulation(n_r, DT, tau_m=exp005.R_TAU,
                              t_refrac=1e9, R_m=exp005.RIN, v_th=1e9)
        pop_l = LIFPopulation(n_l, DT, tau_m=exp005.L_TAU,
                              t_refrac=1e9, R_m=exp005.RIN, v_th=1e9,
                              v_rest=V_K_L_MC_MV)
        pool = GradedSynapsePool(pre_rows_sorted, post[order],
                                 w_sorted.astype(np.float32), l_index,
                                 dt=DT, tau_s=5.0, n_post=n_l,
                                 g_unit=g_unit, e_rev=E_CL_MV)
        photo = exp002.Phototransduction(DT)
        n_field = int(T_END / DT) // 2
        t_ms = np.arange(n_field) * 2 * DT
        phi_l = {"tbar": np.zeros((n_field, 2)),
                 "neurite": np.zeros((n_field, 2)),
                 "meanpos": np.zeros((n_field, 2))}
        phi_r = np.zeros((n_field, 2))
        l12 = np.isin(l_type, ("L1", "L2"))
        v_l_trace = np.zeros(n_field)
        v_r_trace = np.zeros(n_field)
        for k in range(int(T_END / DT)):
            t = k * DT
            flash = LEVEL_PA if FLASH_ON <= t < FLASH_OFF else 0.0
            inc = photo.step(flash)
            pop_r.step(exp005.I_R_BASE + inc
                       + rng.normal(0, exp005.NOISE_SD["R"], n_r))
            r_release = np.clip((pop_r.v - r_lo) / (r_hi - r_lo),
                                0.0, 1.0)
            y = pool.step(r_release)
            i_indep, g_tot = pool.to_neuron_drive()
            pop_l.step(i_indep, g_tot)
            if k % 2 == 0:
                j = k // 2
                i_photo = exp005.I_R_BASE + inc
                phi_l["tbar"][j] = edge_coef_tbar @ pool.edge_currents(
                    pop_l.v) * 1e-12
                phi_l["neurite"][j] = edge_coef_neurite \
                    @ pool.edge_currents(pop_l.v) * 1e-12
                phi_l["meanpos"][j] = mean_field.field(
                    pool.edge_currents(pop_l.v))
                phi_r[j] = photo_field.field(np.full(n_r, i_photo))
                v_l_trace[j] = float(pop_l.v[l12].mean())
                v_r_trace[j] = float(pop_r.v.mean())
        return phi_l, phi_r, v_l_trace, v_r_trace

    # ---- ERG anchors per return geometry ----
    def metrics(phi_tot, phi_lmc):
        m = lambda t0, t1: (t_ms >= t0) & (t_ms < t1)
        s, sl = phi_tot[:, 0], phi_lmc[:, 0]
        base = float(np.median(s[m(100, FLASH_ON - 100)]))
        on_win, on_l = s[m(FLASH_ON, FLASH_ON + 250)], float(
            np.median(sl[m(FLASH_ON, FLASH_ON + 250)]))
        plateau, plateau_l = float(np.median(
            s[m(FLASH_ON + 800, FLASH_OFF - 300)])), float(np.median(
                sl[m(FLASH_ON + 800, FLASH_OFF - 300)]))
        off_win, off_l = s[m(FLASH_OFF, FLASH_OFF + 400)], float(np.median(
            sl[m(FLASH_OFF, FLASH_OFF + 400)]))
        on_d = abs(base - float(on_win.min()))
        off_d = abs(base - float(off_win.max()))
        pl_d = abs(base - plateau)
        return {
            "on_transient_over_plateau": round(on_d / (pl_d + 1e-12), 3),
            "off_over_on_transient": round(off_d / (on_d + 1e-12), 3),
            "lmc_fraction_of_on_transient": round(
                abs(on_l - base) / (on_d + 1e-12), 3),
            "lmc_fraction_of_plateau": round(
                abs(plateau_l - base) / (pl_d + 1e-12), 3),
            "amplitude_uv": {"on": round(on_d * 1e6, 2),
                             "plateau": round(pl_d * 1e6, 2),
                             "off": round(off_d * 1e6, 2)}}

    # ---- 2-D sweep: which return geometry passes the off/on anchor
    # across the whole literature-feasible lamina parameter set? ----
    t_ms = np.arange(int(T_END / DT) // 2) * 2 * DT
    rows = []
    for g in SWEEP_G_UNIT:
        for (r_lo, r_hi) in SWEEP_RELEASE:
            phi_l_g, phi_r_g, vl, vr = run_sim(g, r_lo, r_hi)
            v_dark = float(np.median(
                vl[(t_ms >= 100) & (t_ms < FLASH_ON)]))
            v_light = float(np.median(
                vl[(t_ms >= FLASH_ON + 800) & (t_ms < FLASH_OFF - 300)]))
            depth = v_dark - v_light
            feasible = (DARK_WINDOW_MV[0] <= v_dark <= DARK_WINDOW_MV[1]
                        and LIGHT_DEPTH_WINDOW_MV[0] <= depth
                        <= LIGHT_DEPTH_WINDOW_MV[1])
            anchor = {}
            for v in ("tbar", "neurite", "meanpos"):
                mm = metrics(phi_l_g[v] + phi_r_g, phi_l_g[v])
                anchor[v] = (ANCHORS["off_over_on_transient"][0]
                             <= mm["off_over_on_transient"]
                             <= ANCHORS["off_over_on_transient"][1])
            row = {"g_unit": g, "r_map": [r_lo, r_hi],
                   "v_dark": round(v_dark, 1), "depth_mv": round(depth, 1),
                   "feasible": feasible,
                   "off_on_pass": anchor,
                   "vl": vl, "phi_l": phi_l_g, "phi_r": phi_r_g, "vr": vr}
            rows.append(row)
            print(f"g={g:4.1f} r=({r_lo:.0f},{r_hi:.0f}) "
                  f"dark {v_dark:6.1f} depth {depth:5.1f} "
                  f"{'FEASIBLE' if feasible else '        '} "
                  f"off/on pass {anchor}", flush=True)

    feas = [r for r in rows if r["feasible"]]
    print("feasible lamina configs: %d/%d" % (len(feas), len(rows)))
    verdict = {}
    for v in ("tbar", "neurite", "meanpos"):
        n_pass = sum(1 for r in feas if r["off_on_pass"][v])
        verdict[v] = {"n_pass_of_feasible": n_pass, "n_feasible": len(feas)}
        print(f"  {v:8s}: off/on anchor in {n_pass}/{len(feas)} "
              f"feasible configs")
    # keep a feasible representative (closest to the literature targets)
    # for the full anchor table
    best = min(feas, key=lambda r: abs(r["v_dark"] - V_L_DARK_TARGET_MV)
               + abs(r["depth_mv"] - 15.0)) if feas else rows[0]
    best_g = best["g_unit"]
    phi_l = best["phi_l"]
    phi_r = best["phi_r"]
    v_l_trace = best["vl"]
    v_r_trace = best["vr"]
    summary_extra = {
        "sweep": [{k: r[k] for k in ("g_unit", "r_map", "v_dark",
                                     "depth_mv", "feasible",
                                     "off_on_pass")} for r in rows],
        "sweep_verdict": verdict,
        "chosen": {"g_unit": best["g_unit"], "r_map": best["r_map"],
                   "v_dark": best["v_dark"], "depth_mv": best["depth_mv"]},
    }


    results = {}
    for v in ("tbar", "neurite", "meanpos"):
        tot = phi_l[v] + phi_r
        results[v] = metrics(tot, phi_l[v])
        ok = {a: (ANCHORS[a][0] <= results[v][a] <= ANCHORS[a][1])
              for a in ANCHORS}
        results[v]["anchors_pass"] = ok
        results[v]["n_pass"] = int(sum(ok.values()))
        print(f"{v:8s} {json.dumps(results[v])}")
    verdict = [v for v in results if results[v]["n_pass"] == len(ANCHORS)]

    summary = {
        "config": {"e_cl_mv": E_CL_MV, "v_k_lmc_mv": V_K_L_MC_MV,
                   "g_unit_hist_ns": best_g,
                   "l1_dark_mv": round(v_dark, 1),
                   "l1_light_mv": round(v_light, 1),
                   "r_release_map_mv": best["r_map"],
                   "n_pairs": int(len(pair_dist))},
        "anchors": ANCHORS,
        **summary_extra,
        "variants": results,
        "verdict_all_anchors": verdict,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nverdict (all anchors): {verdict}")
    print(f"written to {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
