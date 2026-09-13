"""Experiment 012: constrain the return-current placement with ERG features.

The absolute amplitude of every extracellular potential in this project
depends on WHERE the return current of each synapse is placed:
  A  per-synapse t-bar return   (source at the presynaptic t-bar,
                                 sink at the matched PSD; dipoles ~0.2 um)
  B  neurite return             (sink at the PSD, source at the L cell's
                                 distal PreSyn cloud; dipoles ~10-30 um)
  C  per-cell mean positions    (source at R's site centroid, sink at
                                 L's; what the viz/exp005+ pipeline uses)
exp002 measured the kernel norms (A ~ 0.004x, B ~ 1.7x of C) but ran under
the pre-exp009 wiring. This experiment re-runs the flash ERG under the
CURRENT model (histamine-inhibitory R->L, LMC dark-depolarized base drive,
calibrated gain) and asks which variants are consistent with the classic
Drosophila ERG:

  anchor 1  on-transient / plateau amplitude ratio ~ 1-3 (the corneal-
            negative transient is comparable to or larger than the
            maintained negative plateau at moderate intensity)
  anchor 2  off-transient / on-transient ~ 0.3-1
  anchor 3  the on-transient is largely LMC-driven while the plateau
            contains a large receptor (photoreceptor-current) component
Classic-intracellular-record anchors, approximate by construction; see
outputs/summary.json "anchors" for the exact ranges used.

Run from repository root:
    python experiments/exp012_return_constraint/run.py    (~8 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005
import importlib.util as _ilu


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp002 = _load("exp002_run",
               ROOT / "experiments" / "exp002_per_synapse_erg" / "run.py")
exp009 = _load("exp009_run",
               ROOT / "experiments" / "exp009_lamina_polarity" / "run.py")
exp003 = _load("exp003_run",
               ROOT / "experiments" / "exp003_asymmetric_off" / "run.py")

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp002.DT
T_END = exp002.T_END
FLASH_ON, FLASH_OFF = exp002.FLASH_ON, exp002.FLASH_OFF
LEVEL_PA = 350.0
SEED = 42
# exp012b: rerun with asymmetric phototransduction (tau_fall = 2.5 ms,
# exp003's validated Off-transient mechanism) -- round 1 showed the
# off/on anchor fails for every variant under the symmetric cascade
TAU_FALLS = (10.0, 2.5)

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
        d = np.linalg.norm(cloud - elec, axis=1)
        if d.min() < exp002.MIN_ELEC_DIST_UM:
            elec += (exp002.MIN_ELEC_DIST_UM - d.min()) * axis_side[s]
        electrodes.append(elec)
    electrodes = np.array(electrodes)
    print(f"electrodes at corneal side of both eyes "
          f"(n_R = {n_r:,}, n_L = {n_l:,})")

    sites = exp002.rl_site_points(r_ids, l_ids)
    pre = edges_df["body_pre"].to_numpy(np.int64)
    post = edges_df["body_post"].to_numpy(np.int64)
    weight = edges_df["weight"].to_numpy(np.float64)
    order = np.lexsort((pre, post))
    n_edges = len(pre)

    pair_pre, pair_post, pair_edge, pair_dist = exp002.match_edge_pairs(
        sites["PreSyn"], sites["PostSyn"], pre[order], post[order],
        weight[order])
    w_sorted = weight[order]

    def per_edge(field):
        return np.stack([
            np.bincount(pair_edge, weights=field.coef[k],
                        minlength=n_edges) / w_sorted
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
    neurite_field = StaticPairField(pair_src, pair_post, electrodes,
                                    sigma=exp002.SIGMA)
    edge_coef_neurite = per_edge(neurite_field)

    mean_field = StaticPairField(
        np.array([r_mean[b] for b in pre[order]]),
        np.array([l_mean[b] for b in post[order]]),
        electrodes, sigma=exp002.SIGMA)

    rhabd_axis = np.array([axis_side[s] for s in r_side])
    rhabd_pos = r_pos + exp002.RHABD_OFFSET_UM * rhabd_axis
    photo_field = StaticPairField(r_pos, rhabd_pos, electrodes,
                                  sigma=exp002.SIGMA)

    # ---- simulation under exp009 signs: histamine R->L (all -1), LMC
    # dark-depolarized base drive, calibrated gain ----
    signed_w = (-weight[order]).astype(np.float32)     # histamine: -1
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)

    n_steps = int(T_END / DT)
    n_field = n_steps // 2
    t_ms = np.arange(n_field) * 2 * DT
    variants = {"tbar": edge_coef_tbar, "neurite": edge_coef_neurite}

    def run_sim(tau_fall):
        rng = np.random.default_rng(SEED)
        pop_r = LIFPopulation(n_r, DT, tau_m=exp005.R_TAU,
                              t_refrac=exp005.R_REF, R_m=exp005.RIN)
        pop_l = LIFPopulation(n_l, DT, tau_m=exp005.L_TAU,
                              t_refrac=exp005.L_REF, R_m=exp005.RIN)
        syn = ExponentialSynapses(pre[order], post[order], signed_w,
                                  l_index, dt=DT, gain=exp009.GAIN_RL,
                                  tau_s=exp009.TAU_RL, n_post=n_l)
        photo = exp003.AsymPhototransduction(DT, 10.0, tau_fall)
        l_base = np.full(n_l, exp009.I_L_BASE)
        phi_l = {v: np.zeros((n_field, 2)) for v in
                 list(variants) + ["meanpos"]}
        phi_r = np.zeros((n_field, 2))
        for k in range(n_steps):
            t = k * DT
            flash = LEVEL_PA if FLASH_ON <= t < FLASH_OFF else 0.0
            inc = photo.step(flash)
            sp_r = pop_r.step(exp005.I_R_BASE + inc
                              + rng.normal(0, exp005.NOISE_SD["R"], n_r))
            sp_l = pop_l.step(l_base + syn.to_neuron_current()
                              + rng.normal(0, exp005.NOISE_SD["L"], n_l))
            y = syn.step(r_ids[sp_r])
            if k % 2 == 0:
                j = k // 2
                i_photo = exp005.I_R_BASE + inc
                for v, coef in variants.items():
                    phi_l[v][j] = coef @ y * 1e-12
                phi_l["meanpos"][j] = mean_field.field(y)
                phi_r[j] = photo_field.field(np.full(n_r, i_photo))
        return phi_l, phi_r

    print(f"simulating {T_END / 1000:.1f} s flash x "
          f"{len(TAU_FALLS)} photo conditions ...")
    sims = {f"tau_fall={tf:g}": run_sim(tf) for tf in TAU_FALLS}

    # ---- metrics vs anchors ----
    def metrics(phi_tot, phi_lmc):
        """Eye-L channel ratios; transient depths relative to baseline."""
        m = lambda t0, t1: (t_ms >= t0) & (t_ms < t1)
        s = phi_tot[:, 0]
        sl = phi_lmc[:, 0]
        base = float(np.median(s[m(100, FLASH_ON - 100)]))
        on_win = s[m(FLASH_ON, FLASH_ON + 250)]
        on_l = float(np.median(sl[m(FLASH_ON, FLASH_ON + 250)]))
        plateau = float(np.median(s[m(FLASH_ON + 800, FLASH_OFF - 300)]))
        plateau_l = float(np.median(sl[m(FLASH_ON + 800, FLASH_OFF - 300)]))
        off_win = s[m(FLASH_OFF, FLASH_OFF + 400)]
        off_l = float(np.median(sl[m(FLASH_OFF, FLASH_OFF + 400)]))
        on_depth = abs(base - float(on_win.min()))
        off_depth = abs(base - float(off_win.max()))
        plateau_depth = abs(base - plateau)
        lmc_on = abs(on_l - base) / (on_depth + 1e-12)
        lmc_plateau = abs(plateau_l - base) / (plateau_depth + 1e-12)
        return {
            "on_transient_over_plateau": round(on_depth /
                                               (plateau_depth + 1e-12), 3),
            "off_over_on_transient": round(off_depth / (on_depth + 1e-12),
                                           3),
            "lmc_fraction_of_on_transient": round(lmc_on, 3),
            "lmc_fraction_of_plateau": round(lmc_plateau, 3),
            "amplitude_uv": {
                "on_transient": round(on_depth * 1e6, 2),
                "plateau": round(plateau_depth * 1e6, 2),
                "off_transient": round(off_depth * 1e6, 2)},
        }

    results = {}
    for cond, (phi_l, phi_r) in sims.items():
        results[cond] = {}
        for v in ("tbar", "neurite", "meanpos"):
            tot = phi_l[v] + phi_r
            results[cond][v] = metrics(tot, phi_l[v])
            ok = {a: (ANCHORS[a][0] <= results[cond][v][a] <= ANCHORS[a][1])
                  for a in ANCHORS}
            results[cond][v]["anchors_pass"] = ok
            results[cond][v]["n_pass"] = int(sum(ok.values()))
            print(f"{cond} {v:8s} {json.dumps(results[cond][v], indent=None)}")
        verdict = [v for v in results[cond]
                   if results[cond][v]["n_pass"] == len(ANCHORS)]
        results[cond]["verdict_all_anchors"] = verdict

    summary = {
        "config": {"level_pa": LEVEL_PA,
                   "flash_ms": [FLASH_ON, FLASH_OFF],
                   "sign": "R->L histamine -1 (exp009), LMC base "
                           f"{exp009.I_L_BASE} pA, gain {exp009.GAIN_RL}",
                   "tau_falls_ms": list(TAU_FALLS),
                   "n_pairs": int(len(pair_dist)),
                   "pair_dist_um_median": round(float(np.median(pair_dist)),
                                                3),
                   "neurite_degenerate": degenerate},
        "anchors": ANCHORS,
        "conditions": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for cond in results:
        print(f"\nverdict {cond} (passes all anchors): "
              f"{results[cond]['verdict_all_anchors']}")
    print(f"written to {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
