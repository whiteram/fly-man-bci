"""exp015: bilateral (both-lobe) circuit -- two questions.

Q1 calibration transfer: per-cell wiring is mirror-symmetric between the
lobes, so the round-5 working point (g_hist = 0.16 nS, g_lm = 0.04,
I_MID = 90 pA) must hold UNCHANGED on the bilateral circuit. Check per
side on the innervated L1/2 (mass >= 50) measurement set: dark median
(-38.4 +- 3.2 mV), light depth (10-25 mV), MID/T4/T5 dark-rate windows,
flash T4 > 5 Hz. Two seeds averaged.

Q2 amplitude: how much larger is the scalp signal with both lobes
active? The bilateral cloud (native span 692 um) does NOT fit inside the
brain sphere at x400 (277 mm > 156 mm diameter), so each circuit runs at
its own fit-scale min(400, fit) with the T4/T5-junction extreme pinned
at the occipital pole (0.90 R): left stays ~x400, bilateral drops to
~x200. Dipole moment scales linearly with scale, so the naive prediction
is ratio ~ 2*200/400 ~ 1.0 -- the lobe doubling is almost exactly
cancelled by the necessary scale reduction. Same seeds, same flash
protocol (2 s dark + 2 s 350 pA), three electrodes (0 deg = eye side,
90 deg, 147 deg -- the old best channel, near-occipital).

Run from repository root:
    python experiments/exp015_bilateral/run.py    (~10 min)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffbm import pipeline as fp
from ffbm.forward import FourSpherePairField
from ffbm.vizprep import fit_scale_shift

from circuit import R_BRAIN, R_CSF, R_SKULL, R_SCALP, \
    build_bilateral_circuit, exp005

OUT = Path(__file__).resolve().parent / "outputs"
T_DARK, T_END = 2000.0, 4000.0
LEVEL_PA = 350.0
SEEDS = (42, 7)
TARGETS = {"MID": (5.0, 20.0), "T4": (2.0, 12.0), "T5": (2.0, 15.0)}
SIGMAS = (0.33, 1.79, 0.013, 0.33)
X1P7 = 1.7
SCALE_NOMINAL = 400.0


def kernel_stack(circuit, elec_deg, side_r=None):
    """Pair kernels for electrodes at `elec_deg` from the eye-side anchor.

    Placement: u_occ points from the cloud centroid toward the T4/T5
    junction; the junction extreme is pinned at the occipital pole
    (0.90 R_brain) by fit_scale_shift, which also picks the largest
    scale that then fits inside 0.98 R_brain. The electrode anchor
    (0 deg) is the anti-occipital direction, i.e. the "eye side"."""
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_pos = np.array([pp[b] for b in circuit["r_ids"]])
    t45_pos = np.array([pp[b] for b in circuit["t45_ids"]])
    l_pos = np.array([qq[b] for b in circuit["l_ids"]])
    mid_pos = np.array([pp[b] for b in circuit["mid_ids"]])
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos]).mean(axis=0)
    u_occ = t45_pos.mean(axis=0) - center
    u_occ = u_occ / np.linalg.norm(u_occ)
    u_anchor = -u_occ            # 0 deg = eye side, 180 deg = occipital

    def pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        return (np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]))

    group_pairs = {"RL": pairs(circuit["e_rl"]),
                   "LM": pairs(circuit["e_lm"])}
    for mt in exp005.MID_TYPES:
        group_pairs[f"MT_{mt}"] = pairs(circuit["e_mt"][mt])
    # rhabdome dipole: each R's phototransduction current points toward
    # ITS OWN eye -- mirror the left eye axis across the midline for the
    # right lobe (side_r: True = left lobe). The left-only circuit has
    # no u_eye_left key; its own eye axis is r-mean minus t45-mean.
    u_eye = circuit.get("u_eye_left")
    if u_eye is None:
        u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
        u_eye = u_eye / np.linalg.norm(u_eye)
    if side_r is None:
        rh_dir = np.tile(u_eye, (len(r_pos), 1))
    else:
        xhat = np.array([1.0, 0.0, 0.0])
        u_right = u_eye - 2.0 * float(np.dot(u_eye, xhat)) * xhat
        u_right = u_right / np.linalg.norm(u_right)
        rh_dir = np.where(side_r[:, None], u_eye[None, :],
                          u_right[None, :])
    group_pairs["PHOTO"] = (r_pos, r_pos + 23.5 * rh_dir)

    native = np.vstack([p for pr, po in group_pairs.values()
                        for p in (pr, po)])
    scale, shift, r_after = fit_scale_shift(native, center, u_occ,
                                            R_BRAIN, SCALE_NOMINAL)
    elec_dirs = [np.cos(np.radians(a)) * u_anchor
                 + np.sin(np.radians(a)) * _perp(u_anchor)
                 for a in elec_deg]
    coefs = {name: [] for name in group_pairs}
    for d in elec_dirs:
        elec = center + 0.985 * R_SCALP * d
        for name, (pr, po) in group_pairs.items():
            pr_s = center + (pr - center) * scale + shift
            po_s = center + (po - center) * scale + shift
            coefs[name].append(FourSpherePairField(
                pr_s, po_s, elec[None, :], center=center,
                r1=R_BRAIN, r2=R_CSF, r3=R_SKULL, r4=R_SCALP,
                sigma1=SIGMAS[0], sigma2=SIGMAS[1], sigma3=SIGMAS[2],
                sigma4=SIGMAS[3]).coef[0])
    coefs = {k: np.vstack(v) for k, v in coefs.items()}
    return {"coefs": coefs, "scale": scale,
            "shift_mm": np.linalg.norm(shift) / 1000.0,
            "r_after_um": r_after}


def _perp(u):
    a = np.array([1.0, 0.0, 0.0]) - u * u[0]
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])


def measure(circuit, cal, ker, seed, innerv_side=None):
    """Dark+flash run; innerv_side = (innervated mask, side mask)."""
    n_r = len(circuit["r_ids"])
    n_mid = len(circuit["mid_ids"])
    is_t4 = np.array([str(s).startswith("T4") for s in circuit["t45_type"]])
    is_t5 = np.array([str(s).startswith("T5") for s in circuit["t45_type"]])
    n_out = int(T_END / fp.DT_MS) // 2
    cnt = {k: np.zeros(n_out) for k in ("R", "MID", "T4", "T5")}
    v_l = np.zeros((n_out, len(circuit["l_ids"])))
    phi = np.zeros((n_out, ker["coefs"]["RL"].shape[0]))

    def record(j, k, t, st, inc_f, sp):
        syn, pops = st["syn"], st["pops"]
        cnt["R"][j] = sp["R"].sum() * 1000.0 / n_r
        cnt["MID"][j] = sp["MID"].sum() * 1000.0 / n_mid
        cnt["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        cnt["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()
        v_l[j] = pops["L"].v

        def y_of(name):
            if name == "PHOTO":
                return cal["I_R_BASE"] + inc_f
            if name == "RL":
                return syn[name].edge_currents(pops["L"].v)
            if name == "LM":
                return syn[name].edge_currents(pops["MID"].v)
            return syn[name].edge_currents(pops["T45"].v)

        for name, c in ker["coefs"].items():
            phi[j] += c @ y_of(name)

    fp.simulate(circuit, cal,
                lambda t: LEVEL_PA if T_DARK <= t < T_END else 0.0,
                seed=seed, t_end_ms=T_END, on_sample=record)
    phi = phi * X1P7 * 1e-12

    def w(t0, t1):
        m = slice(int(t0), int(t1))
        return {k: float(cnt[k][m].mean()) for k in cnt}

    res = {"dark": w(500.0, T_DARK), "flash": w(T_DARK + 300.0, T_END),
           "v_dark": np.median(v_l[100:int(T_DARK)], axis=0),
           "v_light": np.median(v_l[int(T_DARK) + 300:], axis=0)}
    if innerv_side is not None:
        innerv, side = innerv_side
        for tag, m in (("left", innerv & side), ("right", innerv & ~side),
                       ("both", innerv)):
            res[f"vd_{tag}"] = float(np.median(res["v_dark"][m]))
            res[f"depth_{tag}"] = res[f"vd_{tag}"] - float(
                np.median(res["v_light"][m]))
    fl = slice(int(T_DARK + 300.0), int(T_END))
    res["phi_flash_std_uv"] = phi[fl].std(axis=0) * 1e6
    res["phi_dark_std_uv"] = phi[100:int(T_DARK)].std(axis=0) * 1e6
    return res


def innervated_mask(circuit):
    l_ids = circuit["l_ids"]
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(len(l_ids))
    e_rl = circuit["e_rl"]
    mass = np.bincount(
        l_index[e_rl["body_post"].to_numpy(np.int64)],
        weights=e_rl["weight"].to_numpy(np.float64),
        minlength=len(l_ids))
    return np.isin(circuit["l_type"], ("L1", "L2")) & (mass >= 50.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("building left-lobe circuit ...", flush=True)
    left = exp005.build_circuit()
    print(f"building bilateral circuit ... ({time.time() - t0:.0f} s)",
          flush=True)
    bi = build_bilateral_circuit()
    print(f"circuits built ({time.time() - t0:.0f} s)", flush=True)

    elec_deg = [0.0, 90.0, 147.0]
    cal = dict(fp.CAL)
    print("kernels (left) ...", flush=True)
    ker_l = kernel_stack(left, elec_deg)
    print(f"  scale x{ker_l['scale']:.0f}, shift {ker_l['shift_mm']:.1f} mm, "
          f"r_after {ker_l['r_after_um'] / 1000:.1f} mm", flush=True)
    print("kernels (bilateral) ...", flush=True)
    ker_b = kernel_stack(bi, elec_deg, side_r=bi["side_r"])
    print(f"  scale x{ker_b['scale']:.0f}, shift {ker_b['shift_mm']:.1f} mm, "
          f"r_after {ker_b['r_after_um'] / 1000:.1f} mm "
          f"(brain R {R_BRAIN / 1000:.0f} mm)", flush=True)

    inn_b = innervated_mask(bi)

    def avg(runs, key, sub=None):
        vals = [r[key] if sub is None else r[key][sub] for r in runs]
        return float(np.mean(vals))

    print("simulating (seeds", SEEDS, ") ...", flush=True)
    runs_l = [measure(left, cal, ker_l, s) for s in SEEDS]
    runs_b = [measure(bi, cal, ker_b, s, innerv_side=(inn_b, bi["side_l"]))
              for s in SEEDS]

    calib = {
        "l12_dark_mv": round(avg(runs_b, "vd_both"), 1),
        "l12_depth_mv": round(avg(runs_b, "depth_both"), 1),
        "l12_dark_left_mv": round(avg(runs_b, "vd_left"), 1),
        "l12_dark_right_mv": round(avg(runs_b, "vd_right"), 1),
        "dark_rates_hz": {k: round(avg(runs_b, "dark", k), 1) for k in
                          ("R", "MID", "T4", "T5")},
        "flash_rates_hz": {k: round(avg(runs_b, "flash", k), 1) for k in
                           ("R", "MID", "T4", "T5")},
    }
    vd, depth = calib["l12_dark_mv"], calib["l12_depth_mv"]
    calib["windows_pass"] = {
        "l12_dark_-38.4+-3.2": -41.6 <= vd <= -35.2,
        "depth_10_25": 10.0 <= depth <= 25.0,
        **{f"{k}_{lo}_{hi}": lo <= calib["dark_rates_hz"][k] <= hi
           for k, (lo, hi) in TARGETS.items()},
        "flashT4_gt5": calib["flash_rates_hz"]["T4"] > 5.0,
    }
    amp_l = np.mean([r["phi_flash_std_uv"] for r in runs_l], axis=0)
    amp_b = np.mean([r["phi_flash_std_uv"] for r in runs_b], axis=0)

    summary = {
        "working_point": "round-5 CAL, unchanged",
        "calibration": calib,
        "amplitude": {
            "note": "bilateral does NOT fit at x400 (native span 692 um "
                    "-> 277 mm); each circuit uses fit_scale_shift "
                    "(max feasible scale; pole-pinned for the elongated "
                    "single lobe, centered for the wider-than-long "
                    "bilateral V)",
            "scale_left": round(ker_l["scale"], 1),
            "scale_bilateral": round(ker_b["scale"], 1),
            "elec_deg": elec_deg,
            "phi_flash_std_uv_left": [round(float(x), 3) for x in amp_l],
            "phi_flash_std_uv_bilateral": [round(float(x), 3) for x in amp_b],
            "ratio_bilateral_left": [
                round(float(b / a), 2) if a > 1e-9 else None
                for a, b in zip(amp_l, amp_b)],
            "best_elec_left_deg": elec_deg[int(np.argmax(amp_l))],
            "best_elec_bilateral_deg": elec_deg[int(np.argmax(amp_b))],
            "best_uv_left": round(float(amp_l.max()), 3),
            "best_uv_bilateral": round(float(amp_b.max()), 3),
            "shift_mm_left": round(ker_l["shift_mm"], 1),
            "shift_mm_bilateral": round(ker_b["shift_mm"], 1),
            "r_after_um_bilateral": round(ker_b["r_after_um"], 0),
        },
        "timing_s": round(time.time() - t0, 0),
    }
    (OUT / "exp015_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
