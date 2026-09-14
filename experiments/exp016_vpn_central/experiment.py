"""exp016 phase 2 -- does the VPN -> central-brain pathway matter for
the scalp signal?

Arms (each a self-consistent thought-experiment instance, junction
pinned at the occipital pole, auto-fit scale):
  A. bilateral cascade alone (exp015 circuit)
  B. bilateral + VPN(top-10 classes) + top central receivers

Working point for VPN/CB: coarse grid on (I_V_BASE, I_C_BASE) for dark
rates in the 1-10 Hz window (2 s dark), everything else = round-5 CAL.

Verdict metric: flash-window scalp std at 3 electrodes (0/90/147 deg),
same seeds; plus VPN/CB rate windows.

Run from repository root:
    python experiments/exp016_vpn_central/experiment.py   (~20 min)
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffbm import pipeline as fp
from ffbm.forward import FourSpherePairField
from ffbm.vizprep import fit_scale_shift

from circuit import R_BRAIN, R_CSF, R_SKULL, R_SCALP, \
    build_bilateral_circuit
from circuit2 import build_vpn_circuit

# reuse exp015's kernel_stack / measure for arm A via a fixed module name
_spec = importlib.util.spec_from_file_location(
    "exp015_run", ROOT / "experiments" / "exp015_bilateral" / "run.py")
exp015 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp015)

OUT = Path(__file__).resolve().parent / "outputs"
T_DARK, T_END = 2000.0, 4000.0
LEVEL_PA = 350.0
SEEDS = (42, 7)
SIGMAS = (0.33, 1.79, 0.013, 0.33)
X1P7 = 1.7
ELEC_DEG = [0.0, 90.0, 147.0]
I_GRID = (30.0, 60.0, 120.0)


def kernel_stack_v(circuit, elec_deg):
    """exp015 kernel_stack + the M2V / V2C groups."""
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_pos = np.array([pp[b] for b in circuit["r_ids"]])
    t45_pos = np.array([pp[b] for b in circuit["t45_ids"]])
    l_pos = np.array([qq[b] for b in circuit["l_ids"]])
    mid_pos = np.array([pp[b] for b in circuit["mid_ids"]])
    vpn_pos = np.array([pp[b] for b in circuit["extra_pops"]["VPN"]["ids"]])
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos, vpn_pos,
                        np.array([qq[b] for b in circuit["extra_pops"]["CB"]["ids"]])
                        ]).mean(axis=0)
    u_occ = t45_pos.mean(axis=0) - center
    u_occ = u_occ / np.linalg.norm(u_occ)
    u_anchor = -u_occ

    def pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        return (np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]))

    group_pairs = {"RL": pairs(circuit["e_rl"]),
                   "LM": pairs(circuit["e_lm"])}
    for mt in exp015.exp005.MID_TYPES:
        group_pairs[f"MT_{mt}"] = pairs(circuit["e_mt"][mt])
    group_pairs["M2V"] = pairs(circuit["e_m2v"])
    group_pairs["V2C"] = pairs(circuit["e_v2c"])
    u_eye = circuit["u_eye_left"]
    xhat = np.array([1.0, 0.0, 0.0])
    u_right = u_eye - 2.0 * float(np.dot(u_eye, xhat)) * xhat
    u_right = u_right / np.linalg.norm(u_right)
    rh_dir = np.where(circuit["side_r"][:, None], u_eye[None, :],
                      u_right[None, :])
    group_pairs["PHOTO"] = (r_pos, r_pos + 23.5 * rh_dir)

    native = np.vstack([p for pr, po in group_pairs.values()
                        for p in (pr, po)])
    scale, shift, r_after = fit_scale_shift(native, center, u_occ,
                                            R_BRAIN)
    elec_dirs = [np.cos(np.radians(a)) * u_anchor
                 + np.sin(np.radians(a)) * exp015._perp(u_anchor)
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


def measure_v(circuit, cal, ker, seed):
    """exp015 measure + VPN/CB rates and M2V/V2C currents."""
    n_r = len(circuit["r_ids"])
    n_mid = len(circuit["mid_ids"])
    n_vpn = len(circuit["extra_pops"]["VPN"]["ids"])
    n_cb = len(circuit["extra_pops"]["CB"]["ids"])
    is_t4 = np.array([str(s).startswith("T4") for s in circuit["t45_type"]])
    is_t5 = np.array([str(s).startswith("T5") for s in circuit["t45_type"]])
    n_out = int(T_END / fp.DT_MS) // 2
    cnt = {k: np.zeros(n_out) for k in
           ("R", "MID", "T4", "T5", "VPN", "CB")}
    phi = np.zeros((n_out, ker["coefs"]["RL"].shape[0]))

    def record(j, k, t, st, inc_f, sp):
        syn, pops = st["syn"], st["pops"]
        cnt["R"][j] = sp["R"].sum() * 1000.0 / n_r
        cnt["MID"][j] = sp["MID"].sum() * 1000.0 / n_mid
        cnt["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        cnt["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()
        cnt["VPN"][j] = sp["VPN"].sum() * 1000.0 / n_vpn
        cnt["CB"][j] = sp["CB"].sum() * 1000.0 / n_cb

        def y_of(name):
            if name == "PHOTO":
                return cal["I_R_BASE"] + inc_f
            if name == "RL":
                return syn[name].edge_currents(pops["L"].v)
            if name == "LM":
                return syn[name].edge_currents(pops["MID"].v)
            if name == "M2V":
                return syn[name].edge_currents(pops["VPN"].v)
            if name == "V2C":
                return syn[name].edge_currents(pops["CB"].v)
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

    fl = slice(int(T_DARK + 300.0), int(T_END))
    return {"dark": w(500.0, T_DARK), "flash": w(T_DARK + 300.0, T_END),
            "phi_flash_std_uv": phi[fl].std(axis=0) * 1e6}


def dark_rates(circuit, cal, seed=42, t_end=1500.0):
    """Short dark run for the (I_V, I_C) working-point grid."""
    n_vpn = len(circuit["extra_pops"]["VPN"]["ids"])
    n_cb = len(circuit["extra_pops"]["CB"]["ids"])
    cnt = {"VPN": 0, "CB": 0}

    def record(j, k, t, st, inc_f, sp):
        cnt["VPN"] += sp["VPN"].sum()
        cnt["CB"] += sp["CB"].sum()

    n_s = int(t_end)
    fp.simulate(circuit, cal, lambda t: 0.0, seed=seed,
                t_end_ms=t_end, on_sample=record)
    return {"VPN": cnt["VPN"] / n_s, "CB": cnt["CB"] / n_s}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("building bilateral circuit ...", flush=True)
    bi = build_bilateral_circuit()
    print(f"building VPN circuit ... ({time.time() - t0:.0f} s)", flush=True)
    vc = build_vpn_circuit(bi)
    print(f"circuits built ({time.time() - t0:.0f} s)", flush=True)

    # ---- working-point grid for VPN/CB ----
    cal = dict(fp.CAL)
    grid = {}
    best = None
    for iv in I_GRID:
        for ic in I_GRID:
            c = dict(cal)
            c["I_V_BASE"], c["I_C_BASE"] = iv, ic
            r = dark_rates(vc, c)
            ok = all(1.0 <= r[k] <= 10.0 for k in ("VPN", "CB"))
            grid[f"{iv:g}x{ic:g}"] = {k: round(v, 2)
                                      for k, v in r.items()} | {"ok": ok}
            print(f"  I_V={iv:.0f} I_C={ic:.0f} -> VPN {r['VPN']:.2f} "
                  f"CB {r['CB']:.2f} Hz {'OK' if ok else ''}", flush=True)
            if ok and best is None:
                best = (iv, ic)
    iv, ic = best if best else (I_GRID[1], I_GRID[1])
    cal2 = dict(cal)
    cal2["I_V_BASE"], cal2["I_C_BASE"] = iv, ic
    print(f"chosen I_V={iv:.0f} I_C={ic:.0f}")

    # ---- kernels ----
    print("kernels (A: bilateral) ...", flush=True)
    ker_a = exp015.kernel_stack(bi, ELEC_DEG, side_r=bi["side_r"])
    print(f"  scale x{ker_a['scale']:.0f}", flush=True)
    print("kernels (B: +VPN/CB) ...", flush=True)
    ker_b = kernel_stack_v(vc, ELEC_DEG)
    print(f"  scale x{ker_b['scale']:.0f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---- runs ----
    print("simulating arm A ...", flush=True)
    runs_a = [exp015.measure(bi, cal, ker_a, s) for s in SEEDS]
    print("simulating arm B ...", flush=True)
    runs_b = [measure_v(vc, cal2, ker_b, s) for s in SEEDS]

    amp_a = np.mean([r["phi_flash_std_uv"] for r in runs_a], axis=0)
    amp_b = np.mean([r["phi_flash_std_uv"] for r in runs_b], axis=0)
    summary = {
        "working_point": {"i_v_base": iv, "i_c_base": ic, "grid": grid},
        "rates_arm_b": {
            "dark": {k: round(float(np.mean([r["dark"][k] for r in runs_b])),
                              1) for k in ("MID", "T4", "T5", "VPN", "CB")},
            "flash": {k: round(float(np.mean(
                [r["flash"][k] for r in runs_b])), 1)
                for k in ("MID", "T4", "T5", "VPN", "CB")}},
        "amplitude": {
            "scale_a_bilateral": round(ker_a["scale"], 1),
            "scale_b_with_vpn": round(ker_b["scale"], 1),
            "elec_deg": ELEC_DEG,
            "phi_flash_std_uv_a": [round(float(x), 4) for x in amp_a],
            "phi_flash_std_uv_b": [round(float(x), 4) for x in amp_b],
            "ratio_b_over_a": [
                round(float(b / a), 3) if a > 1e-12 else None
                for a, b in zip(amp_a, amp_b)],
        },
        "timing_s": round(time.time() - t0, 0),
    }
    (OUT / "exp016_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
