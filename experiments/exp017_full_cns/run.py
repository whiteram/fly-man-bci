"""exp017: full-CNS assembly -- working point, stability, scalp verdict.

Arms:
  A (reference, from exp016_summary.json): bilateral visual cascade x202
  B (here): + ol_rest + central_brain + vnc (all regions on)

Working point: coordinate descent on (I_OLR, I_CEN, I_VNC) per
G_UNIT_CX in {2, 4, 8} mNS... (values in nS: 0.002/0.004/0.008) for
dark rates in the 1-10 Hz window, then a 6 s stability run (no
explosion > 30 Hz, no silence). Verdict: flash-window scalp std at 3
electrodes vs arm A; VNC groups carry forward=False (simulated, no
scalp kernels).

Run from repository root:
    python experiments/exp017_full_cns/run.py    (~40 min)
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffbm import pipeline as fp
from ffbm import regions as freg
from ffbm.forward import FourSpherePairField
from ffbm.vizprep import fit_scale_shift

OUT = Path(__file__).resolve().parent / "outputs"
T_DARK, T_END = 2000.0, 4000.0
LEVEL_PA = 350.0
SEEDS = (42, 7)
SIGMAS = (0.33, 1.79, 0.013, 0.33)
X1P7 = 1.7
ELEC_DEG = [0.0, 90.0, 147.0]
G_GRID = (0.002, 0.004, 0.008)
I_GRID = (30.0, 60.0, 120.0, 240.0)
REGIONS = ("OLR", "CEN", "VNC")
I_KEYS = {"OLR": "I_OLR_BASE", "CEN": "I_CEN_BASE", "VNC": "I_VNC_BASE"}
RATE_LO, RATE_HI = 1.0, 10.0

_spec = importlib.util.spec_from_file_location(
    "exp015_run", ROOT / "experiments" / "exp015_bilateral" / "run.py")
exp015 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp015)
R_BRAIN, R_CSF, R_SKULL, R_SCALP = (
    exp015.R_BRAIN, exp015.R_CSF, exp015.R_SKULL, exp015.R_SCALP)


def kernel_stack_full(circuit, elec_deg):
    """Kernels for core pairs + forward extra groups (VNC excluded)."""
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_pos = np.array([pp[b] for b in circuit["r_ids"]])
    t45_pos = np.array([pp[b] for b in circuit["t45_ids"]])
    l_pos = np.array([qq[b] for b in circuit["l_ids"]])
    mid_pos = np.array([pp[b] for b in circuit["mid_ids"]])
    extra_pts = [np.array([pp[b] for b in s["ids"]])
                 for s in circuit["extra_pops"].values()]
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos] + extra_pts
                       ).mean(axis=0)
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
    for g, spec in circuit["extra_edges"].items():
        if spec.get("forward", True):
            group_pairs[g] = pairs(spec["table"])
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
    n_pairs = sum(len(v[0]) for v in group_pairs.values())
    return {"coefs": coefs, "scale": scale, "n_pairs": n_pairs,
            "shift_mm": np.linalg.norm(shift) / 1000.0,
            "r_after_um": r_after}


def measure(circuit, cal, ker, seed, t_dark=T_DARK, t_end=T_END):
    n = {k: len(s["ids"]) for k, s in circuit["extra_pops"].items()}
    is_t4 = np.array([str(s).startswith("T4")
                      for s in circuit["t45_type"]])
    is_t5 = np.array([str(s).startswith("T5")
                      for s in circuit["t45_type"]])
    n_out = int(t_end / fp.DT_MS) // 2
    cnt = {k: np.zeros(n_out) for k in ("MID", "T4", "T5", *n)}
    phi = (np.zeros((n_out, ker["coefs"]["RL"].shape[0]))
           if ker is not None else None)

    def record(j, k, t, st, inc_f, sp):
        syn, pops = st["syn"], st["pops"]
        cnt["MID"][j] = sp["MID"].sum() * 1000.0 / st["n_mid"]
        cnt["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        cnt["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()
        for k2 in n:
            cnt[k2][j] = sp[k2].sum() * 1000.0 / n[k2]
        if ker is None:
            return

        def y_of(name):
            if name == "PHOTO":
                return cal["I_R_BASE"] + inc_f
            if name == "RL":
                return syn[name].edge_currents(pops["L"].v)
            if name == "LM":
                return syn[name].edge_currents(pops["MID"].v)
            post = {g: s["post"] for g, s in
                    circuit["extra_edges"].items()}.get(name, "T45")
            return syn[name].edge_currents(pops[post].v)

        for name, c in ker["coefs"].items():
            phi[j] += c @ y_of(name)

    fp.simulate(circuit, cal,
                lambda t: LEVEL_PA if t_dark <= t < t_end else 0.0,
                seed=seed, t_end_ms=t_end, on_sample=record)
    phi = phi * X1P7 * 1e-12

    def w(t0, t1):
        m = slice(int(t0), int(t1))
        return {k: float(cnt[k][m].mean()) for k in cnt}

    fl = slice(int(t_dark + 300.0), int(t_end))
    return {"dark": w(500.0, t_dark), "flash": w(t_dark + 300.0, t_end),
            "phi_flash_std_uv": phi[fl].std(axis=0) * 1e6}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("assembling full-CNS circuit (all regions) ...", flush=True)
    circuit, active = freg.build_circuit({r: True for r in freg.DEFAULT_REGIONS})
    print(f"regions active: {', '.join(active)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    ref = json.loads((ROOT / "experiments" / "exp016_vpn_central"
                      / "outputs" / "exp016_summary.json").read_text())

    cal = dict(fp.CAL)

    # ---- working point: coordinate descent per g ----
    best = None
    for g in G_GRID:
        cal_g = dict(cal)
        cal_g["G_UNIT_CX"] = g
        chosen = {}
        for reg in REGIONS:
            pick = None
            for iv in I_GRID:
                c = dict(cal_g)
                c[I_KEYS[reg]] = iv
                # freeze already-chosen regions
                for r2, v2 in chosen.items():
                    c[I_KEYS[r2]] = v2
                r = measure(circuit, c, None, seed=42,
                            t_dark=1500.0, t_end=1500.0)
                rate = r["dark"][reg]
                print(f"  g={g * 1000:.1f} mNS {reg} I={iv:.0f} -> "
                      f"{rate:.2f} Hz", flush=True)
                if RATE_LO <= rate <= RATE_HI and pick is None:
                    pick = iv
            chosen[reg] = pick if pick is not None else I_GRID[-1]
        # stability: 6 s dark at the chosen point
        c = dict(cal_g)
        for reg in REGIONS:
            c[I_KEYS[reg]] = chosen[reg]
        r = measure(circuit, c, None, seed=42,
                    t_dark=6000.0, t_end=6000.0)
        stable = all(0.0 < r["dark"][reg] < 30.0 for reg in REGIONS)
        print(f"g={g * 1000:.1f} mNS chosen {chosen} stable={stable} "
              f"dark={ {k: round(v, 1) for k, v in r['dark'].items()} }",
              flush=True)
        if stable:
            best = (g, dict(chosen))
            break          # largest g that is stable... grid ascends;
                           # prefer stronger coupling while stable
    if best is None:
        best = (G_GRID[0], {reg: I_GRID[0] for reg in REGIONS})
        print("WARNING: no stable point in grid; falling back to "
              "smallest coupling")
    g, chosen = best
    cal2 = dict(cal)
    cal2["G_UNIT_CX"] = g
    for reg in REGIONS:
        cal2[I_KEYS[reg]] = chosen[reg]

    print("kernels (full CNS, forward groups) ...", flush=True)
    ker = kernel_stack_full(circuit, ELEC_DEG)
    print(f"  scale x{ker['scale']:.0f}, {ker['n_pairs']:,} pairs, "
          f"shift {ker['shift_mm']:.1f} mm "
          f"({time.time() - t0:.0f} s)", flush=True)

    print("simulating full CNS (seeds", SEEDS, ") ...", flush=True)
    runs = [measure(circuit, cal2, ker, s) for s in SEEDS]
    amp = np.mean([r["phi_flash_std_uv"] for r in runs], axis=0)
    amp_ref = ref["amplitude"]["phi_flash_std_uv_a"]

    summary = {
        "working_point": {"g_unit_cx_ns": g, "i_base": chosen},
        "regions": active,
        "rates_dark": {k: round(float(np.mean(
            [r["dark"][k] for r in runs])), 1)
            for k in ("MID", "T4", "T5", *REGIONS)},
        "rates_flash": {k: round(float(np.mean(
            [r["flash"][k] for r in runs])), 1)
            for k in ("MID", "T4", "T5", *REGIONS)},
        "amplitude": {
            "scale_b_full_cns": round(ker["scale"], 1),
            "scale_a_ref_bilateral": ref["amplitude"]["scale_a_bilateral"],
            "elec_deg": ELEC_DEG,
            "phi_flash_std_uv_a_bilateral": amp_ref,
            "phi_flash_std_uv_b_full_cns": [round(float(x), 4)
                                            for x in amp],
            "ratio_full_over_bilateral": [
                round(float(b / a), 3) if a > 1e-12 else None
                for a, b in zip(amp_ref, amp)],
            "note": "VNC groups simulated but excluded from scalp "
                    "kernels (forward=False); arm A reused from "
                    "exp016_summary.json (same seeds/machinery)",
        },
        "timing_s": round(time.time() - t0, 0),
    }
    (OUT / "exp017_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
