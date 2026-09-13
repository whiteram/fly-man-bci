"""exp014 phase 2 -- T4/T5 -> VS wiring: does the lobula plate boost the
scalp signal?

Adds the VS tangential cells (spiking proxy; 18 left-lobe cells, direct
T4/T5->VS dataset edges) on top of the full mechanism stack, runs a
dark + opposite-drift protocol, and decomposes the 4-sphere scalp
potential into the original 5-layer cascade vs the new VS group.
VS cells have centimetre-scale dendritic spans at x400 -- the hypothesis
is that their synaptic currents form the strongest dipole in the network.

Also validates direction preference: VS receives 2/3 of its T4/T5 input
from the d subtypes, so its rate should differ between opposite drift
directions.

Run from repository root:
    python experiments/exp014_circuit_extension/run.py   (~8 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm import pipeline as fp
from ffbm import vizprep as vp
from ffbm.forward import FourSpherePairField
from ffbm.simulation import ColoredCurrentNoise, ExponentialSynapses, LIFPopulation

import run as exp005
from circuit import build_with_lop

OUT = Path(__file__).resolve().parent / "outputs"

SCALE = 400.0
R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SIGMAS = (0.33, 1.79, 0.013, 0.33)
T_DARK, T_DRIFT_A, T_END = 1500.0, 3000.0, 4500.0
SEED = 42
CONTRAST, I_LUM, DRIFT_SPEED, LAM_UM = 2.0, 150.0, 0.1, 22.0
G_UNIT_VS_GRID = (0.0005, 0.001, 0.002)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = build_with_lop()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids, l_ids = circuit["r_ids"], circuit["l_ids"]
    mid_ids, t45_ids = circuit["mid_ids"], circuit["t45_ids"]
    vs_ids = circuit["vs_ids"]
    e_vs = circuit["e_vs"]
    n_r, n_l = len(r_ids), len(l_ids)
    n_mid, n_t45, n_vs = len(mid_ids), len(t45_ids), len(vs_ids)
    mid_type = circuit["mid_type"]

    r_pos = np.array([pp[b] for b in r_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    mid_pos = np.array([pp[b] for b in mid_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    vs_pos = np.array([qq[b] for b in vs_ids])
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos, vs_pos]).mean(axis=0)
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)

    # drift axis (hex regression, as the viz export)
    from ffbm import data as fdata
    ann = fdata.load_annotations()
    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]
    a1 = np.array([1.0, 0.0, 0.0]) - u_eye * u_eye[0]
    a1 /= np.linalg.norm(a1)
    b1 = np.cross(u_eye, a1)
    mi1_ids = mid_ids[mid_type == "Mi1"]
    mi1_pos = np.array([pp[b] for b in mi1_ids])

    def hex_dir(hs):
        h = np.array([hs.get(b, np.nan) for b in mi1_ids])
        ok = np.isfinite(h)
        X = np.stack([mi1_pos[ok] @ a1, mi1_pos[ok] @ b1], axis=1)
        coef, *_ = np.linalg.lstsq(X, h[ok], rcond=None)
        d = coef[0] * a1 + coef[1] * b1
        return d / np.linalg.norm(d)

    e_ds = (np.cos(np.radians(-135.0)) * hex_dir(hex1)
            + np.sin(np.radians(-135.0)) * hex_dir(hex2))
    e_ds /= np.linalg.norm(e_ds)
    x_r = r_pos @ e_ds
    rng_s = np.random.default_rng(SEED)

    def one_over_f(n):
        x = rng_s.standard_normal(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        y = np.fft.irfft(np.fft.rfft(x) / f ** 0.5, n)
        return y / y.std()

    n_tex = 8192
    span = (x_r.max() - x_r.min()) + DRIFT_SPEED * T_END + 4 * LAM_UM
    texture = one_over_f(n_tex)
    tex_dx = span / n_tex

    def tex_at(x):
        idx = np.clip(((x - (x_r.min() - 2 * LAM_UM)) / tex_dx)
                      .astype(int), 0, n_tex - 1)
        return texture[idx]

    def lum(t):
        if t < T_DARK:
            return 0.0
        s = (t - T_DARK) if t < T_DRIFT_A else (2 * (T_DRIFT_A - T_DARK)
                                                - (t - T_DRIFT_A))
        # per-R texture sample (keep the spatial profile!)
        return I_LUM * CONTRAST * tex_at(x_r - DRIFT_SPEED * s)

    # ---- kernels at 3 scalp electrodes (pole band + eye axis) ----
    def pairs_of(e_sub, kind_pre="pre", kind_post="post"):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        pr = np.array([pp[b] for b in pre[order]])
        po = np.array([qq[b] for b in post[order]])
        return pr, po

    e_rl = circuit["e_rl"].copy()
    e_rl["weight"] = e_rl["weight"] * e_rl["sign"]
    e_lm = circuit["e_lm"].copy()
    e_lm["weight"] = e_lm["weight"] * e_lm["sign"]
    e_mt = circuit["e_mt"]
    group_pairs = {"RL": pairs_of(e_rl), "LM": pairs_of(e_lm)}
    for mt in exp005.MID_TYPES:
        group_pairs[f"MT_{mt}"] = pairs_of(e_mt[mt])
    group_pairs["VS"] = pairs_of(e_vs)
    rhabd = r_pos + 23.5 * u_eye

    scaled = {name: (center + (pr - center) * SCALE,
                     center + (po - center) * SCALE)
              for name, (pr, po) in {**group_pairs,
                                     "PHOTO": (r_pos, rhabd)}.items()}
    all_pts = np.vstack([p for pr, po in scaled.values()
                         for p in (pr, po)]) - center
    shift = vp.occipital_shift(all_pts, u_eye, R_BRAIN)
    dirs17 = vp.scalp_electrode_dirs(17, u_eye)
    ang = np.degrees(np.arccos(np.clip(dirs17 @ u_eye, -1, 1)))
    hot = np.argsort(ang)[-2:]
    elec_dirs = np.vstack([dirs17[hot], dirs17[[0]]])
    elec_names = [f"{ang[i]:.0f}deg" for i in hot] + ["0deg"]
    elecs = center + 0.985 * R_SCALP * elec_dirs
    coef = {}
    for name, (pr_s, po_s) in scaled.items():
        coef[name] = FourSpherePairField(
            pr_s + shift, po_s + shift, elecs, center=center,
            r1=R_BRAIN, r2=R_CSF, r3=R_SKULL, r4=R_SCALP,
            sigma1=SIGMAS[0], sigma2=SIGMAS[1], sigma3=SIGMAS[2],
            sigma4=SIGMAS[3]).coef
    print(f"kernels built for electrodes {elec_names}")

    # ---- simulation: 4 base populations via pipeline stack + VS ----
    def run(g_unit_vs):
        cal = dict(fp.CAL)
        rng = np.random.default_rng(SEED)
        st = fp.build_stack(circuit, cal, rng)
        pops, syn, noises = st["pops"], st["syn"], st["noises"]
        vs_index = np.full(int(vs_ids.max()) + 1, -1, dtype=np.int64)
        vs_index[vs_ids] = np.arange(n_vs)
        vs_row = {int(b): i for i, b in enumerate(vs_ids)}
        t45_row = {int(b): i for i, b in enumerate(t45_ids)}
        t45_to_vs = np.full(int(t45_ids.max()) + 1, -1, dtype=np.int64)
        t45_to_vs[t45_ids] = np.arange(n_t45)
        pre = e_vs["body_pre"].to_numpy(np.int64)
        post = e_vs["body_post"].to_numpy(np.int64)
        d = np.array([np.linalg.norm(pp[a] - qq[b])
                      for a, b in zip(pre, post)])
        syn_vs = ExponentialSynapses(
            pre, post, e_vs["weight"].to_numpy(np.float32), vs_index,
            dt=fp.DT_MS, gain=1.0, tau_s=10.0, n_post=n_vs,
            sign=e_vs["sign"].to_numpy(np.float32),
            delay_ms=1.0 + d / 300.0, conductance=True,
            g_unit=g_unit_vs, e_rev_exc=0.0, e_rev_inh=-80.0)
        pop_vs = LIFPopulation(n_vs, fp.DT_MS, tau_m=10.0, t_refrac=2.0,
                               R_m=cal["RIN_GOHM"])
        noise_vs = ColoredCurrentNoise(n_vs, fp.DT_MS, rng, tau_n=8.0,
                                       sigma=60.0)
        photo = fp.PhotoCascade(n_r, fp.DT_MS)
        v_t45_of_edge = (syn[f"MT_{mt}"].post_local for mt in ())  # unused
        n_field = int(T_END / fp.DT_MS) // 2
        phi_all = np.zeros((n_field, len(elec_names)))
        phi_vs = np.zeros((n_field, len(elec_names)))
        rate_vs = np.zeros(n_field)

        def y_of(name, i_photo):
            if name == "PHOTO":
                return i_photo
            if name.startswith("MT_"):
                return syn[name].edge_currents(pops["T45"].v)
            if name == "VS":
                return syn_vs.edge_currents(pop_vs.v)
            return syn[name].y

        for k in range(int(T_END / fp.DT_MS)):
            t = k * fp.DT_MS
            inc_f = photo.step(lum(t))
            sp_r = pops["R"].step(cal["I_R_BASE"] + inc_f
                                  + noises["R"].step())
            sp_l = pops["L"].step(st["l_base"] + syn["RL"].to_neuron_current()
                                  + noises["L"].step())
            sp_mid = pops["MID"].step(st["mid_base"]
                                      + syn["LM"].to_neuron_current()
                                      + noises["MID"].step())
            i_t45 = st["t45_base"] + noises["T45"].step()
            g_t45 = np.zeros(n_t45)
            for mt in cal["MID_TAU_S"]:
                di, dg = syn[f"MT_{mt}"].to_neuron_drive()
                i_t45 += di
                g_t45 += dg
            sp_t45 = pops["T45"].step(i_t45, g_t45)
            di_vs, dg_vs = syn_vs.to_neuron_drive()
            sp_vs = pop_vs.step(di_vs + noise_vs.step(), dg_vs)
            syn["RL"].step(r_ids[sp_r])
            syn["LM"].step(l_ids[sp_l])
            spiked_mid = mid_ids[sp_mid]
            for mt in cal["MID_TAU_S"]:
                syn[f"MT_{mt}"].step(spiked_mid)
            syn_vs.step(t45_ids[sp_t45])
            if k % 2 == 0:
                j = k // 2
                i_photo = cal["I_R_BASE"] + inc_f
                acc = np.zeros(len(elec_names))
                acc_vs = np.zeros(len(elec_names))
                for name in list(group_pairs) + ["PHOTO"]:
                    c = coef[name] @ y_of(name, i_photo)
                    acc += c
                    if name == "VS":
                        acc_vs = c
                phi_all[j] = acc * 1e-12
                phi_vs[j] = acc_vs * 1e-12
                rate_vs[j] = sp_vs.sum() * 1000.0 / n_vs
        return phi_all, phi_vs, rate_vs

    # pick g_unit_vs so the VS dark rate lands in 1-15 Hz
    best = None
    for g in G_UNIT_VS_GRID:
        _, _, rv = run(g)
        dark = float(rv[:int(T_DARK)].mean())
        print(f"g_unit_vs={g:.4f} -> VS dark {dark:.1f} Hz")
        if best is None or abs(dark - 6.0) < best[1]:
            best = (g, abs(dark - 6.0))
    g_unit_vs = best[0]
    print(f"chosen g_unit_vs = {g_unit_vs}")

    phi_all, phi_vs, rate_vs = run(g_unit_vs)
    dark = float(rate_vs[:int(T_DARK)].mean())
    da = float(rate_vs[int(T_DARK):int(T_DRIFT_A)].mean())
    db = float(rate_vs[int(T_DRIFT_A):].mean())
    w_a = slice(int(T_DARK), int(T_DRIFT_A))
    w_b = slice(int(T_DRIFT_A), int(T_END))
    amp_all = np.vstack([phi_all[w_a].std(axis=0),
                         phi_all[w_b].std(axis=0)])
    amp_vs = np.vstack([phi_vs[w_a].std(axis=0), phi_vs[w_b].std(axis=0)])
    summary = {
        "config": {"n_vs": int(n_vs), "g_unit_vs_ns": g_unit_vs,
                   "n_vs_syn": int(e_vs["weight"].sum())},
        "vs_rates_hz": {"dark": round(dark, 2), "driftA": round(da, 2),
                        "driftB": round(db, 2),
                        "asym": round(abs(da - db) / (da + db + 1e-9), 3)},
        "electrodes": elec_names,
        "scalp_uv": {
            "all": [[round(float(x) * 1e6, 4) for x in row]
                    for row in amp_all],
            "vs_only": [[round(float(x) * 1e6, 4) for x in row]
                        for row in amp_vs],
        },
    }
    print(json.dumps(summary, indent=1))
    (OUT / "phase2_vs.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {OUT / 'phase2_vs.json'}")


if __name__ == "__main__":
    main()
