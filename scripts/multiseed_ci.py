"""Across-seed variability of the scalp signal (TECHNICAL.md §12 item B6).

Runs the shared pipeline (flicker-only protocol, 4.5 s) for several seeds
with scalp kernels at 3 representative electrodes (occipital pole band
and eye axis) and reports the flicker-band signal amplitude per seed --
mean +- SD across seeds is the honest confidence interval for the
single-seed amplitude quoted in the viz SNR.

Run:  python scripts/multiseed_ci.py   (~20 min, 3 sims)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm import pipeline as fp
from ffbm.forward import FourSpherePairField

import run as exp005

OUT = ROOT / "scripts" / "outputs"

SCALE = 400.0
R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SIGMAS = (0.33, 1.79, 0.013, 0.33)
T_DARK, T_END = 1500.0, 4500.0          # dark + flicker only
SEEDS = (101, 202, 303)                 # export seed is 42
CONTRAST, I_LUM = 2.0, 150.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = exp005.build_circuit()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    r_ids = circuit["r_ids"]
    l_ids = circuit["l_ids"]
    t45_ids = circuit["t45_ids"]
    mid_ids = circuit["mid_ids"]
    n_r = len(r_ids)

    r_pos = np.array([pp[b] for b in r_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    mid_pos = np.array([pp[b] for b in mid_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    center = np.vstack([r_pos, l_pos, mid_pos, t45_pos]).mean(axis=0)
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)

    from ffbm import vizprep as vp
    dirs17 = vp.scalp_electrode_dirs(17, u_eye)
    ang = np.degrees(np.arccos(np.clip(dirs17 @ u_eye, -1, 1)))
    # pole band (3 channels around the max angle) + eye axis channel
    hot = np.argsort(ang)[-3:]
    elec_dirs = np.vstack([dirs17[hot], dirs17[[0]]])
    elec_names = [f"{ang[i]:.0f}deg" for i in hot] + ["0deg"]

    e_rl = circuit["e_rl"].copy()
    e_rl["weight"] = e_rl["weight"] * e_rl["sign"]
    e_lm = circuit["e_lm"].copy()
    e_lm["weight"] = e_lm["weight"] * e_lm["sign"]
    e_mt = circuit["e_mt"]

    def pairs(e_sub):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        order = np.lexsort((pre, post))
        return (np.array([pp[b] for b in pre[order]]),
                np.array([qq[b] for b in post[order]]))

    group_pairs = {"RL": pairs(e_rl), "LM": pairs(e_lm)}
    for mt in exp005.MID_TYPES:
        group_pairs[f"MT_{mt}"] = pairs(e_mt[mt])
    rhabd = r_pos + 23.5 * u_eye

    scaled = {name: (center + (pr - center) * SCALE,
                     center + (po - center) * SCALE)
              for name, (pr, po) in {**group_pairs,
                                     "PHOTO": (r_pos, rhabd)}.items()}
    all_pts = np.vstack([p for pr, po in scaled.values()
                         for p in (pr, po)]) - center
    shift = vp.occipital_shift(all_pts, u_eye, R_BRAIN)
    elecs = center + 0.985 * R_SCALP * elec_dirs
    coef = {}
    for name, (pr_s, po_s) in scaled.items():
        coef[name] = FourSpherePairField(
            pr_s + shift, po_s + shift,
            elecs, center=center, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL,
            r4=R_SCALP, sigma1=SIGMAS[0], sigma2=SIGMAS[1],
            sigma3=SIGMAS[2], sigma4=SIGMAS[3]).coef
    print(f"kernels built for electrodes: {elec_names}")

    rng_s = np.random.default_rng(7)
    flicker = (lambda x: x / x.std())(np.fft.irfft(
        np.fft.rfft(rng_s.standard_normal(int(T_END)))
        / np.maximum(np.fft.rfftfreq(int(T_END)), 1.0) ** 0.5, int(T_END)))

    n_field = int(T_END / fp.DT_MS) // 2
    rows = []
    for seed in SEEDS:
        phi = np.zeros((n_field, len(elec_names)))

        def record(j, k, t, st, inc_f, sp, phi=phi):
            syn, pops = st["syn"], st["pops"]
            i_photo = fp.CAL["I_R_BASE"] + inc_f
            acc = np.zeros(len(elec_names))
            for name in list(group_pairs) + ["PHOTO"]:
                if name == "PHOTO":
                    y = i_photo
                elif name.startswith("MT_"):
                    y = syn[name].edge_currents(pops["T45"].v)
                else:
                    y = syn[name].y
                acc += coef[name] @ y
            phi[j] = acc * 1e-12

        def lum(t):
            if t < T_DARK:
                return 0.0
            # same non-negative clamp semantics as the viz export
            f = flicker[min(int(t), int(T_END) - 1)]
            return I_LUM * max(CONTRAST * f, -0.95)

        print(f"seed {seed}: simulating {T_END / 1000:.1f} s ...",
              flush=True)
        fp.simulate(circuit, dict(fp.CAL), lum, seed, T_END, on_sample=record)
        flick = phi[int(T_DARK):] * 1e6
        amp = flick.std(axis=0)
        rows.append({"seed": seed,
                     "amp_uv": [round(float(a), 3) for a in amp]})
        print(f"   flicker-band amplitude (uV): "
              f"{dict(zip(elec_names, np.round(amp, 3)))}")

    arr = np.array([r["amp_uv"] for r in rows])
    summary = {"electrodes": elec_names, "seeds": rows,
               "mean_uv": [round(float(x), 3) for x in arr.mean(axis=0)],
               "sd_uv": [round(float(x), 3) for x in arr.std(axis=0)],
               "cv_pct": [round(float(s / m * 100), 1)
                          for s, m in zip(arr.std(axis=0), arr.mean(axis=0))]}
    print("\n" + json.dumps(summary, indent=1))
    (OUT / "multiseed_ci.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {OUT / 'multiseed_ci.json'}")


if __name__ == "__main__":
    main()
