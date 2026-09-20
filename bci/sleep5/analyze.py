"""bci/sleep5 analysis: R(t) = p2/p1 transient strength vs delay.

Per trace: p1 peak = max 1-s scalp rms in [1, 6] s; p2 peak = max in
[D, D+5] s; same windows for the MBON rate.  R(D) = p2/p1 for both
readouts, compared against the single-tau prediction
d(t) = 1 - exp(-(t - t_coll)/tau) with t_coll = 8 s.  A 2-parameter
grid fit (a, tau2) of the double-exponential recovery
R(t) = a*(1-exp(-(t-tc)/tau)) + (1-a)*(1-exp(-(t-tc)/tau2))
quantifies the second slow component sleep4 inferred.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
T_COLL_S = 8.0


def scalp_rms(p):
    e = np.load(p)
    e = e.mean(axis=1) if e.ndim == 2 else e
    return np.array([np.sqrt((e[a:a + 1000] ** 2).mean())
                     for a in range(0, len(e) - 999, 1000)]) * 1e6 * 1.7


def win_peak(rms, a, b):
    return float(rms[a:b].max())


def win_mean(x, a, b):
    return float(x[a:b].mean())


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    tau_s = meta["dose"]["tau_ms"] / 1000.0
    pts = []
    for tag, d_ms in sorted(meta["delays_ms"].items()):
        f = OUT / f"{tag}_scalp.npy"
        if not f.exists():
            continue
        d_s = int(d_ms // 1000)
        rms = scalp_rms(f)
        p1 = win_peak(rms, 1, 6)
        p2 = win_peak(rms, d_s, d_s + 5)
        rec = {"delay_s": d_s, "p1_peak_uv": round(p1, 2),
               "p2_peak_uv": round(p2, 2), "R_scalp": round(p2 / p1, 3)}
        pop = OUT / f"{tag}_pop.npz"
        if pop.exists():
            z = np.load(pop)
            mb = z["MBON"]
            m1, m2 = win_mean(mb, 1000, 6000), win_mean(mb, d_s * 1000,
                                                        (d_s + 5) * 1000)
            rec["p1_mbon_hz"] = round(m1, 1)
            rec["p2_mbon_hz"] = round(m2, 1)
            rec["R_mbon"] = round(m2 / max(m1, 1e-9), 3)
        # single-tau prediction, recovery clock from collapse
        rec["single_tau_pred"] = round(
            1.0 - np.exp(-(d_s - T_COLL_S) / tau_s), 3) \
            if d_s > T_COLL_S else 0.0
        pts.append(rec)
        print(f"D={d_s:3d}s  p1 {p1:.2f}  p2 {p2:.2f} uV  "
              f"R_scalp {rec['R_scalp']:.2f}  "
              f"R_mbon {rec.get('R_mbon', '-')}  "
              f"pred {rec['single_tau_pred']:.2f}")
    # 2-param grid fit of the double-exponential on R_scalp
    if len(pts) >= 4:
        D = np.array([p["delay_s"] for p in pts], float)
        R = np.array([p["R_scalp"] for p in pts], float)
        tc = D - T_COLL_S
        best = None
        for a in np.arange(0.0, 1.01, 0.05):
            for tau2 in np.concatenate([np.arange(5, 105, 5),
                                        [1e9]]):
                pred = a * (1 - np.exp(-tc / tau_s)) + \
                    (1 - a) * (1 - np.exp(-tc / tau2))
                sse = float(((pred - R) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, a, tau2)
        sse, a, tau2 = best
        fit = {"a_fast": round(a, 2),
               "tau2_slow_s": None if tau2 > 1e8 else round(float(tau2), 1),
               "sse": round(sse, 4)}
        print(f"double-exp fit: R = {a:.2f}*(1-exp(-t/{tau_s:g}s)) + "
              f"{1 - a:.2f}*(1-exp(-t/{fit['tau2_slow_s']}s))  "
              f"sse {sse:.4f}")
        res = {"points": pts, "fit": fit, "tau_fast_s": tau_s,
               "t_coll_s": T_COLL_S}
    else:
        res = {"points": pts, "tau_fast_s": tau_s, "t_coll_s": T_COLL_S}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[sleep5] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
