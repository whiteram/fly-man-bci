"""bci/kcdrv1: what drives the plateau KC +6 Hz?  (pure analysis)

sleep7 localized the plateau response cap to AL-engine pinning + KC-leg
failure but left the KC +6 Hz increment's DRIVER open: central direct
(the ignition pulse drives LAL*@L, everything else is downstream of the
CEN_C recurrence) vs ALPN residual (a +1-2 Hz ALPN increment reaching
KC through the PN->KC leg).  Zero acquisition: sleep5's 1-ms population
traces (Kenyon_Cell/MBON/ALPN/DAN) contain the answer as a CAUSAL
ORDERING question --

  naive p1 pulse (t=0.8-2.8 s): healthy cascade order ALPN->KC->MBON
      ->DAN is the reference.
  plateau p2 pulse (t=D..D+2.3 s): if KC no longer follows ALPN
      (onset order flips / KC leads or coincides while ALPN lags), the
      driver is central; if ALPN's increment still precedes KC's, the
      residual-ALPN route survives.

Per file: baseline [D-8,D-2] s, response [D, D+5] s (p2) or
[0.8, 5.8] s (p1, baseline [0, 0.7] s); 20 ms smoothed increments;
onset = first crossing of 10% of peak (sustained 50 ms); lagged
cross-correlation vs KC over +/-100 ms.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "sleep5" / "outputs"
FILES = {35: "d35s_u0.00015_s616_pop.npz",
         50: "d50s_u0.00015_s616_pop.npz",
         80: "d80s_u0.00015_s616_pop.npz"}
SMOOTH = 20  # ms boxcar
LAG = 100    # ms xcorr span


def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def onset(inc, t0, thr):
    """first index after t0 where inc > thr for 50 consecutive ms."""
    run = 0
    for i in range(int(t0), len(inc)):
        run = run + 1 if inc[i] > thr else 0
        if run >= 50:
            return i - 49
    return -1


def xcorr_vs(ref, tgt, t0, t1, lag):
    """lag (ms) maximizing corr(tgt[t], ref[t+lag]); positive = ref
    leads tgt."""
    a = tgt[t0:t1] - tgt[t0:t1].mean()
    best, bl = 0.0, 0
    for L in range(-lag, lag + 1, 5):
        r = ref[t0 + L:t1 + L]
        r = r - r.mean()
        den = (np.sqrt((a * a).sum()) * np.sqrt((r * r).sum()))
        c = float((a * r).sum() / den) if den > 0 else 0.0
        if c > best:
            best, bl = c, L
    return bl, best


def main():
    res = {}
    for D, fn in FILES.items():
        z = np.load(SRC / fn)
        p2 = D * 1000
        for tag, (t0, t1, b0, b1) in {
                "p1_naive": (800, 5800, 0, 700),
                "p2_plateau": (p2, p2 + 5000, p2 - 8000, p2 - 2000)}.items():
            inc = {}
            for k in z.files:
                x = smooth(z[k], SMOOTH)
                base = x[b0:b1].mean()
                inc[k] = x - base
                win = inc[k][t0:t1]
                peak = float(win.max())
                on = onset(inc[k], t0, 0.1 * peak) if peak > 0 else -1
                inc.setdefault("_stats", {})[k] = {
                    "base_hz": round(float(base), 3),
                    "peak_inc_hz": round(peak, 2),
                    "onset_ms": on - t0 if on >= 0 else None}
            lags = {}
            for k in ("ALPN", "MBON", "DAN"):
                L, c = xcorr_vs(inc[k], inc["Kenyon_Cell"],
                                t0, t0 + 4000, LAG)
                lags[k] = {"best_lag_ms": L, "corr": round(c, 3)}
            res[f"D{D}_{tag}"] = {"stats": inc["_stats"],
                                  "kc_xcorr": lags}
            st = inc["_stats"]
            order = sorted((v["onset_ms"] if v["onset_ms"] is not None
                            else 10**9, k)
                           for k, v in st.items())
            print(f"\n-- D={D} s  {tag} --  baseline/peak/onset:")
            for k, v in st.items():
                print(f"   {k:>12}  base {v['base_hz']:8.3f}  "
                      f"peak +{v['peak_inc_hz']:7.2f}  "
                      f"onset {v['onset_ms']} ms")
            print("   onset order: " +
                  " -> ".join(k for _, k in order))
            print("   KC xcorr (ref leads +): " + "  ".join(
                f"{k} {v['best_lag_ms']:+d}ms(c={v['corr']})"
                for k, v in lags.items()))
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "summary.json").write_text(
        json.dumps(res, indent=1))
    print(f"\n[kcdrv1] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
