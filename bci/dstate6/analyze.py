"""bci/dstate6 analysis: high-lock composition + 300 s persistence.

hi_comp: per-class rates in the 20 Hz lock window (pre) vs the high
lock window (post) -- the AL-local ratio (does the whole AL engine
scale ~5x or only ALPN?) and the central share (MBON/DAN/KC vs the
plateau references MBON 395 / KC 74 / DAN 42).
dc300: scalp std + ALPN rate per 90 s block -- decay toward dark
(std ->0.10, ALPN ->0) or persistence at the lock signature
(std 0.0145, ALPN 20).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
AL = ["ALPN", "ALIN", "ALON", "ALLN"]
CEN = ["Kenyon_Cell", "MBON", "DAN"]
PLATEAU = {"Kenyon_Cell": 74.0, "MBON": 395.0, "DAN": 42.0}


def rms_series(e, step=1000):
    return np.array([np.sqrt((e[a:a + step] ** 2).mean())
                     for a in range(0, len(e) - step + 1, step)]) \
        * 1e6 * 1.7


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))

    z = np.load(OUT / "hi_comp_pop.npz")
    pre = {k: float(z[k][10000:29000].mean()) for k in z.files}
    post = {k: float(z[k][31000:50000].mean()) for k in z.files}
    print("== hi_comp: 20 Hz lock -> high lock (seed 62) ==")
    print(f"   {'class':>12}{'pre':>9}{'post':>9}{'ratio':>8}")
    for k in AL + CEN:
        if k in pre:
            r = post[k] / pre[k] if pre[k] > 0.01 else float("nan")
            print(f"   {k:>12}{pre[k]:9.2f}{post[k]:9.2f}{r:8.2f}")
    print("   central share vs plateau: " + "  ".join(
        f"{k} {post.get(k, 0):.1f}/{PLATEAU[k]:.0f} Hz "
        f"({post.get(k, 0) / PLATEAU[k]:.0%})" for k in CEN))

    e = np.load(OUT / "dc300_scalp.npy")
    e = e.mean(axis=1) if e.ndim == 2 else e
    rms = rms_series(e)
    zp = np.load(OUT / "dc300_pop.npz")
    print("\n== dc300: natural-exit test, 90 s blocks ==")
    print(f"   {'block(s)':>12}{'std uV':>9}{'ALPN':>8}{'MBON':>8}")
    for a, b in ((10, 29), (100, 190), (200, 290)):
        alpn = float(zp["ALPN"][a * 1000:b * 1000].mean())
        mbon = float(zp["MBON"][a * 1000:b * 1000].mean())
        print(f"   {f'{a}-{b}':>12}{rms[a:b].std():9.4f}"
              f"{alpn:8.2f}{mbon:8.2f}")
    print("   refs: dark std 0.100 / lock std 0.0145 ALPN 20")
    (OUT / "summary.json").write_text(json.dumps(
        {"hi_comp": {"pre": pre, "post": post},
         "dc300_rms_1s_first120": [round(float(v), 3)
                                   for v in rms[:120:4]]}, indent=1))
    print(f"\n[dstate6] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
