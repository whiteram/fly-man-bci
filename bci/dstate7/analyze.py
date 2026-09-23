"""bci/dstate7 analysis: the dose->state map.  Per arm: ALPN rate,
scalp std/mean, MBON in the [10,29] s window; classified against the
state refs.  The 150 pA lock ref comes from dstate1.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "high_lock": (100.0, 0.017), "mid_latch": (83.0, 0.15)}


def classify(alpn, std):
    best, bd = "other", 1e9
    for name, (ra, rs) in REFS.items():
        d = abs(alpn - ra) / 20.0 + abs(std - rs) / 0.05
        if d < bd:
            best, bd = name, d
    return best


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate7: dose -> state (window [10,29] s, seed 62) ==")
    print(f"   {'dose':>6}{'ALPN':>9}{'MBON':>8}{'KC':>7}"
          f"{'std uV':>9}{'mean uV':>9}  state")
    rows = [("25 pA", "d025"), ("50 pA", "d050"), ("100 pA", "d100"),
            ("150 pA*", "REF dstate1"), ("300 pA", "d300"),
            ("1600 pA", "d1600")]
    for label, tag in rows:
        if tag.startswith("REF"):
            print(f"   {label:>6}{19.8:9.2f}{0.01:8.2f}{0.0:7.2f}"
                  f"{0.0145:9.4f}{-1.821:9.3f}  lock (dstate1 ref)")
            continue
        pop = OUT / f"{tag}_pop.npz"
        sp = OUT / f"{tag}_scalp.npy"
        if not pop.exists():
            continue
        z = np.load(pop)
        alpn = float(z["ALPN"][WIN[0]:WIN[1]].mean())
        mbon = float(z["MBON"][WIN[0]:WIN[1]].mean())
        kc = float(z["Kenyon_Cell"][WIN[0]:WIN[1]].mean())
        e = np.load(sp)
        e = e.mean(axis=1) if e.ndim == 2 else e
        seg = e[WIN[0]:WIN[1]]
        std = float(seg.std()) * 1e6 * 1.7
        mean = float(seg.mean()) * 1e6 * 1.7
        st = classify(alpn, std)
        res[tag] = {"alpn": round(alpn, 2), "mbon": round(mbon, 3),
                    "kc": round(kc, 3), "std": round(std, 4),
                    "mean": round(mean, 3), "state": st}
        print(f"   {label:>6}{alpn:9.2f}{mbon:8.2f}{kc:7.2f}"
              f"{std:9.4f}{mean:9.3f}  {st}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate7] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
