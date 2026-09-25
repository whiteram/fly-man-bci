"""bci/dstate12 analysis: the (F x dose) consolidation surface."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "high_lock": (100.0, 0.017)}


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
    print("== dstate12: (F x dose) consolidation surface ==")
    print("   refs: F=1.0 150->lock 1600->high lock (dstate7); "
          "F=0.7 150->dark (dstate8)")
    print(f"   {'arm':>12}{'ALPN':>9}{'MBON':>8}{'KC':>7}"
          f"{'std uV':>9}  state")
    for tag, spec in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        alpn = float(z["ALPN"][a:b].mean())
        mbon = float(z["MBON"][a:b].mean())
        kc = float(z["Kenyon_Cell"][a:b].mean())
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        seg = e[a:b]
        std = float(seg.std()) * 1e6 * 1.7
        st = classify(alpn, std)
        res[tag] = {"F": spec["F"], "chem": spec["chem"],
                    "alpn": round(alpn, 2), "mbon": round(mbon, 2),
                    "kc": round(kc, 2), "std": round(std, 4),
                    "state": st}
        print(f"   {tag:>12}{alpn:9.2f}{mbon:8.2f}{kc:7.2f}"
              f"{std:9.4f}  {st}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate12] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
