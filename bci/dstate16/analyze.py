"""bci/dstate16 analysis: the gain-axis transition map.  Per arm:
pre [10,29] s (formation check) vs post [40,55] s (post-transition),
classified vs dark/lock/latch/mid-latch/high-lock refs.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PRE, POST = (10000, 29000), (40000, 55000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "mid_latch": (83.0, 0.15), "high_lock": (100.0, 0.017),
        "latch": (390.0, 0.5)}


def classify(mbon, alpn, std):
    best, bd = "other", 1e9
    for name, (rm, ra, rs) in {
            "dark": (0.0, 0.0, 0.005),
            "lock": (0.01, 20.0, 0.0145),
            "mid_latch": (76.0, 83.0, 0.15),
            "high_lock": (12.0, 100.0, 0.017),
            "latch": (390.0, 218.0, 0.5)}.items():
        d = (abs(mbon - rm) / 100.0 + abs(alpn - ra) / 50.0
             + abs(std - rs) / 0.05)
        if d < bd:
            best, bd = name, d
    return best


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate16: gain-axis transitions (drop/rise at 30 s) ==")
    for tag, spec in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        row = {}
        for lab, (a, b) in (("pre", PRE), ("post", POST)):
            mbon = float(z["MBON"][a:b].mean())
            alpn = float(z["ALPN"][a:b].mean())
            std = float(e[a:b].std()) * 1e6 * 1.7
            row[lab] = {"mbon": round(mbon, 1), "alpn": round(alpn, 1),
                        "std": round(std, 4),
                        "state": classify(mbon, alpn, std)}
        res[tag] = {"chem": spec["chem"], "sched": spec["sched"],
                    **row}
        print(f"   {tag:>10} ({spec['chem']} {spec['sched']}): "
              f"pre {row['pre']['state']} "
              f"(MBON {row['pre']['mbon']}, ALPN {row['pre']['alpn']}) "
              f"-> post {row['post']['state']} "
              f"(MBON {row['post']['mbon']}, ALPN {row['post']['alpn']})")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate16] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
