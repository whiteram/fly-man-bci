"""bci/dstate14 analysis: lock pre [10,29] s (formation check) vs
post [35,55] s (post-drop) -- dissolved to dark or hysteretic lock?
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PRE, POST = (10000, 29000), (35000, 55000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145)}


def classify(alpn, std):
    return min(REFS, key=lambda k: (abs(alpn - REFS[k][0]) / 20.0
                                    + abs(std - REFS[k][1]) / 0.05))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate14: gain-drop hysteresis on the established lock ==")
    print("   formation window 0.0016-0.0023 abs; drops at t=30 s")
    for tag, sched in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        row = {}
        for lab, (a, b) in (("pre", PRE), ("post", POST)):
            alpn = float(z["ALPN"][a:b].mean())
            std = float(e[a:b].std()) * 1e6 * 1.7
            row[lab] = {"alpn": round(alpn, 2), "std": round(std, 4),
                        "state": classify(alpn, std)}
        res[tag] = {"sched": sched, **row}
        print(f"   {tag:>7} ({sched or 'hold'}): "
              f"pre {row['pre']['state']} (ALPN {row['pre']['alpn']}, "
              f"std {row['pre']['std']}) -> "
              f"post {row['post']['state']} "
              f"(ALPN {row['post']['alpn']}, std {row['post']['std']})")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate14] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
