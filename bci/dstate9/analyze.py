"""bci/dstate9 analysis: today's sz-entry map -- MBON/ALPN/KC in
[8,28] s per arm, classified against the known levels.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (8000, 28000)
REFS = {"latch": 395.0, "mid_latch": 76.0, "plateau": 3.0, "dark": 0.0}


def classify(mbon):
    return min(REFS, key=lambda k: abs(mbon - REFS[k]))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate9: current-era sz-entry map (window [8,28] s) ==")
    print("   refs MBON Hz: latch 395 / mid-latch 76 / plateau 3 / dark 0")
    print(f"   {'arm':>12}{'MBON':>9}{'ALPN':>9}{'KC':>8}{'DAN':>8}  state")
    for tag, spec in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        mbon = float(z["MBON"][a:b].mean())
        alpn = float(z["ALPN"][a:b].mean())
        kc = float(z["Kenyon_Cell"][a:b].mean())
        dan = float(z["DAN"][a:b].mean())
        st = classify(mbon)
        res[tag] = {"gain_factor": spec["gain_factor"],
                    "std": spec["std"], "mbon": round(mbon, 1),
                    "alpn": round(alpn, 1), "kc": round(kc, 2),
                    "dan": round(dan, 2), "state": st}
        print(f"   {tag:>12}{mbon:9.1f}{alpn:9.1f}{kc:8.2f}{dan:8.2f}  {st}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate9] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
