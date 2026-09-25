"""bci/dstate13: is the SUPER-LATCH (MBON 678-700 Hz) the same
attractor as the classic latch (395) under stronger drive, or a
qualitatively different state?  (pure analysis of dstate9 files)

Test: per-population scaling between the entry-map points.  Homogeneous
scaling (all populations x the same factor) = one attractor, more
drive; a population scaling disproportionately (e.g. DAN via the
MBON->DAN feedback x DAN->MBON w26 trunk) = composition shift.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "dstate9" / "outputs"
WIN = (8000, 28000)
CLASSES = ["Kenyon_Cell", "MBON", "ALPN", "DAN"]
ARMS = ["g10_nostd", "g15_nostd", "g20_nostd",
        "g10_std", "g15_std", "g20_std"]


def main():
    res = {}
    rates = {}
    for tag in ARMS:
        pop = SRC / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        rates[tag] = {k: float(z[k][a:b].mean()) for k in CLASSES}
    print("== dstate13: latch-family composition scaling ==")
    for tag in ARMS:
        if tag not in rates:
            continue
        r = rates[tag]
        print(f"   {tag:>12}: " + "  ".join(
            f"{k} {r[k]:7.1f}" for k in CLASSES))
    if all(t in rates for t in ("g10_nostd", "g20_nostd")):
        print("\n   scaling g10 -> g20 (nostd):")
        for k in CLASSES:
            f20 = rates["g20_nostd"][k] / rates["g10_nostd"][k]
            res[f"scale_{k}"] = round(f20, 2)
            print(f"      {k:>12} x{f20:.2f}")
        fs = [res[f"scale_{k}"] for k in CLASSES]
        print(f"   homogeneous if all factors ~= equal; "
              f"spread {max(fs) / min(fs):.2f}x "
              f"(DAN over-scaling = MBMD feedback x DAN->MBON trunk "
              f"loop)")
    # mid-latch vs latch composition (std g10 vs nostd g10)
    if all(t in rates for t in ("g10_std", "g10_nostd")):
        print("\n   composition ratios MBON:ALPN:KC:DAN")
        for t in ("g10_std", "g10_nostd", "g20_nostd"):
            m = rates[t]["MBON"]
            print(f"      {t:>12}: " + ":".join(
                f"{rates[t][k] / m:.2f}" for k in CLASSES))
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "summary.json").write_text(
        json.dumps({"rates": rates, "scaling": res}, indent=1))
    print(f"\n[dstate13] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
