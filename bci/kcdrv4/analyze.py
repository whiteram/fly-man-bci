"""bci/kcdrv4 analysis: pairing verdict -- KC response to the weak
drive alone vs the paired weak drive + DAN US.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (600, 2400)
BASE = (0, 400)
CLASSES = ["Kenyon_Cell", "MBON", "DAN"]


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for tag in meta["arms"]:
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        row = {}
        for k in CLASSES:
            b = float(z[k][BASE[0]:BASE[1]].mean())
            u = float(z[k][WIN[0]:WIN[1]].mean())
            row[k] = round(u - b, 3)
        res[tag] = row
        print(f"{tag:>10}: " + "  ".join(
            f"{k} +{row[k]:.3f}" for k in CLASSES))
    if "kc_w400" in res and "kc_p400" in res:
        w = res["kc_w400"]["Kenyon_Cell"]
        p = res["kc_p400"]["Kenyon_Cell"]
        print(f"\n[kcdrv4] KC: weak alone +{w:.3f} vs paired +{p:.3f} "
              f"(DAN contribution {p - w:+.3f} Hz)")
        print("[kcdrv4] verdict: " + (
            "DAN input converts subthreshold -> suprathreshold "
            f"({(p / w if w else float('nan')):.1f}x)" if w
            and p > 1.5 * w and p > 0.5 else
            "permissive gain negligible -- the w6.4 channel adds "
            "nothing even paired" if p - w < 0.3 else "inconclusive"))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[kcdrv4] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
