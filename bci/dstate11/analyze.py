"""bci/dstate11 analysis: late-window central rates per visual
protocol vs the dark reference -- did ANY protocol consolidate a
state (sustained ALPN self-drive, MBON latch)?
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (20000, 28000)
CLASSES = ["Kenyon_Cell", "MBON", "ALPN", "DAN"]


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    print("== dstate11: visual-only state formation (window [20,28] s) ==")
    print("   dark ref: all ~0 Hz")
    for tag in meta["arms"]:
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a, b = WIN
        row = {k: round(float(z[k][a:b].mean()), 3) for k in CLASSES}
        res[tag] = row
        state = ("LATCH" if row["MBON"] > 100 else
                 "state-forming" if row["ALPN"] > 5 or row["MBON"] > 5
                 else "decays (no state)")
        print(f"   {tag:>10}: " + "  ".join(
            f"{k} {row[k]:8.3f}" for k in CLASSES) + f"  -> {state}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[dstate11] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
