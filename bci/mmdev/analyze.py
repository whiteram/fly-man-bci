"""bci/mmdev analysis: repetition suppression, deviance, recovery.

Per event: net response = rms(pulse window) - rms(pre-pulse 200 ms).
Reports, per condition (std_on / std_off):
  - the A1..A5 suppression curve
  - deviance ratio devB / A5  (MMN analog: deviant > adapted standard)
  - recovery pulse A6 vs A1

Usage:
    python bci/mmdev/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    ev = meta["events_ms"]
    labels = meta["labels"]
    for cond in meta["conditions"]:
        curves = []
        for r in range(meta["runs"]):
            p = OUT / f"{cond}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            net = []
            for (a, b) in ev:
                base = np.sqrt((phi[a - 200:a] ** 2).mean())
                net.append(np.sqrt((phi[a:b] ** 2).mean()) - base)
            curves.append(net)
        m = np.array(curves).mean(0)
        sd = np.array(curves).std(0)
        print(f"== {cond} ==")
        print("  " + "  ".join(f"{l} {v:+.4f}+-{s:.4f}"
                               for l, v, s in zip(labels, m, sd)))
        a1, a5, dev = m[0], m[4], m[5]
        print(f"  suppression A5/A1: {a5 / max(a1, 1e-9):.2f} | "
              f"deviance devB/A5: {dev / max(abs(a5), 1e-9):.2f} | "
              f"recovery recA/A1: {m[6] / max(a1, 1e-9):.2f}")


if __name__ == "__main__":
    main()
