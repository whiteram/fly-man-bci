"""bci/mmdev2 analysis: identity deviance at matched intensity.

Per event: net response = rms(pulse window) - rms(pre-pulse 200 ms).
Per condition (std_off / std_on_u0.05 / std_on_u0.1):
  - the A1..A5 suppression curve, devB, recA
  - deviance devB/A5 (deviant vs ADAPTED standard, MMN analog)
  - devB/A1 (deviant vs FRESH standard: the fixed DA1/DM2 pathway
    asymmetry baseline, independent of adaptation)
Cross-condition: deviance lift vs std_off.

Usage:
    python bci/mmdev2/analyze.py
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
    dev_ratios = {}
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
        r5 = dev / max(abs(a5), 1e-9)
        r1 = dev / max(abs(a1), 1e-9)
        dev_ratios[cond] = (r5, r1)
        print(f"  suppression A5/A1: {a5 / max(a1, 1e-9):.2f} | "
              f"deviance devB/A5: {r5:.2f} | devB/A1: {r1:.2f} | "
              f"recovery recA/A1: {m[6] / max(a1, 1e-9):.2f}")
    if "std_off" in dev_ratios:
        base5 = dev_ratios["std_off"][0]
        print("== deviance lift vs std_off (devB/A5) ==")
        for cond, (r5, _) in dev_ratios.items():
            print(f"  {cond}: {r5:.2f} (lift {r5 - base5:+.2f})")


if __name__ == "__main__":
    main()
