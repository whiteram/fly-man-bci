"""bci/kcdrv3 analysis: does the SAME direct DAN injection lift KC in
the plateau that failed in naive?  KC/DAN/MBON increments over the
plateau baseline [6,9.5] s vs the US window [10.1,11.9] s; the ctl
arm provides the plateau's own fluctuation floor.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
BASE = (11000, 14500)
US = (15100, 16900)
CLASSES = ["Kenyon_Cell", "MBON", "ALPN", "DAN"]


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
            u = float(z[k][US[0]:US[1]].mean())
            row[k] = {"base": round(b, 3), "us": round(u, 3),
                      "inc": round(u - b, 3)}
        res[tag] = row
        print(f"-- {tag} --  baseline [11,14.5] s vs US [15.1,16.9] s")
        for k in CLASSES:
            r = row[k]
            print(f"   {k:>12}  base {r['base']:8.3f}  us {r['us']:8.3f}  "
                  f"inc {r['inc']:+8.3f}")
    print("\n== gate ladder (KC increment in US window) ==")
    for tag in ("lock_us", "ml_ctl", "ml_us"):
        if tag in res:
            r = res[tag]["Kenyon_Cell"]
            print(f"   {tag:>10}: KC base {r['base']:6.3f} -> "
                  f"inc {r['inc']:+.3f} Hz")
    if "lock_us" in res and "ml_us" in res and "ml_ctl" in res:
        dl = res["lock_us"]["Kenyon_Cell"]["inc"]
        dm = res["ml_us"]["Kenyon_Cell"]["inc"]
        dc = res["ml_ctl"]["Kenyon_Cell"]["inc"]
        net = dm - dc
        print("[kcdrv3] verdict: " + (
            f"gate OPENS with depolarization (lock {dl:+.2f}, "
            f"mid-latch net {net:+.2f})"
            if net > 0.5 else
            f"gate CLOSED at every reproducible state (lock {dl:+.2f}, "
            f"mid-latch net {net:+.2f}) -- kcdrv1's DAN->KC attribution "
            "is REVISED: correlated, not confirmed"))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[kcdrv3] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
