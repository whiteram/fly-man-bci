"""bci/dstate17: WHAT is the latch@0.002 residual?  (pure analysis)

dstate16 found the latch dropped to 0.002 (the lock window's center)
leaves a strong residual (MBON 44.6 / ALPN 98.2) that only 0.001
dissolves.  The classifier called it "high lock", but its MBON share
(0.45 of ALPN) is 4x the dstate5/6 high lock's (0.12).  This arc
localizes the residual's composition against the two AL-instrumented
references:

  dstate16 latch_dn2 post [40,55] s   -- the residual
  dstate5  hi_comp      post [31,50] s -- the high lock (10x-probe)

Readout: per-class rates + MBON:ALPN and KC:ALPN ratios; verdict on
whether the residual is a high-lock variant (AL-dominated) or a
mid-latch-like central state (MBON/KC elevated).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D16 = HERE.parent / "dstate16" / "outputs"
D6 = HERE.parent / "dstate6" / "outputs"
AL = ["ALPN", "ALIN", "ALON", "ALLN"]
CEN = ["Kenyon_Cell", "MBON", "DAN"]
CASES = [
    ("residual (d16 latch_dn2 post)", D16 / "latch_dn2_pop.npz",
     (40000, 55000)),
    ("high lock (d6 hi_comp post)", D6 / "hi_comp_pop.npz",
     (31000, 50000)),
]


def main():
    res = {}
    rows = {}
    for name, path, (a, b) in CASES:
        if not path.exists():
            print(f"[dstate17] missing {path}")
            continue
        z = np.load(path)
        rows[name] = {k: float(z[k][a:b].mean())
                      for k in AL + CEN if k in z.files}
    print("== dstate17: residual composition vs the high lock ==")
    for name, r in rows.items():
        alpn = r["ALPN"]
        print(f"\n-- {name} --")
        for k in AL + CEN:
            rat = r[k] / alpn if alpn else float("nan")
            print(f"   {k:>12} {r[k]:8.2f} Hz   x{rat:5.2f} of ALPN")
        res[name] = {k: round(v, 2) for k, v in r.items()}
        res[name]["ratios"] = {k: round(r[k] / alpn, 3) if alpn else None
                               for k in AL + CEN}
    if len(rows) == 2:
        (n1, r1), (n2, r2) = rows.items()
        mb = (r1["MBON"] / r1["ALPN"]) / (r2["MBON"] / r2["ALPN"])
        kc = (r1["Kenyon_Cell"] / r1["ALPN"]) / \
             (r2["Kenyon_Cell"] / r2["ALPN"])
        print(f"\n[dstate17] residual/high-lock ratios: "
              f"MBON share x{mb:.2f}, KC share x{kc:.2f}")
        verdict = ("high-lock VARIANT (AL-dominated, central leak "
                   "elevated)" if mb < 2 else
                   "central-heavy state (distinct from the high lock)")
        res["verdict"] = {"mbon_share_ratio": round(mb, 2),
                          "kc_share_ratio": round(kc, 2),
                          "text": verdict}
        print(f"[dstate17] verdict: {verdict}")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[dstate17] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
