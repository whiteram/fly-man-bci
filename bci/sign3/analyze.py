"""bci/sign3 analysis: ALL-gain scan under sign -- does the AL odor
response come back?  A1 increment per arm vs the base (no-sign) and
sign (neutral gain) references.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CLASSES = ["ORN", "ALPN", "ALIN", "ALON", "ALLN",
           "Kenyon_Cell", "MBON", "DAN"]


def scalp_uv(p, w0, w1, b0=0, b1=250):
    if not p.exists():
        return float("nan")
    e = np.load(p)
    e = e.mean(axis=1) if e.ndim == 2 else e
    return float(np.sqrt(((e[w0:w1] - e[b0:b1].mean()) ** 2).mean())
                 * 1e6 * 1.7)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    print("== sign3: ALL-gain scan under sign (mmn_match A1) ==")
    print("   refs: base(no sign) ALPN +11.27 ALON +27.62 | "
          "sign@0.002 ALPN +0.14 (sign2)")
    res = {}
    for tag, g in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        inc = {k: round(float(z[k][300:600].mean())
                        - float(z[k][0:250].mean()), 3)
               for k in CLASSES if k in z.files}
        uv = scalp_uv(OUT / f"{tag}_scalp.npy", 300, 600) - \
            scalp_uv(OUT / f"{tag}_scalp.npy", 0, 250)
        res[tag] = {"gain": g, "inc_A1": inc, "scalp_dUV": round(uv, 4)}
        print(f"   {tag} (g={g:g}): ALPN {inc.get('ALPN'):7.3f}  "
              f"ALON {inc.get('ALON'):7.3f}  ALLN {inc.get('ALLN'):7.3f}  "
              f"KC {inc.get('Kenyon_Cell'):6.3f}  scalp {uv:+.4f} uV")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[sign3] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
