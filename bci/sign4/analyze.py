"""bci/sign4 analysis: does the E:I rebalance restore a graded odor
response under sign?  A1 increment + tail latch check per arm.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CLASSES = ["ORN", "ALPN", "ALIN", "ALON", "ALLN",
           "Kenyon_Cell", "MBON", "DAN"]


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    print("== sign4: E:I rebalance scan under sign (mmn_match A1) ==")
    print("   refs: base ALPN +11.3 tail 0 | sign3 g002 (I=1.0) "
          "ALPN +0.17 tail 0 | sign3 latch tail >=134")
    res = {}
    for tag, bal in meta["arms"].items():
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        inc = {k: round(float(z[k][300:600].mean())
                        - float(z[k][0:250].mean()), 3)
               for k in CLASSES if k in z.files}
        tail = round(float(z["ALPN"][3200:3600].mean()), 2)
        e = np.load(OUT / f"{tag}_scalp.npy")
        e = e.mean(axis=1) if e.ndim == 2 else e
        uv = float(np.sqrt(((e[300:600] - e[0:250].mean()) ** 2)
                           .mean()) * 1e6 * 1.7) - \
            float(np.sqrt(((e[0:250] - e[0:250].mean()) ** 2)
                          .mean()) * 1e6 * 1.7)
        res[tag] = {"balance": bal, "inc_A1": inc, "tail_ALPN": tail,
                    "scalp_dUV": round(uv, 4)}
        print(f"   {tag} ({bal}): ALPN A1 {inc.get('ALPN'):8.3f}  "
              f"tail {tail:8.2f}  "
              f"{'LATCH' if tail > 20 else 'graded-or-dead'}  "
              f"scalp {uv:+.4f} uV")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[sign4] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
