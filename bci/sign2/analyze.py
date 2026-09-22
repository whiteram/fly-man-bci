"""bci/sign2 analysis: stage-wise localization of the sign odor collapse.

Per arm: per-class mean rate in each protocol window minus baseline
(A1 = first DA1 standard pulse, the Phase B collapse readout), plus
scalp rms in A1 vs baseline.  The chain ORN -> ALPN/LN -> KC -> MBON
-> scalp then shows WHERE sign kills the signal: a stage whose
increment survives is upstream of the failure; the first dead stage
localizes the lesion.  ampsign_F arms give the C2 compensation curve
(A1 increment vs chem amplitude, sign on).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = {"baseline": (0, 250), "A1": (300, 600), "A5": (1900, 2200),
       "dev": (2300, 2600), "ret": (2700, 3000)}
CHAIN = ["ORN", "ALPN", "ALIN", "ALON", "ALLN",
         "Kenyon_Cell", "MBON", "DAN"]


def scalp_uv(tag, w0, w1, b0=0, b1=250):
    p = OUT / f"{tag}_scalp.npy"
    if not p.exists():
        return np.nan
    e = np.load(p)
    e = e.mean(axis=1) if e.ndim == 2 else e
    base = e[b0:b1]
    return float(np.sqrt(((e[w0:w1] - base.mean()) ** 2).mean())
                 * 1e6 * 1.7)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for tag in meta["arms"]:
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        rates = {}
        for wname, (w0, w1) in WIN.items():
            rates[wname] = {k: round(float(z[k][w0:w1].mean()), 3)
                            for k in z.files}
        base_r = rates["baseline"]
        incs = {w: {k: round(rates[w][k] - base_r[k], 3)
                    for k in rates[w]} for w in WIN if w != "baseline"}
        res[tag] = {"rates": rates, "inc": incs,
                    "scalp_uv": {w: round(scalp_uv(tag, w0, w1), 4)
                                 for w, (w0, w1) in WIN.items()
                                 if w != "baseline"},
                    "scalp_uv_baseline_corrected": {
                        w: round(scalp_uv(tag, w0, w1)
                                 - scalp_uv(tag, 0, 250), 4)
                        for w, (w0, w1) in WIN.items()
                        if w != "baseline"}}
        print(f"\n-- {tag} --  increment over baseline (Hz) per window")
        hdr = f"   {'class':>12}" + "".join(
            f"{w:>10}" for w in WIN if w != "baseline")
        print(hdr)
        for k in CHAIN:
            if k not in z.files:
                continue
            row = f"   {k:>12}" + "".join(
                f"{incs[w].get(k, float('nan')):10.2f}"
                for w in WIN if w != "baseline")
            print(row)
        print("   scalp dRMS(uV): " + "  ".join(
            f"{w} {res[tag]['scalp_uv_baseline_corrected'][w]:+.4f}"
            for w in ("A1", "dev", "ret")))

    if "base" in res and "sign" in res:
        print("\n== collapse localization (sign/base increment ratio,"
              " A1) ==")
        for k in CHAIN:
            b = res["base"]["inc"]["A1"].get(k, np.nan)
            s = res["sign"]["inc"]["A1"].get(k, np.nan)
            r = s / b if b and b == b and b != 0 else np.nan
            print(f"   {k:>12}  base {b:8.2f}  sign {s:8.2f}  "
                  f"ratio {r if r == r else float('nan'):7.3f}")
    amps = [(t, res[t]["inc"]["A1"].get("ALPN", np.nan),
             res[t]["scalp_uv_baseline_corrected"]["A1"])
            for t in res if t.startswith("ampsign_")]
    if amps:
        print("\n== C2 amplitude compensation curve (sign on, A1) ==")
        print("   (base: ALPN +11.3 Hz, scalp +0.104 uV; sign 1x: "
              "ALPN +0.14, scalp -0.066)")
        for t, v, s in sorted(amps, key=lambda x: float(x[0].split("_")[1])):
            print(f"   {t:>12}  ALPN {v:8.2f} Hz   scalp {s:+.4f} uV")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n[sign2] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
