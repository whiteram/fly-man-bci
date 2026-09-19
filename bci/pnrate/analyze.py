"""bci/pnrate analysis: PN/ORN rate ratio vs odor intensity.

Verdict logic vs Bhandawat 2007 (weak-input amplification):
  ratio DECREASING with intensity  -> expansive nonlinearity MATCH
  ratio ~flat                      -> linear transmission
  ratio < 1 and falling            -> net attenuation
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ODOR = (400, 1100)      # ms, settled odor window
BASE = (50, 250)        # ms, pre-odor baseline


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    rows = []
    for f in meta["scales"]:
        for s in meta["seeds"]:
            p = OUT / f"f{f:g}_s{s}_pop.npz"
            if not p.exists():
                continue
            d = np.load(p)
            r = {}
            for g in d.files:
                tr = d[g]
                r[g] = (tr[ODOR[0]:ODOR[1]].mean()
                        - tr[BASE[0]:BASE[1]].mean())
            rows.append({"f": f, "seed": s,
                         "orn": r.get("ORN", np.nan),
                         "alpn": r.get("ALPN", np.nan),
                         "alln": r.get("ALLN", np.nan),
                         "kc": r.get("Kenyon_Cell", np.nan),
                         "mbon": r.get("MBON", np.nan)})

    print("[pnrate] intensity dose-response (evoked Hz/neuron, "
          "odor 0.4-1.1 s minus baseline):")
    print("   xORNpA   ORN    ALPN   ALLN    KC   MBON   PN/ORN")
    by_f = {}
    for row in rows:
        by_f.setdefault(row["f"], []).append(row)
    for f in sorted(by_f):
        rs = by_f[f]
        m = {k: float(np.nanmean([r[k] for r in rs]))
             for k in ("orn", "alpn", "alln", "kc", "mbon")}
        ratio = m["alpn"] / m["orn"] if m["orn"] > 0 else np.nan
        print(f"   {f * 150:6.0f}  {m['orn']:6.1f} {m['alpn']:6.1f} "
              f"{m['alln']:6.1f} {m['kc']:6.2f} {m['mbon']:6.2f} "
              f"{ratio:7.2f}")
        m["pn_orn"] = ratio
        m["n"] = len(rs)
    fs = sorted(by_f)
    # anchor on the LOWEST intensity that evokes an ORN response --
    # sub-threshold arms (both populations silent) are not 0-gain, they
    # are below threshold
    ev = [f for f in fs
          if float(np.nanmean([r["orn"] for r in by_f[f]])) > 0.1]
    if len(ev) >= 2:
        lo_f, hi_f = ev[0], ev[-1]
        lo = float(np.nanmean([r["alpn"] for r in by_f[lo_f]])) / \
            float(np.nanmean([r["orn"] for r in by_f[lo_f]]))
        hi = float(np.nanmean([r["alpn"] for r in by_f[hi_f]])) / \
            float(np.nanmean([r["orn"] for r in by_f[hi_f]]))
    else:
        lo_f = hi_f = None
        lo = hi = np.nan
    if lo > hi * 1.5:
        verdict = ("expansive: weak-input amplification (MATCH "
                   "Bhandawat: PN gain highest at threshold, "
                   "compressing with intensity)")
    elif lo < hi * 0.67:
        verdict = "compressive: weak input attenuated (opposite)"
    else:
        verdict = "near-flat ratio (linear transmission)"
    print(f"\n[pnrate] PN/ORN at x{lo_f}: {lo:.2f} vs x{hi_f}: {hi:.2f}"
          f"  -> {verdict}")
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows, "verdict": verdict,
         "ratio_lo": lo, "ratio_hi": hi}, indent=1))
    print(f"[pnrate] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
