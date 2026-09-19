"""bci/shunt analysis: sparse-code effect of shunting vs subtractive APL."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ODOR = slice(300, 1200)          # ms = 1-ms bins
BASE = slice(50, 250)
RESP_SPIKES = 2                  # >=2 spikes in the 0.9-s odor window


def main():
    metas = sorted(OUT.glob("meta_g*.json"))
    meta = json.loads(metas[-1].read_text())
    g = meta["mbm_gain"]
    print(f"[shunt] MBM gain {g:g}, ALK {meta['al_gain']}, "
          f"odor {meta['chem']} x{meta['intensities']}")
    print("   arm    xORN   KC_hz   KC_resp%   MBON_hz   PN_hz")
    res = {}
    for inten in meta["intensities"]:
        for arm in meta["erevs"]:
            tag = f"{arm}_g{g:g}_x{inten:g}"
            pop, neu = OUT / f"{tag}_pop.npz", OUT / f"{tag}_neurons.npz"
            if not pop.exists():
                continue
            d = np.load(pop)
            kc = float(d["Kenyon_Cell"][ODOR].mean()
                       - d["Kenyon_Cell"][BASE].mean())
            mb = float(d["MBON"][ODOR].mean())
            pn = float(d["ALPN"][ODOR].mean())
            frac = np.nan
            if neu.exists():
                n = np.load(neu)["Kenyon_Cell"]
                frac = float((n >= RESP_SPIKES).mean() * 100.0)
            res[f"{arm}_x{inten:g}"] = {"kc_hz": kc, "kc_resp": frac,
                                        "mbon_hz": mb, "pn_hz": pn}
            print(f"   {arm:6s} x{inten:4.0f}  {kc:6.1f}  "
                  f"{frac:8.1f}   {mb:7.1f}  {pn:6.1f}")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[shunt] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
