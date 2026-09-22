"""bci/kcdrv2 analysis: does DAN-US alone lift naive KC?

Per arm: per-class mean rate in baseline [0,400] / US [600,2400] /
after [2600,4900] ms windows.  Verdict line: KC increment during US
and which pools responded (DAN rate verifies the injection took;
MBON should follow KC through KCM if KC fires).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = {"baseline": (0, 400), "us": (600, 2400), "after": (2600, 4900)}
CLASSES = ["Kenyon_Cell", "MBON", "ALPN", "DAN"]


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for tag in meta["arms"]:
        pop = OUT / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        rates = {w: {k: round(float(z[k][a:b].mean()), 4)
                     for k in CLASSES if k in z.files}
                 for w, (a, b) in WIN.items()}
        res[tag] = rates
        base = rates["baseline"]
        print(f"\n-- {tag} --  rate (Hz) and increment over baseline")
        for k in CLASSES:
            if k not in base:
                continue
            print(f"   {k:>12}  base {base[k]:7.3f}  "
                  f"us {rates['us'][k]:7.3f} "
                  f"(+{rates['us'][k] - base[k]:6.3f})  "
                  f"after {rates['after'][k]:7.3f} "
                  f"(+{rates['after'][k] - base[k]:6.3f})")
    if "us_ppl103" in res and "us_pam07" in res:
        dk = (res["us_ppl103"]["us"]["Kenyon_Cell"]
              - res["us_ppl103"]["baseline"]["Kenyon_Cell"])
        dp = (res["us_pam07"]["us"]["Kenyon_Cell"]
              - res["us_pam07"]["baseline"]["Kenyon_Cell"])
        print(f"\n[kcdrv2] KC lift during US: PPL103 +{dk:.3f} Hz vs "
              f"PAM07 +{dp:.3f} Hz (wiring fan-in 1978 vs 147 rows)")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[kcdrv2] summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
