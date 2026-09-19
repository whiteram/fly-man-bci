"""bci/sleep2 analysis: which pathway maintains the up-state?"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PLATEAU = slice(10000, 20000)     # ms index = sample (1 kHz scalp)
IGNITE = slice(3000, 6000)


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    rows = {}
    print("[sleep2] up-state maintenance attribution "
          "(plateau = scalp rms, last 10 s of 20)")
    print("   arm       ignite_uv   plateau_uv   classes on plateau (Hz)")
    for arm in meta["arms"]:
        sc = OUT / f"{arm}_scalp.npy"
        if not sc.exists():
            continue
        e = np.load(sc)
        e = e.mean(axis=1) if e.ndim == 2 else e
        ig = float(np.sqrt((e[IGNITE] ** 2).mean()) * 1e6 * 1.7)
        pl = float(np.sqrt((e[PLATEAU] ** 2).mean()) * 1e6 * 1.7)
        cls_txt = ""
        pop = OUT / f"{arm}_pop.npz"
        if pop.exists():
            d = np.load(pop)
            hot = [f"{g} {d[g][PLATEAU].mean():.0f}" for g in d.files
                   if d[g][PLATEAU].mean() > 1.0]
            cls_txt = ", ".join(hot) or "(all < 1 Hz)"
        rows[arm] = {"ignite_uv": ig, "plateau_uv": pl}
        print(f"   {arm:9s}  {ig:9.3f}  {pl:10.3f}   {cls_txt}")
    if "base" in rows:
        b = rows["base"]["plateau_uv"]
        coll = [a for a, r in rows.items()
                if a != "base" and r["plateau_uv"] < 0.3 * b]
        none_txt = "NONE OF THE TESTED PATHWAYS"
        print(f"\n[sleep2] base plateau {b:.2f} uV; collapsed by: "
              f"{', '.join(coll) if coll else none_txt}")
        PATH = {"noCEN_C": "CEN_C recurrence (full 0.004 gain)",
                "noCEN_R": "OLR->CEN feedforward",
                "noOLR_V": "visual core->OLR",
                "noCEN_V": "visual core->CEN",
                "noORN_C": "ORN->CEN"}
        verdict = (f"maintainer = {PATH[coll[0]]}" if len(coll) == 1
                   else ("distributed/no single pathway" if not coll
                         else f"multiple contributors: "
                              f"{[PATH.get(a, a) for a in coll]}"))
        print(f"[sleep2] verdict: {verdict}")
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows, "verdict": (verdict if "base" in rows else None)},
        indent=1))


if __name__ == "__main__":
    main()
