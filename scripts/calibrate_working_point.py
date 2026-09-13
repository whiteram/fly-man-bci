"""Working-point calibration, round 4 -- under the shared pipeline.

Round 4 upgrades the remaining white-noise populations (R/L/MID) to OU
background noise (ffbm.pipeline CAL["OU"]), so every population now has
physiologically graded membrane fluctuations. The added noise raises
spontaneous rates, so I_MID_BASE / I_T4_BASE need re-fitting.

Uses the SAME simulate() as the viz export (single code path -- this
script and the export can no longer drift apart).

Protocol per combo: 4 s (2 s dark + 2 s 350 pA flash). Literature
spontaneous windows (Hz): MID 5-20, T4 2-12, T5 2-15.

Run from repository root:
    python scripts/calibrate_working_point.py     (~15 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm import pipeline as fp

import run as exp005

OUT = ROOT / "scripts" / "outputs"
T_DARK, T_FLASH = 2000.0, 2000.0
T_END = T_DARK + T_FLASH
LEVEL_PA = 350.0

TARGETS = {"MID": (5.0, 20.0), "T4": (2.0, 12.0), "T5": (2.0, 15.0)}

# (I_MID_BASE, I_T4_BASE); everything else = fp.CAL defaults
COMBOS = [
    (90.0, 175.0),
    (70.0, 175.0),
    (55.0, 175.0),
    (40.0, 175.0),
    (55.0, 165.0),
    (40.0, 165.0),
]


def run_combo(circuit, i_mid_base, i_t4_base):
    cal = dict(fp.CAL)
    cal["I_MID_BASE"] = i_mid_base
    cal["I_T4_BASE"] = i_t4_base
    r_ids = circuit["r_ids"]
    t45_type = circuit["t45_type"]
    n_r = len(r_ids)
    is_t4 = np.array([str(s).startswith("T4") for s in t45_type])
    is_t5 = np.array([str(s).startswith("T5") for s in t45_type])
    n_t45 = len(is_t4)
    n_l = len(circuit["l_ids"])
    n_mid = len(circuit["mid_ids"])
    n_out = int(T_END / fp.DT_MS) // 2
    cnt = {k: np.zeros(n_out) for k in ("R", "L", "MID", "T4", "T5")}

    def record(j, k, t, st, inc_f, sp):
        cnt["R"][j] = sp["R"].sum() * 1000.0 / n_r
        cnt["L"][j] = sp["L"].sum() * 1000.0 / n_l
        cnt["MID"][j] = sp["MID"].sum() * 1000.0 / n_mid
        cnt["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        cnt["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()

    def lum(t):
        return LEVEL_PA if T_DARK <= t < T_END else 0.0

    fp.simulate(circuit, cal, lum, seed=42, t_end_ms=T_END,
                on_sample=record)

    def w(t0, t1):
        m = slice(int(t0), int(t1))
        return {k: round(float(cnt[k][m].mean()), 1) for k in cnt}

    return {"dark": w(500.0, T_DARK), "flash": w(T_DARK + 300.0, T_END)}


def score(r):
    s = 0.0
    for key, (lo, hi) in TARGETS.items():
        for win, weight in (("dark", 1.0), ("flash", 0.35)):
            v = r[win][key]
            s += weight * (max(0.0, lo - v) + max(0.0, v - hi)) ** 2
    if r["flash"]["T4"] < 5.0:
        s += 400.0
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("building circuit ...")
    circuit = exp005.build_circuit()
    rows = []
    for i_mid, i_t4 in COMBOS:
        print(f"combo I_MID={i_mid:.0f} I_T4={i_t4:.0f} ...", flush=True)
        r = run_combo(circuit, i_mid, i_t4)
        r["params"] = {"I_MID_BASE": i_mid, "I_T4_BASE": i_t4}
        r["score"] = round(score(r), 2)
        rows.append(r)
        print(f"   dark {r['dark']}  flash {r['flash']}  score {r['score']}")
    best = min(rows, key=lambda r: r["score"])
    print("\nBEST:", json.dumps(best, indent=1))
    (OUT / "working_point_calibration_r4.json").write_text(
        json.dumps({"combos": rows, "best": best}, indent=1))
    print(f"written to {OUT / 'working_point_calibration_r4.json'}")


if __name__ == "__main__":
    main()
