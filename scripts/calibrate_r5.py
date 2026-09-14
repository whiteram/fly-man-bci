"""Working-point calibration, round 5 -- mechanistic lamina (2-stage).

Stage A: the lamina histamine conductance was calibrated in exp013 on
the BOTH-lobe circuit (innervated L1/2 tetrad mass ~150 synapses). The
left-lobe pipeline circuit carries ~2x that mass, which over-clamps L1/2
to -55..-60 mV and zeroes the L release map. Re-scale g_hist here so the
INNERVATED L1/2 dark median lands at the literature -38.4 mV.
Stage B: with the lamina fixed, sweep (g_unit_lm, I_MID_BASE) for the
MID/T4/T5 windows.

Run:  python scripts/calibrate_r5.py     (~20 min)
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
T_DARK, T_END = 2000.0, 4000.0
LEVEL_PA = 350.0
TARGETS = {"MID": (5.0, 20.0), "T4": (2.0, 12.0), "T5": (2.0, 15.0)}
G_HIST_GRID = (0.16,)
G_LM_GRID = (0.04, 0.07)
I_MID_GRID = (90.0,)


def run(circuit, cal):
    n_r = len(circuit["r_ids"])
    n_mid = len(circuit["mid_ids"])
    is_t4 = np.array([str(s).startswith("T4") for s in circuit["t45_type"]])
    is_t5 = np.array([str(s).startswith("T5") for s in circuit["t45_type"]])
    n_out = int(T_END / fp.DT_MS) // 2
    cnt = {k: np.zeros(n_out) for k in ("R", "MID", "T4", "T5")}
    v_l_all = np.zeros((n_out, len(circuit["l_ids"])))

    def record(j, k, t, st, inc_f, sp):
        cnt["R"][j] = sp["R"].sum() * 1000.0 / n_r
        cnt["MID"][j] = sp["MID"].sum() * 1000.0 / n_mid
        cnt["T4"][j] = sp["T45"][is_t4].sum() * 1000.0 / is_t4.sum()
        cnt["T5"][j] = sp["T45"][is_t5].sum() * 1000.0 / is_t5.sum()
        v_l_all[j] = st["pops"]["L"].v

    fp.simulate(circuit, cal,
                lambda t: LEVEL_PA if T_DARK <= t < T_END else 0.0,
                seed=42, t_end_ms=T_END, on_sample=record)

    def w(t0, t1):
        m = slice(int(t0), int(t1))
        return {k: round(float(cnt[k][m].mean()), 1) for k in cnt}

    return {"dark": w(500.0, T_DARK), "flash": w(T_DARK + 300.0, T_END),
            "v_dark": np.median(v_l_all[100:int(T_DARK)], axis=0),
            "v_light": np.median(v_l_all[int(T_DARK) + 300:], axis=0)}


def score(r, innerv):
    s = 0.0
    for key, (lo, hi) in TARGETS.items():
        for win, weight in (("dark", 1.0), ("flash", 0.35)):
            v = r[win][key]
            s += weight * (max(0.0, lo - v) + max(0.0, v - hi)) ** 2
    if r["flash"]["T4"] < 5.0:
        s += 400.0
    vd = float(np.median(r["v_dark"][innerv]))
    depth = vd - float(np.median(r["v_light"][innerv]))
    if not (-41.6 <= vd <= -35.2):
        s += 100.0
    if not (10.0 <= depth <= 25.0):
        s += 100.0
    return s, vd, depth


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("building circuit ...")
    circuit = exp005.build_circuit()
    # innervated L1/L2 (mass >= 50 R synapses), exp013's measurement set
    l_ids = circuit["l_ids"]
    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(len(l_ids))
    e_rl = circuit["e_rl"]
    mass = np.bincount(l_index[e_rl["body_post"].to_numpy(np.int64)],
                       weights=e_rl["weight"].to_numpy(np.float64),
                       minlength=len(l_ids))
    l_type = circuit["l_type"]
    innerv = np.isin(l_type, ("L1", "L2")) & (mass >= 50.0)
    print(f"innervated L1/L2: {int(innerv.sum())} / {len(l_ids)} L cells; "
          f"mass median {np.median(mass[innerv]):.0f}")

    # ---- stage A: g_hist for the left-lobe lamina ----
    best_a = None
    for g_hist in G_HIST_GRID:
        cal = dict(fp.CAL)
        cal["G_UNIT_HIST_NS"] = g_hist
        r = run(circuit, cal)
        vd = float(np.median(r["v_dark"][innerv]))
        print(f"A: g_hist={g_hist:.2f} -> innervated L1/2 dark {vd:.1f} mV",
              flush=True)
        if best_a is None or abs(vd + 38.4) < best_a[1]:
            best_a = (g_hist, abs(vd + 38.4))
    g_hist = best_a[0]
    print(f"stage A chosen g_hist = {g_hist}")

    # ---- stage B: (g_lm, I_MID, I_T5_BASE) ----
    rows = []
    for g_lm in G_LM_GRID:
        for i_mid in I_MID_GRID:
            for t5base in (0.0, 140.0):
                cal = dict(fp.CAL)
                cal["G_UNIT_HIST_NS"] = g_hist
                cal["G_UNIT_LM_NS"] = g_lm
                cal["I_MID_BASE"] = i_mid
                cal["I_T5_BASE"] = t5base
                r = run(circuit, cal)
                sc, vd, depth = score(r, innerv)
                row = {"params": {"g_hist": g_hist, "g_unit_lm_ns": g_lm,
                                  "i_mid_base": i_mid,
                                  "i_t5_base": t5base},
                       "dark": r["dark"], "flash": r["flash"],
                       "l12_dark_mv": round(vd, 1),
                       "l12_depth_mv": round(depth, 1),
                       "score": round(sc, 2)}
                rows.append(row)
                print(f"B: g_lm={g_lm:.3f} I_MID={i_mid:.0f} "
                      f"I_T5={t5base:.0f} | "
                      f"MID {r['dark']['MID']:5.1f} "
                      f"T4 {r['dark']['T4']:4.1f} "
                      f"T5 {r['dark']['T5']:4.1f} | flashT4 "
                      f"{r['flash']['T4']:4.1f} | L {vd:.1f}/{depth:.1f} "
                      f"| {row['score']}", flush=True)
    best = min(rows, key=lambda x: x["score"])
    print("\nBEST:", json.dumps(best, indent=1))
    (OUT / "working_point_calibration_r5.json").write_text(
        json.dumps({"stage_a": {"g_hist": g_hist}, "combos": rows,
                    "best": best}, indent=1))
    print("written")


if __name__ == "__main__":
    main()
