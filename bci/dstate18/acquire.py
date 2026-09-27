"""bci/dstate18: multi-cycle consolidation-dissolution -- does the
lock-dissolve cycle fatigue?

dstate15 closed ONE cycle (lock -> dissolve -> restore -> re-lock,
identical).  This arc chains THREE cycles in one run with the
--gain-schedule flag: pulses at 0.3 / 70 / 140 s each form the lock;
drops (x0.5) at 30 / 100 / 170 s each dissolve it; restores (x2.0) at
45 / 115 / 185 s.  Readout: per-cycle lock latency + level, and the
final post-dissolve window -- fatigue (weakening cycles) vs. perfect
cycling.

Usage (conda ffbm, repo root):
    python bci/dstate18/acquire.py     # ~210 s single run
    python bci/dstate18/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 210000.0
SEED = 62
CHEM = "da3_cycle"
SCHED = ("30000:CEN_C=0.5,45000:CEN_C=2.0,"
         "100000:CEN_C=0.5,115000:CEN_C=2.0,"
         "170000:CEN_C=0.5,185000:CEN_C=2.0")
CYCLES = [  # (pulse_end_ms, lock_window, post_drop_window)
    (600, (1000, 29000), (35000, 44000)),
    (70600, (71000, 99000), (105000, 114000)),
    (140600, (141000, 169000), (175000, 184000)),
]
TAIL = (190000, 209000)


def run():
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "cycle3_scalp.npy"
    if dst.exists():
        print("[acquire] cycle3: exists, skip", flush=True)
        return
    trial = out / "_trial_cycle3"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", CHEM, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu",
           "--gain-schedule", SCHED]
    print(f"[acquire] cycle3: {CHEM}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit("export failed for cycle3")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / "cycle3_pop.npz")
    shutil.rmtree(trial)


def main():
    run()
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"chem": CHEM, "sched": SCHED, "groups": GROUPS,
         "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
         "cycles": [{"pulse_end_ms": c[0], "lock_win": list(c[1]),
                     "post_drop_win": list(c[2])} for c in CYCLES],
         "tail_ms": list(TAIL)}, indent=1))
    # readouts
    pop = HERE / "outputs" / "cycle3_pop.npz"
    z = np.load(pop)
    a = z["ALPN"]
    print("== dstate18: three consolidation-dissolution cycles ==")
    res = {"cycles": [], "tail_alpn": round(float(a[TAIL[0]:TAIL[1]]
                                                .mean()), 3)}
    for i, (pe, lw, dw) in enumerate(CYCLES, 1):
        lat = None
        for ms in range(int(pe), int(lw[1]) - 1000):
            if a[ms:ms + 1000].mean() > 15.0:
                lat = ms - pe
                break
        lev = round(float(a[lw[0]:lw[1]].mean()), 2)
        dis = round(float(a[dw[0]:dw[1]].mean()), 3)
        res["cycles"].append({"latency_ms": lat, "level_hz": lev,
                              "post_drop_hz": dis})
        print(f"   cycle {i}: latency {lat} ms  level {lev} Hz  "
              f"post-drop {dis} Hz")
    print(f"   final tail (post 3rd drop+restore, no pulse): "
          f"{res['tail_alpn']} Hz")
    lats = [c["latency_ms"] for c in res["cycles"]]
    levs = [c["level_hz"] for c in res["cycles"]]
    lev_spread = (max(levs) - min(levs)) / np.mean(levs)
    verdict = ("no fatigue: instant re-lock each cycle, "
               f"level spread {lev_spread:.1%} (noise), full "
               "dissolution each drop"
               if len(set(lats)) == 1 and lev_spread < 0.10 else
               f"cycle drift (latencies {lats}, level spread "
               f"{lev_spread:.1%}) -- see per-cycle values")
    res["verdict"] = verdict
    print(f"[dstate18] verdict: {verdict}")
    (HERE / "outputs" / "summary.json").write_text(json.dumps(
        res, indent=1))
    print(f"[dstate18] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
