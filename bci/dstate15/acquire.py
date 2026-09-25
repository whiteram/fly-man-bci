"""bci/dstate15: the re-exposure trace hypothesis -- does a dissolved
lock leave a sensitization trace (faster re-lock)?

dstate14 cured the lock with a mid-run gain drop and left one open
question: is the cured network truly naive, or does re-exposure
re-lock faster/lower?  One 62-s run chains everything with the new
--gain-schedule flag:

  pulse 1 @0.3 s        -> lock forms (latency L1 from naive dark)
  t=30 s  gain x0.5     -> dissolved (kcdrv: dark by ~40 s)
  t=45 s  gain x1.0     -> restored; 45-50 s shows whether the
                           dissolved network RE-LOCKS SPONTANEOUSLY
  pulse 2 @50 s         -> latency L2; L2 < L1 = trace/sensitization

Arms (seed 62):
  trace_test   da2_trace + schedule {30s x0.5, 45s x1.0}
  restore_only da1_dc60  + same schedule (no pulse 2: spontaneous
               re-lock check only)

Usage (conda ffbm, repo root):
    python bci/dstate15/acquire.py
    python bci/dstate15/analyze.py
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
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN"
T_END = 62000.0
SEED = 62
ARMS = {"trace_test": ("da2_trace", "30000:CEN_C=0.5,45000:CEN_C=2.0"),
        "restore_only": ("da1_dc60", "30000:CEN_C=0.5,45000:CEN_C=2.0")}
DT = 1.0  # ms resolution for latency readout


def lock_latency(z, t0_ms, ceil=55000):
    """ms from t0 until the 1-s mean ALPN first exceeds 15 Hz."""
    alpn = z["ALPN"]
    for ms in range(int(t0_ms), int(ceil) - 1000):
        if ms + 1000 > len(alpn):
            break
        if alpn[ms:ms + 1000].mean() > 15.0:
            return ms - t0_ms
    return None


def run(tag, chem, sched):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu",
           "--gain-schedule", sched]
    print(f"[acquire] {tag}: {chem}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, (chem, sched) in ARMS.items():
        run(tag, chem, sched)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: {"chem": v[0], "sched": v[1]}
                  for k, v in ARMS.items()},
         "groups": GROUPS, "dur_ms": T_END, "seed": SEED,
         "regions": REGIONS,
         "events_ms": {"pulse1": [300, 600], "drop": 30000,
                       "restore": 45000, "pulse2": [50000, 50600]}},
        indent=1))
    # readouts
    print("== dstate15: re-exposure trace test ==")
    res = {}
    for tag in ARMS:
        pop = HERE / "outputs" / f"{tag}_pop.npz"
        if not pop.exists():
            continue
        z = np.load(pop)
        a = z["ALPN"]
        r = {
            "latency1_ms": lock_latency(z, 600),
            "alpn_40_44s": round(float(a[40000:44000].mean()), 2),
            "alpn_46_50s": round(float(a[46000:50000].mean()), 2),
            "latency2_ms": lock_latency(z, 50600, 62000),
            "alpn_tail": round(float(a[56000:62000].mean()), 2),
        }
        res[tag] = r
        print(f"   {tag:>12}: L1 {r['latency1_ms']} ms  "
              f"post-dissolve {r['alpn_40_44s']} Hz  "
              f"post-restore {r['alpn_46_50s']} Hz  "
              f"L2 {r['latency2_ms']} ms  tail {r['alpn_tail']} Hz")
    if "trace_test" in res:
        l1, l2 = res["trace_test"]["latency1_ms"], \
            res["trace_test"]["latency2_ms"]
        if l1 and l2:
            print(f"[dstate15] verdict: "
                  f"(L2 {l2} vs L1 {l1}) "
                  f"{'SENSITIZED (trace)' if l2 < 0.7 * l1 else 'no trace'}")
    (HERE / "outputs" / "summary.json").write_text(
        json.dumps(res, indent=1))
    print(f"[dstate15] summary -> {HERE / 'outputs' / 'summary.json'}")


if __name__ == "__main__":
    main()
