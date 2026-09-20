"""bci/sleep5: the re-arm recovery curve R(t).

sleep4's grid found re-arm INCOMPLETE at ~5 tau post-collapse (p2
transient 26-58% of p1 where the single-tau resource model predicts
99%+ recovery) -- evidence for a second, much slower recovery
component.  This study quantifies it: fix the cycle dose
(U=1.5e-4, tau=10 s, seed 616 -- the deterministic-prefix seed) and
scan the SECOND ignition delay D over {10, 20, 35, 50, 80} s.

  D=35 s  reuses sleep3's sz_l2 trace u0.00015_t10000 (same seed,
          same dose, bitwise-identical p1 -- verified in sleep4)
  D=80 s  reuses chem sz_l3 with --t-end 105000
  D=10/20/50 s  new chem entries sz_l5/sz_l6/sz_l7

Readout: R(D) = p2 transient peak / p1 transient peak (scalp rms and
MBON rate), vs the single-tau prediction d(t)=1-exp(-(t-t_coll)/tau).
A systematic shortfall -> fit R(t) = a*(1-exp(-(t-tc)/tau))
+(1-a)*(1-exp(-(t-tc)/tau2)) for the second time constant.

Usage (conda ffbm, repo root):
    python bci/sleep5/acquire.py
    python bci/sleep5/analyze.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
SEED = 616
TAU_MS = 10000.0
U = 0.00015
# delay_ms -> (chem entry, trial t_end ms)
DELAYS = {10000.0: ("sz_l5", 35000.0), 20000.0: ("sz_l6", 45000.0),
          50000.0: ("sz_l7", 75000.0), 80000.0: ("sz_l3", 105000.0)}
S3_D35 = ROOT / "bci" / "sleep3" / "outputs" / "u0.00015_t10000_scalp.npy"
S3_D35_POP = ROOT / "bci" / "sleep3" / "outputs" / "u0.00015_t10000_pop.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", default=f"{U:g}")
    args = ap.parse_args()
    us = [float(u) for u in args.us.split(",")]
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"dose": {"u": U, "tau_ms": TAU_MS, "seed": SEED},
            "delays_ms": {}, "traces": {}}
    for u in us:
        for d_ms, (chem, t_end) in DELAYS.items():
            tag = f"d{d_ms / 1000:g}s_u{u:g}_s{SEED}"
            meta["delays_ms"][tag] = d_ms
            meta["traces"][tag] = {"chem": chem, "t_end_ms": t_end}
            dst = out / f"{tag}_scalp.npy"
            if dst.exists():
                print(f"[acquire] {tag}: exists, skip", flush=True)
                continue
            trial = out / f"_trial_{tag}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", chem, "--pop-rate", GROUPS,
                   "--t-end", str(t_end), "--seed", str(SEED),
                   "--std-gates", f"CEN_C:{u:g},{TAU_MS:g}",
                   "--out", str(trial), "--gpu"]
            print(f"[acquire] {tag}: p2 at {d_ms / 1000:g} s", flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                raise SystemExit(f"export failed for {tag}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
            shutil.rmtree(trial)
    # the D=35 point is sleep3's existing trace (same seed + dose)
    if S3_D35.exists():
        for u in us:
            tag = f"d35s_u{u:g}_s{SEED}"
            dst = out / f"{tag}_scalp.npy"
            if not dst.exists():
                shutil.copy(S3_D35, dst)
                if S3_D35_POP.exists():
                    shutil.copy(S3_D35_POP, out / f"{tag}_pop.npz")
            meta["delays_ms"][tag] = 35000.0
            meta["traces"][tag] = {"chem": "sz_l2 (sleep3 reuse)",
                                   "t_end_ms": 60000.0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
