"""bci/sleep4: is the slow oscillation endogenous? + tau>=15 s dwell.

sleep3 closed the loop: post-fix slow STD on CEN_C yields the full
cycle at U=1.5e-4 / tau=10 s (up ~5 s -> collapse -> down ~28 s ~ 3tau
re-arm -> re-ignition) -- but re-ignition used the EXTERNAL 35 s pulse,
so "endogenous vs externally driven" stayed open (honest note).

Arm 1 (endogenous test): chem sz_l1 = the ignition ONCE, trial 90 s.
After collapse (~8 s) and full re-arm (~3tau ~ 30 s), the window
[40, 90] s carries NO external pulse: any up-episode there means the
network re-latches on noise alone -> endogenous slow oscillation
(major upgrade).  Silence -> the external-drive boundary is confirmed
(honest close).  2 seeds.

Arm 2 (tau grid, if external): chem sz_l3 = second ignition moved to
80 s (at tau = 15/20 s, 3tau re-arm completes ~55-70 s after the ~8 s
collapse; the old 35 s pulse would hit partially-armed resources).
Readout: collapse delay after pulse 2 vs tau (dwell scaling), plus the
pre-pulse level at 70-80 s (re-arm completeness).

Usage (conda ffbm, repo root):
    python bci/sleep4/acquire.py --chem sz_l1 --t-end 90000 \
        --tau 10000 --us 0.00015 --seeds 616,617
    python bci/sleep4/acquire.py --chem sz_l3 --t-end 120000 \
        --tau 15000 --us 0.0001,0.00015 --seeds 616
    python bci/sleep4/analyze.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chem", default="sz_l1",
                    help="chem entry id (sz_l1 single pulse / sz_l3 two)")
    ap.add_argument("--t-end", type=float, default=90000.0)
    ap.add_argument("--tau", type=float, default=10000.0)
    ap.add_argument("--us", default="0.00015")
    ap.add_argument("--seeds", default="616")
    ap.add_argument("--ctl", action="store_true",
                    help="also run the no-STD control arm")
    args = ap.parse_args()
    us = [float(u) for u in args.us.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {"arms": {}}
    arms = []
    if args.ctl:
        arms.append(("ctl", None))
    for u in us:
        arms.append((f"u{u:g}_t{args.tau:g}", f"CEN_C:{u:g},{args.tau:g}"))
    for name, std in arms:
        for seed in seeds:
            tag = f"{args.chem}_{name}_s{seed}"
            meta["arms"][tag] = {
                "chem": args.chem, "t_end_ms": args.t_end,
                "std": std, "tau_ms": args.tau, "seed": seed}
            dst = out / f"{tag}_scalp.npy"
            if dst.exists():
                print(f"[acquire] {tag}: exists, skip", flush=True)
                continue
            trial = out / f"_trial_{tag}"
            cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
                   "--regions", REGIONS, "--visual-input", "dark",
                   "--chem-input", args.chem, "--pop-rate", GROUPS,
                   "--t-end", str(args.t_end), "--seed", str(seed),
                   "--out", str(trial), "--gpu"]
            if std:
                cmd += ["--std-gates", std]
            print(f"[acquire] {tag}: std={std}", flush=True)
            r_ = subprocess.run(cmd, cwd=str(ROOT))
            if r_.returncode != 0:
                meta_p.write_text(json.dumps(meta, indent=1))
                raise SystemExit(f"export failed for {tag}")
            shutil.copy(trial / "_debug_phi_scalp.npy", dst)
            shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
            shutil.rmtree(trial)
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
