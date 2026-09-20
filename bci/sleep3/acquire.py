"""bci/sleep3: slow-variable bistability (the sleep route, take 3).

sleep2 identified the up-state maintainer: the CEN_C recurrence at the
full 0.004 gain (an excitation-locked attractor; up-state rates are
saturated -- MBON ~395 Hz).  The mechanism-14 probes showed the STD
gate "as parameterized" never bit the maintaining current (fast
tau_rec refills within the ignition), so bistability was never
reachable.  Route per that diagnosis: a SLOW resource gate
(tau_rec 2-10 s) depletes the maintaining current over seconds during
the up-state, tips the network down, then recovers during silence --
the minimal slow negative feedback.

Design: sz_l2 = the sz_l ignition given TWICE (0.8-2.8 s and
35-37.3 s) in a 60 s trial.  The second pulse after the STD-driven
collapse is the hysteresis test: does the re-armed network latch again
(and collapse again)?

  ctl        no STD (latch forever -- the sleep2 baseline, 60 s form)
  U sweep    --std-gates CEN_C:U,5000  for U in {0.001, 0.002, 0.005, 0.01}

Readout: 1-s scalp rms + class rates; up/down thresholds 3 / 1.5 uV;
collapse delay after each pulse; dwell times.

Usage (conda ffbm, repo root):
    python bci/sleep3/acquire.py            # pilot arms (~4x8 min)
    python bci/sleep3/acquire.py --tau 10000  # tau sweep refinement
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
T_END = 60000.0
SEED = 616
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
PILOT_US = [0.001, 0.002, 0.005, 0.01]
PILOT_TAU = 5000.0


def run(tag, std):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", "sz_l2", "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if std:
        cmd += ["--std-gates", std]
    print(f"[acquire] {tag}: std={std}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=PILOT_TAU,
                    help="tau_rec in ms for the U sweep")
    ap.add_argument("--us", default=None,
                    help="comma-separated U list (default pilot set)")
    args = ap.parse_args()
    us = [float(u) for u in args.us.split(",")] if args.us else PILOT_US
    tau = args.tau
    run("ctl60", None)
    for u in us:
        run(f"u{u:g}_t{tau:g}", f"CEN_C:{u:g},{tau:g}")
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {"ctl60": None,
                  **{f"u{u:g}_t{tau:g}": f"CEN_C:{u:g},{tau:g}"
                     for u in us}},
         "dur_ms": T_END, "seed": SEED, "chem": "sz_l2",
         "pulses_ms": [[800, 2800], [35000, 37300]],
         "groups": GROUPS, "regions": REGIONS,
         "thresholds_uv": {"up": 3.0, "down": 1.5}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
