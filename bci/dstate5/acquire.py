"""bci/dstate5: what EXITS the quiet lock?  (dstate1-4 follow-up)

dstate1 found the odor-DC state is a deep QUIET LOCK (single DA1@150
pulse; ALPN self-sustains ~20 Hz alone-ish, scalp std 0.0145, persists
>=60 s with zero decay); dstate2/3 mapped the ENTRY phase diagram
(CEN_C gain ladder: lock window 0.0016-0.0023).  The open question is
the EXIT: what perturbation, applied to the ESTABLISHED lock, kicks
the network back to dark (ALPN->0, std->~0.10)?

Design: da1_dc60 establishes the lock (300-600 ms); at t=30 s a probe
pulse lands on the locked network (60 s trials, seeds 62/63):

  exit_ctl    DA1@150   (same drive -- re-assert or ignore?)
  exit_600    DA1@600   (4x)
  exit_1600   DA1@1600  (~10x, the conditioning-level drive)
  exit_szl    LAL*@L 600 pA x2.3 s (the central ignition pulse)

State readouts: 1-s scalp std + class rates in pre [10,29] s vs post
[31,50] s windows.  Exit = ALPN ->~0 and std ->~0.10 (dark); staying
locked = unchanged; escalation = std ->0.18 (plateau) or attractor.

Usage (conda ffbm, repo root):
    python bci/dstate5/acquire.py --arms exit_ctl,exit_600
    python bci/dstate5/analyze.py
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
T_END = 60000.0
SEEDS = [62, 63]
ARMS = {"exit_ctl": "da1_exit150", "exit_600": "da1_exit600",
        "exit_1600": "da1_exit1600", "exit_szl": "da1_exitszl"}
PRE, POST = (10000, 29000), (31000, 50000)


def run(arm, chem, seed):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{arm}_s{seed}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {arm} s{seed}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{arm}_s{seed}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", chem, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {arm} s{seed}: " + " ".join(cmd[1:]), flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {arm} s{seed}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz",
                out / f"{arm}_s{seed}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for arm in arms:
        for seed in seeds:
            run(arm, ARMS[arm], seed)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {a: ARMS[a] for a in arms}, "seeds": seeds,
         "groups": GROUPS, "dur_ms": T_END, "regions": REGIONS,
         "probe_ms": [30000, 30600],
         "windows_ms": {"pre": list(PRE), "post": list(POST)},
         "state_refs": {"dark": {"alpn_hz": 0, "std_uv": 0.100},
                        "lock": {"alpn_hz": 19.8, "std_uv": 0.0145},
                        "plateau": {"alpn_hz": 38, "std_uv": 0.18}}},
        indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
