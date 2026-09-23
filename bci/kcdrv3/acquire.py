"""bci/kcdrv3 v2: map the DAN->KC gate across TODAY's state ladder.

The kcdrv1 target state (sleep-era post-collapse plateau, KC baseline
0.1) is NOT reproducible in today's code: sleep5's exact recipe
(sz_l3 + std 0.00015/5000, seed 616) now settles at a MID-LATCH
(ALPN 83 / MBON 76 / KC 12 / DAN 6.5) instead of the old plateau
(ALPN 38 / MBON 3) -- the state landscape drifted with the code
(itself a recorded finding).  So the bracket is completed on today's
ladder instead:

  lock_us   quiet lock (da1@150; ALPN 20, KC baseline 0) + PPL103 US
            at 15-17 s -> predict DEAD (same subthreshold naive)
  ml_us     mid-latch (auto 0.002 x2.0 = 0.004 + std 0.00015/5000;
            ALPN 83, KC baseline 12) + same US -> alive?
  ml_ctl    mid-latch without US -> fluctuation floor

If the channel opens between lock and mid-latch, gating tracks the
state's depolarization level; if ml is also dead, the kcdrv1 +6 was
carried by that state's specific central chain and the story is
revised (honest).

Usage (conda ffbm, repo root):
    python bci/kcdrv3/acquire.py
    python bci/kcdrv3/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "Kenyon_Cell,MBON,ALPN,DAN"
T_END = 25000.0
SEED = 616
ARMS = {
    "lock_us": {"chem": "kc_lock_us"},
    "ml_us": {"chem": "kcp_us", "std": True, "g2": True},
    "ml_ctl": {"chem": "kcp_ctl", "std": True, "g2": True},
}
BASE = (11000, 14500)
US = (15100, 16900)


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"], "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if spec.get("std"):
        cmd += ["--std-gates", "CEN_C:0.00015,5000"]
    if spec.get("g2"):
        cmd += ["--gain-scale", "CEN_C=2.0"]
    print(f"[acquire] {tag}: {spec}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "groups": GROUPS, "dur_ms": T_END,
         "seed": SEED, "regions": REGIONS,
         "windows_ms": {"base": list(BASE), "us": list(US)},
         "state_ladder_hz": {"naive_kc": 0.0, "quiet_lock_kc": 0.0,
                             "mid_latch_kc": 12.0, "full_latch_kc": 74.0},
         "era_note": "sleep-era plateau (KC 0.1) not reproducible "
                     "today; bracket mapped on the current ladder"},
        indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
