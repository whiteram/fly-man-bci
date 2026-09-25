"""bci/dstate10: the lock-formation DURATION threshold.

dstate7 mapped the dose axis (a single 300-ms pulse consolidates at
150 pA but not 100).  The untested axis: how BRIEF can the exposure
be at consolidating dose (150 pA) and still leave the permanent quiet
lock?  Clinically: can a flash trigger the permanent state?

Arms (60 s, seed 62; the 300-ms reference = dstate7/dstate1's lock):
  t025 / t050 / t100   DA1@150 pA for 25 / 50 / 100 ms
  s63                  seed-63 replication of the 300-ms lock
                       (second-seed point for dstate7's edge)

Usage (conda ffbm, repo root):
    python bci/dstate10/acquire.py
    python bci/dstate10/analyze.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 60000.0
ARMS = {"t025": ("da1_t25", 62), "t050": ("da1_t50", 62),
        "t100": ("da1_t100", 62), "s63": ("da1_dc60", 63)}
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145)}


def run(tag, chem, seed):
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
           "--t-end", str(T_END), "--seed", str(seed),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {chem} s{seed}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, (chem, seed) in ARMS.items():
        run(tag, chem, seed)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: {"chem": v[0], "seed": v[1]}
                  for k, v in ARMS.items()},
         "groups": GROUPS, "dur_ms": T_END, "regions": REGIONS,
         "window_ms": list(WIN), "dose_pa": 150.0,
         "refs": {k: {"alpn": v[0], "std": v[1]}
                  for k, v in REFS.items()}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
