"""bci/dstate12: the (cell loss x dose) consolidation surface.

dstate8: at F=0.7 AL cell loss the quiet lock is LOST at consolidat-
ing dose (150 pA) -- subsampling pushes the loop out of its window.
dstate7: at full population the dose axis is sharp (100-150 pA ->
lock, 1600 -> high lock).  The clinical question in between: does a
DEGENERATED network re-consolidate at higher doses, i.e. does cell
loss move the exposure hazard map, and can the high lock still form?

Arms (60 s, seed 62, single DA1 pulse [300,600] ms, working point,
--al-local-gain 0.002 + --al-subsample F):
  f070_d600   F=0.7, 600 pA   (mid dose)
  f070_d1600  F=0.7, 1600 pA  (full-scale dose)
  f050_d600   F=0.5, 600 pA
  f050_d1600  F=0.5, 1600 pA
Refs: dstate7 (F=1.0: 150->lock, 1600->high lock),
dstate8 (F<=0.7 at 150 pA: dark).

Usage (conda ffbm, repo root):
    python bci/dstate12/acquire.py
    python bci/dstate12/analyze.py
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
SEED = 62
ARMS = {"f070_d600": ("da1_d600", 0.7), "f070_d1600": ("da1_d1600", 0.7),
        "f050_d600": ("da1_d600", 0.5), "f050_d1600": ("da1_d1600", 0.5)}
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "high_lock": (100.0, 0.017)}


def classify(alpn, std):
    best, bd = "other", 1e9
    for name, (ra, rs) in REFS.items():
        d = abs(alpn - ra) / 20.0 + abs(std - rs) / 0.05
        if d < bd:
            best, bd = name, d
    return best


def run(tag, chem, f):
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
           "--al-local-gain", "0.002", "--al-subsample", f"{f:g}",
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {chem} F={f:g}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, (chem, f) in ARMS.items():
        run(tag, chem, f)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: {"chem": v[0], "F": v[1]} for k, v in ARMS.items()},
         "groups": GROUPS, "dur_ms": T_END, "seed": SEED,
         "regions": REGIONS, "window_ms": list(WIN),
         "refs": {k: {"alpn": v[0], "std": v[1]}
                  for k, v in REFS.items()},
         "refs_prior": {"f100": "dstate7 (150 lock / 1600 high lock)",
                        "f070_d150": "dstate8 f070 (dark)"}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
