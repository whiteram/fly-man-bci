"""bci/dstate7: the lock-formation dose threshold.

dstate1: a single DA1@150 pA pulse leaves the deep QUIET LOCK
(ALPN 20, std 0.0145, >=60 s).  dstate5: a 1600 pA probe INTO the
lock escalates it to the high lock (ALPN 100).  Unmeasured so far:
where on the INTENSITY axis does the naive->lock formation threshold
sit, and does a strong FIRST pulse (300/1600) form a higher state
directly, skipping the 20 Hz lock?

Arms (single DA1 pulse [300,600] ms, 60 s, seed 62; 150 pA == the
dstate1 reference, not rerun): 25 / 50 / 100 / 300 / 1600 pA.
Classification vs the dstate refs: dark (ALPN 0, std 0.10), lock
(20, 0.0145), high lock (100, 0.017), mid/full latch.

Usage (conda ffbm, repo root):
    python bci/dstate7/acquire.py
    python bci/dstate7/analyze.py
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
GROUPS = "ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 60000.0
SEED = 62
ARMS = {"d025": "da1_d25", "d050": "da1_d50", "d100": "da1_d100",
        "d300": "da1_d300", "d1600": "da1_d1600"}
WIN = (10000, 29000)
REFS = {"dark": (0.0, 0.005), "lock": (20.0, 0.0145),
        "high_lock": (100.0, 0.017), "mid_latch": (83.0, 0.15)}


def classify(alpn, std):
    best, bd = "other", 1e9
    for name, (ra, rs) in REFS.items():
        d = abs(alpn - ra) / 20.0 + abs(std - rs) / 0.05
        if d < bd:
            best, bd = name, d
    return best


def run(tag, chem):
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
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {chem}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    for a in args.arms.split(","):
        run(a, ARMS[a])
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": ARMS, "groups": GROUPS, "dur_ms": T_END,
         "seed": SEED, "regions": REGIONS,
         "window_ms": list(WIN),
         "refs": {k: {"alpn": v[0], "std": v[1]}
                  for k, v in REFS.items()},
         "ref_150pA_dstate1": {"alpn": 19.8, "std": 0.0145}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
