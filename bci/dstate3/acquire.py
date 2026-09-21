"""dstate3: the lock-window fine scan + the AL-local split test.

dstate2 placed the quiescence lock between CEN_C 0.001 (no state)
and 0.003 (broad attractor) at the working point 0.002.  Two deep
dives in one study:

Part 1 -- fine scan: absolute CEN_C gains 0.0015/0.0018/0.0022/
0.0025/0.0028 (via --gain-scale multipliers on the 0.002 working
point): how WIDE is the lock window, and where exactly do the
formation and transformation thresholds sit?

Part 2 -- AL-local split (new --al-local-gain surgery): the AL's
internal rows (ALPN/ALLN/ALIN/ALON -> same set, the PN<->LN loops)
split out of CEN_C into an ALL group at an absolute gain, CEN_C held
at the 0.002 working point:
  all_g0002  neutral replication (ALL 0.002 ~ the current regime)
  all_g0001  killed (ALL 0.0001): does the lock die without the
             AL-local loop?  (dstate2's engine prediction)
  all_g0004  boosted (ALL 0.004): does the lock become the broad
             attractor at LOWER CEN_C?

Readout: ALPN/ALLN/ALON/MBON plateau rates (10-60 s), |mean|/std.

Usage (conda ffbm, repo root):
    python bci/dstate3/acquire.py
    python bci/dstate3/analyze.py
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
GROUPS = "ORN,MBON,ALPN,Kenyon_Cell,DAN,ALLN,ALIN,ALON"
SEED = 62
T_END = 60000.0
WP = 0.002
SCAN = {"f0015": 0.0015, "f0018": 0.0018, "f0022": 0.0022,
        "f0025": 0.0025, "f0028": 0.0028}          # absolute CEN_C
SPLITS = {"all_g0002": 0.002, "all_g0001": 0.0001, "all_g0004": 0.004}


def run(tag, gain_scale=None, al_local=None):
    out = HERE / "outputs"
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", "da1_dc60", "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    if gain_scale is not None:
        cmd += ["--gain-scale", f"CEN_C={gain_scale / WP:g}"]
    if al_local is not None:
        cmd += ["--al-local-gain", f"{al_local:g}"]
    print(f"[acquire] {tag}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["scan", "split",
                                                      "all"])
    args = ap.parse_args()
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if args.part in ("scan", "all"):
        for tag, g in SCAN.items():
            run(f"da1_dc60_{tag}_s{SEED}", gain_scale=g)
    if args.part in ("split", "all"):
        for tag, g in SPLITS.items():
            run(f"da1_dc60_{tag}_s{SEED}", al_local=g)
    (out / "meta.json").write_text(json.dumps(
        {"scan_abs": SCAN, "splits": SPLITS, "working_point": WP,
         "seed": SEED, "dur_ms": T_END, "chem": "da1_dc60",
         "groups": GROUPS, "regions": REGIONS}, indent=1))
    print(f"[acquire] done -> {out}")


if __name__ == "__main__":
    main()
