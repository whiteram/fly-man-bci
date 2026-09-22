"""bci/sign4: per-leg rebalance of the sign-on AL engine -- the first
real surgery of the sign recalibration program.

sign3's verdict: under sign the PN<->LN loop is dead-or-latched at
EVERY whole-group ALL gain (no window).  The remaining knob is the
E:I RATIO: --al-leg-balance 'E,I' scales the ALL group's excitatory
(sign=+1) and inhibitory (sign=-1) row weights independently.  Weaken
the now-live inhibitory leg (I<1) to hand the loop back its net
regenerative gain at the neutral total gain, and hunt for a graded,
non-latching odor response (base reference: ALPN A1 +11.3 Hz, tail 0).

Arms (all --use-sign-all --al-local-gain 0.002, mmn_match A1, seed
616; I=1.0 == sign3's g002 arm, not rerun):
  i030 '1.0,0.3'   i050 '1.0,0.5'   i070 '1.0,0.7'   e150 '1.5,1.0'

Usage (conda ffbm, repo root):
    python bci/sign4/acquire.py
    python bci/sign4/analyze.py
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
GROUPS = "ORN,ALPN,ALIN,ALON,ALLN,Kenyon_Cell,MBON,DAN"
T_END = 3600.0
SEED = 616
ARMS = {"i030": ("0.002", "1.0,0.3"), "i050": ("0.002", "1.0,0.5"),
        "i070": ("0.002", "1.0,0.7"), "e150": ("0.002", "1.5,1.0"),
        "g003i050": ("0.003", "1.0,0.5"), "g004i050": ("0.004", "1.0,0.5"),
        "g005i050": ("0.005", "1.0,0.5"), "g003i030": ("0.003", "1.0,0.3"),
        "g004i030": ("0.004", "1.0,0.3"), "g005i030": ("0.005", "1.0,0.3")}


def run(tag, spec):
    gain, bal = spec
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", "mmn_match", "--use-sign-all",
           "--al-local-gain", gain,
           "--al-leg-balance", bal, "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: gain {gain} balance {bal}", flush=True)
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
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{a: {"gain": ARMS[a][0],
                                 "balance": ARMS[a][1]}
                             for a in args.arms.split(",")}},
                 "sign": True, "chem": "mmn_match", "groups": GROUPS,
                 "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
                 "ref_base": {"ALPN_A1": 11.27, "ALON_A1": 27.62},
                 "ref_sign3_g002": {"ALPN_A1": 0.17, "tail": 0.0}})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
