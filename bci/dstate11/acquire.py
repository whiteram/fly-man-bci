"""bci/dstate11: can the VISUAL channel alone consolidate a state?

The entry maps cover the odor axis (dstate7/10) and the central
ignition axis (dstate9) -- but the model's primary sensory channel
(visual, and the BCI's main interface) has no current-era state map.
Prior visual work (c-VEP/SSVEP/P300) measured decaying responses
only.  Question: does sustained or flickering visual drive at
available contrast push the central engine over its entry edge?

Arms (30 s, seed 616, pop-rate central classes; late window 20-28 s):
  vis_loom   loom_static  (sustained bright disk -- closest to a
             visual DC exposure)
  vis_ssvep  ssvep1_f14.0 (14 Hz flicker)
  vis_dir    dir_270      (4 Hz drifting grating)
Reference: dark baseline (all arcs: MBON/ALPN -> 0 in late windows).

Usage (conda ffbm, repo root):
    python bci/dstate11/acquire.py
    python bci/dstate11/analyze.py
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
T_END = 30000.0
SEED = 616
ARMS = {"vis_loom": "loom_static", "vis_ssvep": "ssvep1_f14.0",
        "vis_dir": "dir_270"}
WIN = (20000, 28000)


def run(tag, vis):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", vis,
           "--pop-rate", GROUPS,
           "--t-end", str(T_END), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: visual={vis}", flush=True)
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    shutil.rmtree(trial)


def main():
    for tag, vis in ARMS.items():
        run(tag, vis)
    (HERE / "outputs" / "meta.json").write_text(json.dumps(
        {"arms": {k: ARMS[k] for k in ARMS}, "groups": GROUPS,
         "dur_ms": T_END, "seed": SEED, "regions": REGIONS,
         "window_ms": list(WIN),
         "ref_dark_late": {"mbon": 0.0, "alpn": 0.0}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
