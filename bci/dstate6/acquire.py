"""bci/dstate6: the high lock's composition + long-timescale persistence.

dstate5 left two threads: (a) the 'high lock' (10x probe escalates the
20 Hz quiet lock to ALPN ~100 Hz, MBON 12, KC 4, mean -3.43 uV, std
0.017) -- is it the AL engine scaled as a WHOLE (all AL-local classes
~5x) with only marginal central recruitment?  (b) does the quiet lock
EVER decay spontaneously (60 s showed zero decay)?

Arms (seed 62, chem working point 0.002 auto):
  hi_comp    da1_exit1600 re-run with the full AL instrumentation
             (ALPN/ALIN/ALON/ALLN + central classes) -- composition of
             the high lock vs the 20 Hz lock (dstate1/5 refs)
  dc300      da1_dc300 (single DA1@150, 300 s) -- natural exit test at
             5x the dstate1 timescale

Usage (conda ffbm, repo root):
    python bci/dstate6/acquire.py
    python bci/dstate6/analyze.py
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
T_END_COMP = 60000.0
T_END_300 = 300000.0
SEED = 62
ARMS = {
    "hi_comp": {"chem": "da1_exit1600", "t_end": T_END_COMP},
    "dc300": {"chem": "da1_dc300", "t_end": T_END_300},
}


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
           "--t-end", str(spec["t_end"]), "--seed", str(SEED),
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: " + " ".join(cmd[1:]), flush=True)
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
        {"arms": ARMS, "groups": GROUPS, "seed": SEED,
         "regions": REGIONS,
         "windows_ms": {"lock20": [10000, 29000],
                        "hilock": [31000, 50000],
                        "dc300_blocks": [[10, 29], [100, 190],
                                         [200, 290]]}}, indent=1))
    print(f"[acquire] done -> {HERE / 'outputs'}")


if __name__ == "__main__":
    main()
