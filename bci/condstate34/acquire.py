"""bci/condstate34: naive 300 s 长跑——MI 诚实化 + 底的长时程稳定性。

cs33 留下样本受限带（干净 MI 1.26-1.62 bit，naive 池仅 11 窗）；
本弧跑 naive 底的 300 s（暗、无 chem——cs25 floor 探针协议外推，
seed 62），一石二鸟：
  1. ~145 个非重叠 naive 窗 → cs32 MI 的诚实复测（关闭 1.26-1.62 带）;
  2. 第 4 个态的长时程稳定性：视觉通路强直底（L 100%/MID 18 Hz）
     300 s 是否漂移 → naive τ 下界（与三态寿命谱合流）。

用法 (conda ffbm, repo root):
    python bci/condstate34/acquire.py     # ~45 min GPU, 幂等
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN,CX"
TYPES = "lLN1_bc"
LAYERS = ("R", "L", "MID", "T4", "T5")
T_END = 300000.0
ARMS = {"nll62": {}}


def run(tag):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "nll62_layers.npz").exists():
        print("[acquire] nll62: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--t-end", str(T_END), "--seed", "62",
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: dark no-chem t_end {T_END/1000:.0f} s",
          flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("export failed")
    shutil.copy(trial / "_debug_phi_scalp.npy", out / f"{tag}_scalp.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    vd = json.loads((trial / "viz_data.json").read_text(encoding="utf-8"))
    np.savez(out / f"{tag}_layers.npz",
             **{k: np.asarray(v, dtype=np.float32)
                for k, v in vd["rates"].items() if k in LAYERS})
    time.sleep(1.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] {tag}: done {(time.time()-t0)/60:.1f} min",
          flush=True)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"naive_300s": {"regions": REGIONS, "chem": None,
                                "visual": "dark", "dur_ms": T_END,
                                "seed": 62,
                                "question": "naive MI 诚实化 + 底 300 s "
                                            "稳定性/寿命下界"}})
    meta_p.write_text(json.dumps(meta, indent=1))


def main():
    for tag in ARMS:
        run(tag)
    lay = np.load(HERE / "outputs" / "nll62_layers.npz")
    for k in LAYERS:
        r = lay[k]
        print(k, [round(float(r[a:b].mean()), 2)
                  for a, b in ((2000, 10000), (100000, 110000),
                               (280000, 290000))])


if __name__ == "__main__":
    main()
