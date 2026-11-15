"""bci/condstate21: 静默锁 300 s 漂移率——τ_sig 剂量-响应第三点。

cs15（hlock：DAN 积分 5110 ms → τ_sig 336 s）/ cs20（resid：362 ms
→ 2948 s）给出"可塑性写入速率决定签名寿命"的两点。本弧补第三点：
qlock（静默锁，ALPN 20 Hz，门压制 −20%——预测 τ_sig 居中或更长）。

臂: cs21locknt（新目录条目：DA1@150 [300,600] 单脉冲成静默锁，
无训练/探针——cs2lock 去掉训练通道），seed 62，t_end 300 s，
无调度，dangate2 标定配方同 condstate2。

用法 (conda ffbm, repo root):
    python bci/condstate21/acquire.py     # 幂等, ~52 min GPU
数据: outputs/qll62_scalp.npy (300k x 17), qll62_pop.npz,
      states/qll62.npz, meta.json
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
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN"
T_END = 300000.0
ARMS = {"qll62": {"chem": "cs21locknt", "seed": 62}}


def run(tag, spec):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    dst = out / f"{tag}_scalp.npy"
    if dst.exists() and (out / "states" / f"{tag}.npz").exists() \
            and (out / f"{tag}_pop.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--chem-input", spec["chem"], "--pop-rate", GROUPS,
           "--plastic-mb", "--plastic-lr=-2e-7",
           "--plastic-w0", "0.03",
           "--plastic-kcm-gain", "0.015",
           "--mbon-dan-gain", "178",
           "--plastic-dan-gate", "150,0.02,400",
           "--t-end", str(T_END), "--seed", str(spec["seed"]),
           "--out", str(trial), "--gpu",
           "--plastic-state-out", str(out / "states" / f"{tag}.npz")]
    print(f"[acquire] {tag}: {spec['chem']} seed {spec['seed']} "
          f"t_end {T_END/1000:.0f} s", flush=True)
    t0 = time.time()
    r_ = subprocess.run(cmd, cwd=str(ROOT))
    if r_.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", dst)
    shutil.copy(trial / "_dan_trace.npy", out / f"{tag}_dan.npy")
    shutil.copy(trial / "_pop_rate.npz", out / f"{tag}_pop.npz")
    time.sleep(2.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] {tag}: done in {(time.time()-t0)/60:.1f} min",
          flush=True)


def main():
    for tag, spec in ARMS.items():
        run(tag, spec)
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{t: ARMS[t] for t in ARMS}},
                 "groups": GROUPS, "dur_ms": T_END,
                 "regions": REGIONS,
                 "chem": "cs21locknt (DA1@150 only, no training)",
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)",
                 "question": "qlock 签名 300 s 漂移率——τ_sig "
                             "剂量-响应第三点"})
    meta_p.write_text(json.dumps(meta, indent=1))
    pop = np.load(HERE / "outputs" / "qll62_pop.npz")
    for g in GROUPS.split(","):
        r = pop[g]
        segs = [(2000, 10000), (100000, 110000), (280000, 290000)]
        print(g, [round(float(r[a:b].mean()), 2) for a, b in segs])


if __name__ == "__main__":
    main()
