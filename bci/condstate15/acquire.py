"""bci/condstate15: 静息签名时间稳定性——高锁态保持 300 s。

condstate14 的监测器在 8-17 s 稳态窗上训练/验证；dstate15 证明
溶锁后逐位恢复，但从未测过**状态自身的长时程平稳性**：高锁
（AL 自持 ~99 Hz）在无进一步刺激下保持 5 分钟，签名（ALPN /
头皮带 RMS / 峭度 / 谱指数）会不会漂移？监测器在 t>22 s 还成立
吗？

臂（t_end 300 s，dangate2 标定配方同 condstate4，单臂单 trial）：
  hll62  cs4hlnt（DA1@1600 [300,600] 单脉冲 → 高锁，无训练/探针）
        —— cs4hlnt 脉冲全部在前 0.6 s，t_end 由 CLI 决定，
        直接外推到 300 s 不改化学目录。

判据（stability.py）：
  1. 签名平稳性：ALPN/KC/MBON 池率 + 头皮 2 s 滑窗特征
     （带 RMS/峭度/谱指数/事件率）在 300 s 内的漂移量 vs
     condstate14 训练分布的类内散布；
  2. 监测器时间外推：condstate14 的 seed63 折模型（训练数据
     完全不含本 run）逐窗打标签——高锁分类保持率、有无滑向
     其他类的漂移；
  3. 若有漂移：特征-时间斜率与主漂移方向。

用法 (conda ffbm, repo root):
    python bci/condstate15/acquire.py          # 幂等
数据: outputs/hll62_scalp.npy (300k x 17), hll62_pop.npz,
      states/hll62.npz, meta.json
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
ARMS = {"hll62": {"chem": "cs4hlnt", "seed": 62}}


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
    # meta: load-merge-rewrite（era 规则）
    meta_p = HERE / "outputs" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"arms": {**meta.get("arms", {}),
                          **{t: ARMS[t] for t in ARMS}},
                 "groups": GROUPS, "dur_ms": T_END,
                 "regions": REGIONS,
                 "chem": "cs4hlnt (DA1@1600 [300,600] only)",
                 "calib": "dangate2 (lr -2e-7, w0 0.03, kcm 0.015, "
                          "mbmd 178, gate 150,0.02,400)",
                 "question": "高锁静息签名 300 s 平稳性 + "
                             "condstate14 监测器时间外推"})
    meta_p.write_text(json.dumps(meta, indent=1))
    # 快速读出：签名区间（成锁后/中段/末段）
    pop = np.load(HERE / "outputs" / "hll62_pop.npz")
    for g in GROUPS.split(","):
        r = pop[g]
        segs = [(2000, 10000), (100000, 110000), (280000, 290000)]
        print(g, [round(float(r[a:b].mean()), 2) for a, b in segs])


if __name__ == "__main__":
    main()
