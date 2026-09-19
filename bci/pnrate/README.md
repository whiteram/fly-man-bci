# bci/pnrate — PN 层弱信号放大（验证表第 5 行）

机制无关的**仪器化补全**研究：VALIDATION.md 第 5 行（PN 层弱信号
放大，Bhandawat 2007：弱 ORN 输入在 PN 层被放大、强输入不放大）
此前标 ⏸"待 PN 分群速率记录"。`--pop-rate` 落地后补齐。

设计：单一气味条目 `odor_da1_short`（ORN_DA1 150 pA，脉冲
300-1200 ms）× `--chem-amp-scale` {0.25, 0.5, 1, 2, 4} × 2 种子，
记录 ORN / ALPN / ALLN / Kenyon_Cell / MBON 分群速率
（`_pop_rate.npz`）。路径：ORN（化学驱动）→ ORN_C（全增益前馈，
不受 0.002 工作点影响）→ ALPN（CEN 内）。

判定（PN/ORN 诱发率比 vs 强度）：

- 比值随强度下降 ≥1.5× → 扩张非线性（匹配 Bhandawat）
- 近平坦 → 线性传输
- 弱端更低 → 净衰减

诚实边界：ORN 侧是电流注入驱动（等效发放率），非真实气味浓度
曲线；LIF 点神经元无树突非线性——扩张性若缺失，与 KC 稀疏化
缺失（VALIDATION #3）同源（点 LIF + 电导实现的已知局限）。

## 用法

```bash
conda activate ffbm
python bci/pnrate/acquire.py     # 10 试次（~3.6 s/试次）
python bci/pnrate/analyze.py
```
