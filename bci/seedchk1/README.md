# bci/seedchk1：头条数字的多种子稳健性抽检

项目主体结论多为单种子（各 README 均有诚实边界）。本弧线对两个
最承重的量化边做**新种子家族复现**：共现阻断比（此前只有
1042/1092 系）与 sign 气味塌缩（此前只有 616）。

## 结果：两条边都跨种子稳固

| 量 | 原种子 | 新种子 (2042/2092 / 618) | 判决 |
|---|---|---|---|
| comp 臂 B 学习比（共现双奖励） | 1.00 | **0.98** | ✓ 复现 |
| glob 臂 B 学习比（阻断） | 0.68 | **0.67** | ✓ 复现（dw_B 0.179/0.266 vs 0.181/0.267） |
| sign 气味塌缩（ALPN A1） | 82×（+11.27/+0.14） | **63×（+9.25/+0.15）** | ✓ 复现（量级差在种子噪声内） |
| sign@0.002 死态响应 | +0.137 | +0.147 | ✓ 几乎一致 |

顺带测量：base 气味响应本身有 ~18% 种子浮动（+11.27 → +9.25），
而 sign 死态几乎无浮动（+0.137 → +0.147）——**种子噪声主要
进入环路的放大输出，塌缩后的前馈直通分量是种子无关的**（与
sign4 的「0.15 Hz = 前馈直通」归因互证）。

结论：**comp 系列的 2×2 判决（1.00 vs 0.68）与 sign 系列的塌缩
定位不依赖特定种子**。项目其余单种子结论的置信度按此抽样推断
（定性边稳健、绝对值 ~20% 浮动）。

诚实边界：每边仅 1 个新种子家族（n=2 家族）；comp3 的 glob 臂
本次为现场跑（2042+500/2092+500）非复用；sign 侧 618 单种子。

## 用法

```bash
conda activate ffbm
python bci/comp3/acquire.py --arms comp_tr,glob_tr --sessions 1 \
    --seed0 2042,2092 --outdir bci/seedchk1/outputs
python bci/comp3/acquire.py --arms comp_ctl,glob_ctl --sessions 1 \
    --seed0 2042,2092 --outdir bci/seedchk1/outputs
python bci/comp3/analyze.py --outdir bci/seedchk1/outputs
```

汇总：bci/seedchk1/outputs/summary.json。
