# 模型-生理对照验证表（v1）

目的：把模型的关键动力学量与公开的果蝇在体生理数据并排——匹配
的确认、不匹配的如实标注并给机制解释。所有模型侧度量来自本仓库
的实际运行（方法列）；文献值为在体电生理/成像（引用列）。

| # | 量 | 模型 | 文献在体 | 判定 | 方法/引用 |
|---|---|---|---|---|---|
| 1 | ERG 闪光时程 | 闪光→R1-R6 级联+ lamina LIF→眼面电位，时程吻合 ERG 波形结构 | ERG 经典波形 | ✅（exp001 建模即以此为目标）| exp001 README |
| 2 | KC 自发发放率 | **0 Hz**（暗场 4.5s × 4,064 KC，elig 稳态探针灵敏度 <5×10⁻⁴ Hz/细胞）| **~0.004 Hz**（蛾在体，n=117；果蝇"近可忽略"）——比 PN 低 ~2000× | ✅ **匹配**（两者实际静默）| 稳态 elig = 率×tau（`--plastic-mb` 全窗开，hyb_n 工作点）；[Nature Neurosci 2008](https://www.nature.com/articles/nn.2192)、[J Neurophysiol 2008](https://journals.physiology.org/doi/full/10.1152/jn.01283.2007)、[Neuron 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970) |
| 3 | KC 气味诱发率 | 驱动版 ~78-125 Hz（elig 反推）；真实气味通路（al-gain 0.01）~78 Hz max | **稀疏：数 Hz，多数 KC 无响应**（每气味 ~5-10 尖峰）| ❌ **偏高 1-2 个量级** | 稳态 elig 反推（condit/condit3 校准运行）；[Honegger 2011](https://www.jneurosci.org/content/33/25/10568)、[Cell Reports 2017](https://www.sciencedirect.science/article/pii/S0896627317305639) |
| 4 | 嗅觉响应起始延迟 | 头皮起始 **126 ms**（气味脉冲后 3σ 判据），峰 678 ms | PN 响应"数十 ms"量级起始、快速上升并适应（强调气味起始/导数）| ✅ 量级一致；峰时程=适应包络（定性一致 "rise and accommodate"）| bci/olfactory da1_r0 事后分析；[Bhandawat 2007](https://pmc.ncbi.nlm.nih.gov)、[Kim 2015](https://pmc.ncbi.nlm.nih.gov) |
| 5 | PN 层弱信号放大 | 未单独记录（PN 在 CEN 大群内）| 弱 ORN 输入在 PN 层放大、强的不放大 | ⏸ 待仪器化（PN 分群速率记录）| Bhandawat 2007 |

## 不匹配项的机制解读（#3）

模型 KC 诱发率偏高的根因与 APL 弧线的发现自洽：真实 KC 的稀疏
化依赖 (a) APL/GABA 能分流抑制、(b) 树突非线性（爪状输入的阈
上整合）——两者都不在点 LIF + 指数电导的实现里。APL 反馈环独立
增益化的四点表征（见 bci/condit3 README）显示加性反馈无法重建
除性归一化——**同一缺失的两种表现**： KC 率偏高与 APL 手术
无法稳定读出腿。

## 诚实边界

- 头皮信号由外周（ORN/感觉末梢）主导——中央速率类度量无法从
  头皮反推，需分群速率记录仪器化（列为后续）；
- elig 稳态探针给的是突触前率的上包络（max 边）；均值的反推
  需按 w 衰减积分，未做（避免过度推算）；
- 文献值跨物种（蛾/果蝇）混用处在 #2 已标注；量级比较为主。

## 复现

```bash
# KC 自发率（elig 探针）
python viz/export_data.py --regions visual_bilateral,ol_rest,\
central_brain,olfactory --visual-input dark --chem-input hyb_n \
  --plastic-mb --plastic-window 500,5000 --plastic-lr 3e-6 \
  --t-end 5000 --seed 901 --out <tmp> --gpu   # -> "elig max 0.0000"
# 嗅觉起始延迟：bci/olfactory 输出 + 3σ 判据（见本表方法列）
```
