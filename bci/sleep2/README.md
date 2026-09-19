# bci/sleep2 — 上行态维持机制的鉴定（睡眠探针后续）

机制 14 遗留问题：sz_l 点火后 10.5 µV 上行态平台**与 CEN_C 的
STD 门控无关**——维持电流不走（被阻尼的）复发通路。本研究用
`--gain-scale` 逐个零化候选通路做归因：

| 臂 | 零化 | 检验 |
|---|---|---|
| base | — | 平台基线 |
| noCEN_C | CEN_C ×0.02 | 复发（已预期无关，对照）|
| noCEN_R | CEN_R ×0.02 | OLR→CEN 前馈 |
| noOLR_V | OLR_V ×0.02 | 视觉核→OLR |
| noCEN_V | CEN_V ×0.02 | 视觉核→CEN 直通 |
| noORN_C | ORN_C ×0.02 | ORN→CEN |

读出：20 s 试次末 10 s 平台的头皮 rms（原观测变量）+ 分群速率
（--pop-rate：哪些 class 在平台上活跃）。**维持者=零化后平台
塌缩的通路**；全都不塌 → 维持是分布式的，问题升级。

## 用法

```bash
conda activate ffbm
python bci/sleep2/acquire.py     # 6 × 20 s
python bci/sleep2/analyze.py
```
