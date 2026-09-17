# bci/auditory：求偶歌解码（侧别 × 歌模式，4 类）

果蝇 BCI 的听觉范式。数据侧：Johnston 器听觉亚群 **JO-A 115 个**
（subclass `auditory`，rootSide L 63/R 52；26 个无突触前末梢记录的
被排除，保留 89），末梢进脑构成 **JOA_C 1,578 对 / 24,392 突触**
（目标 SAD001/CB1076/SAD051/GNG636/AMMC0xx 等，均在 central_brain
区域，forward=True）。电路构建器
`experiments/exp021_audition/circuit6.py`（区域名 `audition`，
exp019 ORN 同款模式）。

## 范式

每试次 3.6 s，类别 4 类（6 重复）：**300–2000 ms 在一侧 JO-A 群
（`*@L` / `*@R`，600 pA）播放求偶歌幅度包络**——pulse song =
25 Hz 包络方波（20/20 ms），sine song = 持续平台。载波 150–300 Hz
远超 LIF 群跟随能力（tau 10 ms），**按包络编码**（文档化建模选择），
经共享 PhotoCascade 转导级联。

## 结果（24 试次，干净信号，17 维空间模式 LOO 最近质心）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 4 类（侧别×模式） | **54%** | **0.007** | 25%（perm 95 分位 42%）|
| 侧别 L/R | **75%** | **0.027** | 50% |
| 模式 pulse/sine | **75%** | **0.025** | 50% |

- 类间**总 rms 几乎相同**（1.3933–1.3967 µV）——可分性完全来自
  17 通道空间模式，不是幅度；这与"单边 JO-A 群 → 偏侧偶极子 +
  包络形状→不同 CEN 募集"的机制一致。
- 侧别可分性不对称的解剖学原因：保留的 JO-A 中 L 62 / R 27
  （rootSide），R 侧群只有一半细胞。
- 观察到一个待标定现象：600 pA 驱动后 rms 平台在驱动结束后**不回落**
  （1.393 → 1.392 持平到试次结束）——CEN 在 JOA 输入下的弱双稳态/
  持续态候选，类间共模、不影响解码，但工作点增益扫描列为后续。

## 用法

```bash
conda activate ffbm
python bci/auditory/acquire.py     # 4 类 × 6 重复（~115 s/试次）
python bci/auditory/analyze.py
```
