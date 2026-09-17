# bci/motor_imagery：运动想象保持解码（2 类）

把 exp020 的 MI-v2 协议固化为可重复的 BCI 采集会话。每试次 3.6 s，
类别 = 侧别（L/R），6 重复：

- **300–1200 ms 指令窗**：一侧腿本体感受器 400 pA 复反馈（意向运动
  的感知副本）；
- **1300–3500 ms 想象窗**：同侧 LAL* 群 1200 pA 内部驱动（含
  PhotoCascade 800 ms 适配补偿，稳态 360 pA），经**真实 LAL→DN 突触**
  保持运动计划层活动——无任何外周输入。

复用 exp020 的模拟运行（`--refetch` 可重跑 export_data）。

## 结果（6+6 试次，干净信号）

| 窗口 | L−R (µV) | perm-p |
|---|---|---|
| cue | +0.0023 | 0.002 |
| imagery 1300–2000 | +0.0021 | 0.003 |
| imagery 2000–2700 | +0.0010 | 0.013 |
| imagery 2700–3500 | +0.0011 | 0.005 |

**想象窗空间模式 LOO 最近质心：100%（perm-p=0.003，chance 50%）**
——"无执行、自上而下保持"的运动想象状态可在整个想象窗从头皮解码。

与人类 MI-BCI 的对应：指令=提示，想象窗=运动想象期，解码目标=想象
内容的侧别。诚实声明：效应为背景的 0.1–1%，干净信号前提下显著；
对照 exp018 缩放设备噪声需大量平均。

## 用法

```bash
conda activate ffbm
python bci/motor_imagery/acquire.py     # 复制 exp020 运行（秒级）
python bci/motor_imagery/analyze.py
```

通路依据见 `experiments/exp020_proprioception/dn_probe.py`（DN 输入
392 万突触：LAL/cb_intrinsic 61%、PRO 直连 0.27%）与 exp020 README。
