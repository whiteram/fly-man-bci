# SSVEP-Benchmark 范式（对齐 Wang et al. 2016）

对齐数据集：**Tsinghua SSVEP-Benchmark**（Wang, Chen, Gao & Gao 2016,
IEEE TBME 25(10):1746–1752；35 名被试、40 目标 8–15.8 Hz Δ0.2 Hz、
联合频率-相位编码、60 Hz 刷新率、6 s 试次）。

```bibtex
@article{wang2016benchmark,
  title={A benchmark dataset for SSVEP-based brain--computer interfaces},
  author={Wang, Yijun and Chen, Xiaogang and Gao, Xiaorong and Gao, Shangkai},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={25}, number={10}, pages={1746--1752}, year={2016}, publisher={IEEE}
}
```

## 与人类基准的异同

| 维度 | 人类基准 | 本模拟 | 说明 |
|---|---|---|---|
| 目标数 | 40 字符 5×8 | 40 槽 5×8 | 一致 |
| 频率 | 8–15.8 Hz Δ0.2 | 同 | 一致 |
| 编码 | 联合频率-相位 | 列内相位错开 0.5 s×列 | 简化 |
| 波形 | 方波亮灭 | 正弦亮度调制 | 频谱更干净 |
| 试次时长 | 6 s | 10.5 s | 管线协议长度 |
| "注意" | 被试注视一个目标 | 单目标试次=只有该槽闪烁 | 无注意机制，用刺激侧替代 |
| 电极 | 64 导 | 45 导（10-20 系） | 近似对齐 |
| 被试 | 35 人 | 1 个"果蝇被试" | 多试次=多频率，暂无个体差异 |

## 试点结果（2026-09-13，visual_bilateral 规模，GPU）

**试点 1：all-40 同时闪烁**（每个试次含全部 40 频率；检验"SSVEP 频段
响应是否存在且可分"）：

- 干净信号带内 SNR（闪烁频率处功率 vs 邻域）：**中位 22.6 dB**
  （最小 12.2，最大 30.3）——**8–15.8 Hz 稳态响应强烈存在**
- 这直接回答了 exp018 的遗留问题：模拟 EEG 的 δ 主导（0.1–4 Hz 占
  72%）是**演示视频太慢的属性，不是模型的属性**——果蝇级联对 8–16 Hz
  频闪忠实地跟随并传到头皮，频带分布完全由刺激决定
- 加 1.5 µV 设备白噪（替代前提）后单试次带内 SNR 降到 ~0.6 dB——
  与 exp018 §7 一致，恢复靠平均/空间滤波

**试点 2：单目标（注视模拟）**：`acquire.py --freqs ...` 生成"只有
一个槽闪烁"的试次，`analyze.py` 用标准 CCA 检测器做 6 类分类
（正弦/余弦 1–3 次谐波参考，SVD 求典型相关）：

- 干净信号 top-1 **4/6 = 67%**（随机 17%）；带内 SNR 中位 25.0 dB
- 混淆集中在频率相邻对（15.2 vs 15.8 仅差 0.6 Hz）——全 40 类
  Δ0.2 Hz 需更长窗口/试次平均，这是后续工作
- +1.5 µV 设备白噪单试次 2/6=33%：与 exp018 §7 的 SNR≈0.6 dB 一致，
  实用检测需 k 试次平均（√k 增益）
- 结论：**"果蝇被试"具有可检测的 SSVEP**——与人类 BCI 用户一样，
  其头皮信号携带频率标记的稳态视觉响应，标准 BCI 解码算法可用

## 复现

```bash
conda activate ffbm
python bci/ssvep_benchmark/acquire.py --freqs 8.0 9.8 11.6 13.4 15.2 15.8
python bci/ssvep_benchmark/analyze.py
# 全量 40 频率（约 50 min）：
python bci/ssvep_benchmark/acquire.py --all40
```

（全量运行在 acquire.py 中加 `--all40` 即可循环 40 频率。）

## 下一步

- 全 40 类 + 相位编码 + k 试次平均下的检测曲线（信噪比 vs 平均数）
- 把同一 `analyze.py` 直接跑 Tsinghua 真实数据集（下载后换数据加载层），
  模拟 vs 实测同表对比
- 背景条件扩展：+α/1/f（反事实共存情景）、+空间滤波预处理
