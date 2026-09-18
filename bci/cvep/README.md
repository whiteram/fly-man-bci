# bci/cvep：编码视觉诱发电位（c-VEP，BETA 范式对齐）

对齐清华 [BETA 数据集](https://arxiv.org/abs/1911.13045)（Liu et al.
2021，40 目标 m-序列编码，[公开下载](http://bci.med.tsinghua.edu.cn/download.html)）。
与 SSVEP 的区别：目标身份编码在**码形**（伪随机序列的响应形状）里而
非频率线上——40 个目标同时闪烁也互不干扰（m-序列循环自相关 = δ 函数）。

## 码本（make_code.py）

- 基础序列：127 位 m-序列（度 7 本原多项式 LFSR；平衡 64/63 与满周期
  断言保证；循环自相关峰值 127、旁瓣全 −1——教科书性质，逐项验证）；
- 40 目标 = 循环移位 3k 位（127/40 ≈ 3.2，近最大间隔的标准 c-VEP 间距）；
- 亮度 = BASE ± AMP × 码位，60 fps 采样；BETA 对码做带限调制，
  这里由光转导级联提供带限（记录的偏离，同 SSVEP 分支惯例）；
- 时间线（3.0 s）：0.5 s 基准导入 → 1 个完整码周期（2.117 s，解码窗）
  → 0.4 s 收尾。
- 偏离记录：单目标全屏驱动（每眼整幅映射，不对齐 40 槽位空间显示、
  不建模注意）——与 SSVEP 分支同一"attend-slot analog"惯例。

## 结果（40 目标 × 2 重复，干净信号）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 8 目标试点 | 100% | 0.0005 | 12%（perm95 25%）|
| **40 目标全量** | **100%** | **0.0005** | 2.5%（perm95 6%）|

**40 类全可分**——码形信息在模拟头皮 EEG 上完整保留。解码 = 模板
匹配（BETA 式）：17 通道 × 码周期展平 → 留一模板最大 Pearson 相关
→ 2000 次标签置换（置换在试次-试次相关矩阵上进行，数学上与朴素
实现恒等，秒级完成）。

与 BETA 人类数据的对照留作后续：人类 40 类 c-VEP 典型 single-trial
准确率 ~70–90%（依数据集处理），本模拟在干净信号前提下为上界；
对照 exp018 缩放设备噪声的版本是下一步。

## 用法

```bash
conda activate ffbm
python bci/cvep/make_code.py --all                          # 40 刺激
python bci/cvep/acquire.py --targets 0 5 10 15 20 25 30 35 --repeats 2  # 试点
python bci/cvep/acquire.py --all40 --repeats 2              # 全量（~70 s/试次）
python bci/cvep/analyze.py
```
