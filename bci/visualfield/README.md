# bci/visualfield：单眼视野闪烁（2 类）

新几何模式 `eye_map: "half_split"`（export_data）：每个复眼只采样
画面**自己的一半**（右叶→左半帧、左叶→右半帧，几何镜像沿用
per_eye 的验证框架）。刺激 vf_left / vf_right = 半帧 10 Hz 方波
闪烁（另半帧恒灰），类别 = 被驱动的眼。区域集纯 visual_bilateral
（缓存命中）。

## 结果（12 试次，17 通道空间模式 LOO + 置换）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 受刺激眼 L/R | **100%** | **0.0035** | 50%（perm95 83%）|

**解读**：单眼视觉驱动的偏侧签名完全可解码——为"视野位置"类
范式（偏侧注意的刺激通道、双眼竞争的结构基础）打开了刺激维度。
与 hybrid 的发现对照：单眼闪烁的**空间**签名强（100%），而频率
标签弱（被中央背景掩没）——空间模式是这个模型里最稳健的视觉
读出轴。

## 用法

```bash
conda activate ffbm
python bci/visualfield/acquire.py
python bci/visualfield/analyze.py
```
