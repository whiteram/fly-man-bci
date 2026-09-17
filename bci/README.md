# BCI 范式数据采集（模拟）分支

本目录是**脑机接口范式**的模拟采集分支：把项目当作"被试"（放大
×400 的果蝇视觉网络 + 四球壳头模型），按**公开脑电数据集的真实范式**
生成刺激、跑模拟、产出逐试次头皮脑电，并用该范式的标准检测算法评估
"能否从模拟脑电中检测到刺激相关响应"。

与 `experiments/` 的分工：
- `experiments/expNNN_*`——科学问题驱动的单次实验（一次性分析）；
- `bci/<范式>/`——**可重复的采集会话**：刺激目录 + 采集驱动 + 检测
  分析，对齐某个公开数据集的范式，之后可直接拿真实数据集对比同一
  分析代码。

## 范式子目录

| 目录 | 对齐数据集 | 状态 |
|---|---|---|
| `ssvep_benchmark/` | Tsinghua SSVEP-Benchmark（Wang et al. 2016，40 目标 8–15.8 Hz Δ0.2） | ✅ 频率编码试点 |
| `motor_imagery/` | 运动想象 L/R 保持解码（BCI 竞赛 MI 范式思路；复用 exp020 MI-v2 协议） | ✅ 2 类 |
| `olfactory/` | 气味身份 3 类解码（DA1/DM2/VA1v；exp019 电路） | ✅ 3 类 |
| `auditory/` | 求偶歌侧别×模式 4 类解码（exp021 JOA 电路） | ✅ 4 类 |

后续可加：`p300/`（oddball）、`n170/` 等。

## 每个范式目录的固定结构

```
bci/<paradigm>/
  README.md        范式说明、与公开数据集的异同表
  make_stimulus.py 生成该范式的刺激帧序列（.npy，进 visual_inputs.json 目录）
  acquire.py       采集驱动：逐试次调 export_data.py，收集每试次头皮电位
  analyze.py       该范式的标准检测/分类算法（SSVEP=CCA），出指标与图
  outputs/         采集数据（eeg_f<freq>.npy）与结果图
```

## 为什么这样设计（用户决策记录）

1. **对齐公开数据集的范式**，而不是自造刺激——分析代码今后可直接
   换上真实数据（同一 CCA、同一指标），模拟与实测同表对比；
2. 采集复用主管线（电路缓存 + 前向核缓存 + GPU 环路），每个试次
   只换刺激亮度函数，电路/核零重建；
3. `phi_scalp_all_uV` / `_debug_phi_scalp.npy` 是**干净**信号（背景
   另存）——采集数据默认干净，噪声情景由分析端按前提叠加（与
   exp018 的前提结论一致：项目默认=替代，仅设备噪声）。

## 背景脑电参数的文献依据（回应"文献参数有无引用"）

`src/ffbm/params.py` 的 background_eeg 组标注 "literature"，取的是
教科书典型值；此前未钉具体出处，正式参考：

- **α 峰 10 Hz、枕区主导、闭眼 20–50 µV**：Niedermeyer's
  Electroencephalography: Basic Principles, Clinical Applications, and
  Related Fields（经典教科书章节 "The Normal EEG"）；Bickford, *EEG
  in Clinical Practice* 同量级
- **1/f 无周期分量（谱指数 ~1，头皮 µV 量级）**：He, Zempel, Snyder
  & Raichle 2010, *Neuron* 66:353–369（"The temporal structures and
  functional significance of scale-free brain activity"）；
  Dehghani et al. 2016, *J Neurophysiol*（头皮/皮层 1/f 对比）
- **放大器输入噪声 ~1–2 µV rms（临床级）**：临床 EEG 放大器数据手册
  量级（如 Natus/Cadwell 规格）；研究级 DC 放大器可 <0.5 µV

注意：这些值定义的是"反事实共存情景"（若与完好人脑共存的叠加背景）；
**项目默认前提是替代**——神经源性背景不存在，只有设备噪声（exp018
§7 的前提扫描）。SSVEP 检测默认在干净信号与设备噪声两种条件下评估。

## 低频主导是否有意义？——SSVEP 就是检验

exp018 发现模拟信号 δ 主导（0.1–4 Hz 占 72%）。这是**刺激的属性**
（demo 视频只有慢运动）还是**模型的属性**（级联只能传慢信号）？
SSVEP 频闪直接回答：若模型对 8–15.8 Hz 频闪产生可检测的稳态响应，
则高频有意义信号存在，低频主导只是当前刺激太慢。果蝇光转导-板层
通路的带宽（数十 Hz，文献已知）预测响应存在——试点验证之。
