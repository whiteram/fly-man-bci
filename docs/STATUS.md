# 项目进度总览（STATUS）

更新：2026-09。承接 HANDOFF.md（交接/环境/数据）与 docs/TECHNICAL.md
（方法学细节）；本文回答"现在做到哪了"。

---

## 1. 一句话现状

果蝇视觉系统（MaleCNS v1.0 连接组，区域可选装配）在自然视觉刺激下的
群体放电仿真已全链路打通，并经四球壳人脑前向核投影为头皮脑电，配有
完整 3D 可视化、交互式标定体系和多套电极配置；刺激支持合成协议与
**真实视频输入**（灰度→小网膜采样映射，`visual_inputs.json` 目录 +
页面同步双视图），其他模态的扩展路径见 docs/STIMULUS.md §6。

## 2. 已完成的仿真阶段

| 阶段 | 状态 | 产出 | 代码/实验 |
|---|---|---|---|
| 数据管线（连接组下载/子网/注释） | ✅ | 视觉子网 + 注释 + 递质 + 坐标 | scripts/, docs/data.md |
| ERG 前向（光转导→R→板层→眼表面电位） | ✅ | 与实测 ERG 波形对标 | exp001–002 |
| ON/OFF 不对称、运动 gratings | ✅ | 方向响应基线 | exp003–004 |
| T4/T5 方向选择机制化 | ✅ | Hex 轴 DS 电路（T4/T5 preferred −135°） | exp005–006, exp013 |
| 头内"果蝇 EEG" + 导体模型 | ✅ | 密封二球核 SealedHeadPairField | exp007–008 |
| 板层极性/电流模式修正 | ✅ | RL/LM 电流基 + 机制化 LMC | exp009, exp013 |
| 人比例思想实验（×400 装入四球壳） | ✅ | FourSpherePairField + 自适应 scale | exp010 |
| 刺激检测论（信号 vs 背景脑电） | ✅ | T7 d′≈2（约 46 次试验） | exp011 |
| 返回约束/保守性检验 | ✅ | — | exp012 |
| 双侧视叶 + VPN→中央脑 + 全 CNS 装配 | ✅ | 区域可选装配（regions 开关）；全 CNS 150,601 神经元仿真（全部 Traced∩有 soma 的三大超类，覆盖口径见 HANDOFF §4.3） | exp014–017, regions.py |
| 头皮电极阵列扩展 | ✅ | 45/64/128/EGI241 四套配置（elec_configs.json） | make_elec_configs.py |
| **计算加速：numba + GPU + 分级缓存** | ✅ | numba 内核 2.38×；`--gpu` 循环 47 min→156 s（轨迹逐位一致）；电路/前向核双缓存（换电极配置只重建核） | docs/ACCELERATION_PLAN.md P2/P3/P5 |
| 交互标定体系 | ✅ | 10-20 手柄标定（ni/th/yaw/roll）、视角标定、人头形态校准 | viz/index.html |
| 可视化页面 | ✅ | 电极名称标签、热图、时段色条、配置切换、预览波形 | viz/index.html |
| **头皮脑电 EDA（分布/通道差异/刺激可解码性）** | ✅ | 干净信号带通 0.5-6 Hz 解码 CV R²=0.84、级联滞后 23/17 ms、空间参与秩≈2；频带分布 δ 主导（0.1-4 Hz 占 72%，无 α 峰，AC 幅度比人脑背景小 ~130×）；前提=果蝇替代人脑（仅设备噪声）：0.1-1 Hz + 主成分空间滤波单段 SNR≈1.2，~3 段平均到 SNR 2 | exp018 |
| **BCI 范式采集分支（SSVEP-Benchmark）** | ✅ | 对齐 Wang 2016（40 类 8-15.8 Hz）；多种子 40 类 × 5 重复：CCA top-1 干净 72% / 缩放假想设备噪声 83%（随机 2.5%，与人类基准同量级）；平均不提精度——瓶颈是 0.2 Hz 类间距可分性而非噪声；SSVEP 频段响应带内 SNR ~27 dB——低频主导是刺激属性而非模型属性 | bci/ssvep_benchmark |
| **BCI 范式：运动想象 / 嗅觉 / 听觉** | ✅ | `motor_imagery`（2 类，复用 exp020 MI-v2）：想象窗空间 LOO 100%（p=0.003）；`olfactory`（3 气味身份，exp019 电路，ORN 群大小补偿幅度）：3 类 89%（p=0.0005，DA1/DM2/VA1v 空间模式可分）；`auditory`（新区域 `audition`：JO-A 115→JOA_C 1,578 对/24,392 突触→中央脑；求偶歌包络建模）：4 类 54%（p=0.007）/ 侧别 75% / 模式 75%——类间总 rms 相同，可分性全在空间模式；`cvep`（对齐 BETA，127 位 m-序列 40 循环移位码）：**40 类模板匹配 100%（p=0.0005，chance 2.5%）** | bci/*, exp021, docs/SENSORY_INPUTS.md §4-5 |
| **刺激：真实图像输入** | ✅ | 通用视觉输入模块：`visual_inputs.json` 目录 + 灰度视频→小网膜采样（模糊/归一化/每眼整幅眼平面映射）+ 页面同步双视图 | docs/STIMULUS.md §5 |
| **刺激：化学感觉（嗅觉 + 味觉）** | ✅ 嗅觉验证通过 / 味觉外周通过 | 新区域 `olfactory`/`gustatory`（ORN 2,639 / GRN 1,428，连边 ORN_C 57k 对→触角叶、GRN_C 8.4k 对→SEZ）；`--chem-input` 脉冲协议；ORN/GRN 转导级联（共享视觉动力学）；标定发现：中央脑默认增益是点火吸引子（82 Hz 自持），chem 工作点 CEN_C→0.002 后触角叶特异响应可衰减；**嗅觉头皮签名 ×1.4–1.65**；味觉因 GRN→VNC 大头不投影，签名弱（×1.04，800 pA 阴性）| exp019, docs/SENSORY_INPUTS.md |
| **刺激：本体感觉 + 运动意图解码** | ✅ 完成（v1+v2） | 新区域 `proprioception`（PRO 1,454，rootSide L/R 分组；PRO_V→VNC 47.9k 对、PRO_C→中央 4.8k 对）；MI-v1：指令窗可解码（99–100%，0.3% 背景水平、p=0.002，纯 PRO_C 末梢偶极子），延迟窗无保持（首版带通 78–89% 是 filtfilt 长尾伪影）；**MI-v2：DN 通路探针**（DN 输入 392 万突触 96% 已在电路：cb_intrinsic/LAL 61%、DN→DN 13%、上升 AN 13%；PRO 直连仅 0.27%）→ 想象窗内部驱动（LAL* 路由 1200 pA / DN 直驱 800 pA 对照，各 12 试次）：**想象窗全程可解码**（路由 +0.001~0.002 µV p≤0.013；直驱 −0.005~−0.008 µV p≤0.002，谷氨酸能极性反转），无点火 | exp020, docs/SENSORY_INPUTS.md §4 |
| **刺激：其他模态（听觉）** | ⏸ 规划 | 听觉 JO+AMMC 数据在、社区空白 | docs/SENSORY_INPUTS.md §5 |
| 64/128/241 导精确波形重导出 | ⏸ 待定稿 | GPU+缓存后每配置约 3-8 min（首次核构建为主；EGI241 最慢），命令见 USAGE §6 | — |

## 3. 超参数：写在哪里、怎么改

| 位置 | 内容 | 修改方式 |
|---|---|---|
| `src/ffbm/params.py`（SECTIONS 注册表） | 全部模型超参：LIF、突触、光转导、MT、工作点、VPN/中央脑、人脑四球壳、**刺激协议**（时段表/对比度/漂移速度/种子等） | 直接编辑；`python -m ffbm.params` 重新生成 docs/PARAMS.md（有测试保证文档不过期） |
| `params.cal()` | 运行时 CAL 字典（键名即覆盖入口，按次序解析） | 程序化覆盖：`cal = dict(fp.CAL); cal["I_R_BASE"]=...` |
| `viz/data/elec_configs.json` | 电极配置目录（45/64/128/EGI241 + 自定义） | `viz/make_elec_configs.py` 重新生成，或手工追加 `kind:"custom"` 条目 |
| `viz/data/elec_layout_1010.json` | 当前导出用的 45 导布局 | `viz/make_elec_layout.py`（消费标定面板导出的 JSON） |
| 浏览器 localStorage | 页面三类标定的工作状态（calib1020 / viewDirs / headFormCalib_v3）+ 电极编辑草稿 | 页面滑块/手柄直接调；「复制JSON」可导出 |
| `viz/export_data.py` CLI | `--elec-layout`、`--regions` | 命令行 |

状态标签含义（PARAMS.md 每行第三列）：dataset=数据集直接给出｜
literature=文献值｜calibrated=对实测波形拟合｜phenomenol.=现象学构造｜
assumed=工作假设｜numerical=算法选择（已验证）｜chosen=设计选择｜
deferred=开放项。

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| HANDOFF.md | 交接：环境恢复、数据事实、API 参考、bug 史、领域调研 |
| docs/TECHNICAL.md | 方法学：连接组→电路→仿真→前向核→背景 EEG 的全部数学 |
| docs/PARAMS.md | 超参数注册表（自动生成，勿手编） |
| docs/STIMULUS.md | 刺激协议：实现、参数、真实图像可行性 |
| docs/USAGE.md | 本手册：页面操作、标定工作流、导出流程 |
| docs/data.md | 数据文件说明 |
| viz/ELEC_CONFIGS.md | 电极配置文件格式 + 四套配置说明 |

## 5. 已知限制

- 刺激仅视觉：合成协议（1-D 纹理线扫描 + 全场闪烁）与真实视频输入
  均已实现（visual_inputs.json 目录）；真实视频作为正式"实验范式"
  使用前建议核对 docs/STIMULUS.md 的映射限制说明（归一化统计口径等）。
- R7/R8（色觉）未装配；色觉输入需先补小网膜 R7/R8 电路。
- 颈部 VNC 只参与动力学、不投影到头皮（位于放大头球之外，物理上
  正确）。
- 64/128/241 导目前为插值预览波形；精确导出待定稿后执行（§2 末行）。
- EGI 256 是厂商测地几何（E 编号无解剖含义），与规则推导的 10-x 帽
  不完全重合（预期行为）。
