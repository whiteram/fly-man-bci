# 项目进度总览（STATUS）

更新：2026-09。承接 HANDOFF.md（交接/环境/数据）与 docs/TECHNICAL.md
（方法学细节）；本文回答"现在做到哪了"。

---

## 1. 一句话现状

果蝇视觉系统（MaleCNS v1.0 连接组，区域可选装配）在自然视觉刺激下的
群体放电仿真已全链路打通，并经四球壳人脑前向核投影为头皮脑电，配有
完整 3D 可视化、交互式标定体系和多套电极配置；刺激当前为合成视觉
协议，接收真实图像的改造路径已明确（docs/STIMULUS.md §5）。

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
| 双侧视叶 + VPN→中央脑 + 全 CNS 装配 | ✅ | 区域可选装配（regions 开关）；全脑 111k 神经元仿真 | exp014–017, regions.py |
| 头皮电极阵列扩展 | ✅ | 45/64/128/EGI241 四套配置（elec_configs.json） | make_elec_configs.py |
| 交互标定体系 | ✅ | 10-20 手柄标定（ni/th/yaw/roll）、视角标定、人头形态校准 | viz/index.html |
| 可视化页面 | ✅ | 电极名称标签、热图、时段色条、配置切换、预览波形 | viz/index.html |
| **刺激：真实图像输入** | ✅ | 通用视觉输入模块：`visual_inputs.json` 目录 + 灰度视频→小网膜采样（模糊/归一化/眼平面仿射）+ 页面同步双视图 | docs/STIMULUS.md §5 |
| **刺激：其他模态（嗅觉等）** | ⏸ 未开工 | 区域开关机制已就绪 | docs/STIMULUS.md §6 |
| 64/128/241 导精确波形重导出 | ⏸ 待定稿 | 耗时 13/25/45 min，命令见 USAGE §6 | — |

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

- 刺激仅视觉（1-D 纹理线扫描 + 全场闪烁）；真实图像输入是下一步
  最小改动（docs/STIMULUS.md §5）。
- R7/R8（色觉）未装配；色觉输入需先补小网膜 R7/R8 电路。
- 颈部 VNC 只参与动力学、不投影到头皮（位于放大头球之外，物理上
  正确）。
- 64/128/241 导目前为插值预览波形；精确导出待定稿后执行（§2 末行）。
- EGI 256 是厂商测地几何（E 编号无解剖含义），与规则推导的 10-x 帽
  不完全重合（预期行为）。
