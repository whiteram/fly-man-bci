# 超参数注册表（自动生成，勿手编）

来源：`src/ffbm/params.py`——运行 `python -m ffbm.params` 再生成。
状态含义：数据集（MaleCNS 直接给出）｜文献（文献值）｜校准（第 4 轮扫描拟合）｜现象学（对测得波形特征手工构造）｜假设（工作假设）｜数值（算法选择，已验证）｜选定（设计选择）｜待定（开放项）。

## 电路（双侧视叶五层级联，exp015）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| lobes | both | L/R | 选定 | exp015 起双叶接入（原生几何、同侧连线；跨叶边=0 实测）——右眼 R1-6 追踪不完整（1,112 vs 左 2,265），受支配 L1/2 突触质量中位数两侧一致（148 vs 153），工作点直接迁移 |
| lobe_split_iters | 20 | iterations | 数值 | x 轴 2-means 收敛轮数（20 轮内必然收敛） |
| rhabd_offset_um | 23.5 | um | 校准 | 光感受器偶极长度（小网膜方向外推，exp002 对 ERG 标定）；双叶各自朝本侧眼（镜像） |

## LIF 神经元（量级取自果蝇文献，未逐细胞拟合）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| v_rest | -70.0 | mV | 文献 | 全层统一 |
| v_th | -50.0 | mV | 文献 | 全层统一（阈值-静息 20 mV） |
| v_reset | -70.0 | mV | 文献 | 等于静息 |
| r_in | 0.1 | GOhm | 文献 | 全层统一输入电阻；注意它把 pA 噪声换算成 mV 波动 |
| tau_ms.R | 10.0 | ms | 文献 | |
| tau_ms.L | 20.0 | ms | 文献 | |
| tau_ms.MID | 10.0 | ms | 文献 | |
| tau_ms.T45 | 10.0 | ms | 文献 | |
| t_refrac_ms.R | 3.0 | ms | 文献 | |
| t_refrac_ms.L | 2.0 | ms | 文献 | |
| t_refrac_ms.MID | 2.0 | ms | 文献 | |
| t_refrac_ms.T45 | 2.0 | ms | 文献 | |

> tau/refrac 为文献量级；放电率可信度由工作点校准兜底，而非逐细胞参数。

## 光转导（现象学：双级低通 + 慢适应）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| tau_stage_ms | 10.0 | ms | 现象学 | 每级 10 ms，两级共 ~20 ms 感受器潜伏期（exp002 强度系列 / exp004 闪烁融合对形） |
| tau_adapt_ms | 800.0 | ms | 现象学 | 慢亮度适应时间常数 |
| adapt_sag | 0.3 | fraction | 现象学 | 光平台期 30% 下垂（对 ERG 平台/瞬变比） |

> 非对称变体（tau_fall=2.5 ms，exp003）已实现但主管线未启用——exp012b 证明它在 exp009 符号下不改变 off 瞬变。

## 突触（RL/LM 电流基；MT 电导基）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| gain_rl_pa_per_syn | 12.0 | pA | 校准 | R→L 每突触每脉冲电流峰值（符号来自数据集组胺 −1） |
| gain_lm_pa_per_syn | 20.0 | pA | 校准 | L→Mi/Tm 每突触每脉冲（符号来自数据集） |
| tau_rl_ms | 5.0 | ms | 文献 | R→L 突触电流衰减 |
| tau_lm_ms | 5.0 | ms | 文献 | L→Mi/Tm |
| g_unit_mt_ns | 0.02 | nS | 校准 | MT 每突触峰值电导；在 v=−70 mV 时等效 1.4 pA/突触 |
| e_rev_exc_mv | 0.0 | mV | 文献 | 乙酰胆碱类 |
| e_rev_inh_mv | -80.0 | mV | 假设 | 抑制性翻转电位（GABA/GluCl 文献范围 −70~−85 的中值；影响抑制强度 ~±20%） |
| mt_tau_s_ms.Mi1 | 25.0 | ms | 现象学 | |
| mt_tau_s_ms.Tm3 | 8.0 | ms | 现象学 | |
| mt_tau_s_ms.Mi4 | 10.0 | ms | 现象学 | |
| mt_tau_s_ms.Mi9 | 15.0 | ms | 现象学 | |
| mt_tau_s_ms.Tm1 | 20.0 | ms | 现象学 | |
| mt_tau_s_ms.Tm2 | 8.0 | ms | 现象学 | |
| mt_tau_s_ms.Tm4 | 10.0 | ms | 现象学 | |
| mt_tau_s_ms.Tm9 | 20.0 | ms | 现象学 | |

> MT 分化动力学（exp005 KINETICS.differentiated）：为方向选择性涌现设计的每类型时间常数；宽为 ±5 ms 时DS 显著减弱（exp006）。

## 工作点（第 4 轮校准，score 0.0）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| i_r_base_pa | 150.0 | pA | 校准 | R 基线电流（黑暗期由 OU 噪声驱动 ~2.7 Hz） |
| i_l_base_pa | 250.0 | pA | 校准 | LMC 暗去极化基流（exp009 的 ON/OFF 架构支柱；机制正确版 = inverted-Cl 组胺电导，deferred） |
| i_mid_base_pa | 90.0 | pA | 校准 | Mi/Tm 基流（黑暗期 19 Hz） |
| i_t4_base_pa | 175.0 | pA | 校准 | T4 未建模输入背景（黑暗期 7 Hz；见 §6 结构发现） |
| ou_sigma_pa.R | 50.0 | pA | 校准 | |
| ou_sigma_pa.L | 40.0 | pA | 校准 | |
| ou_sigma_pa.MID | 40.0 | pA | 校准 | |
| ou_sigma_pa.T45 | 60.0 | pA | 校准 | |
| ou_tau_ms | 8.0 | ms | 假设 | OU 相关时间 ≈ 突触时间尺度；膜波动 2.7–4 mV 的目标由此达成，tau 在 5–12 ms 内结果不敏感 |

> 全部黑暗/闪光期放电率同时落文献窗（R 2.7 / L 12.5 / MID 19.0 / T4 7.0 / T5 3.0 Hz）。敏感性：I_*_BASE ±10 pA 约改变对应层黑暗率数 Hz（平滑梯度，非刀锋）。

## 传导/突触延迟

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| syn_delay_ms | 1.0 | ms | 文献 | 化学突触固定延迟 |
| v_axon_um_per_ms | 300.0 | um/ms | 文献 | 0.3 m/s，小轴突传导速度 |
| jitter_ms | 0.5 | ms | 假设 | ±均匀抖动；影响微小（实测校准点不变） |

## 视觉下游 VPN→中央脑（exp016 可选层，现象学工作点）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| vpn_classes | ['LC17', 'LC12', 'LC10a', 'LC10d', 'LC11', 'LC18', 'LPLC2', 'LLPC1', 'LC9', 'LC16'] | - | 选定 | CB 突触量前 10 的 VPN 类（承担 3.17M VPN→CB 突触中的 1.35M；全部胆碱能；跨度 326-561 um） |
| g_unit_m2v_ns | 0.02 | nS | 现象学 | MID/T45→VPN 单突触电导（全部 ACh 兴奋性） |
| g_unit_v2c_ns | 0.02 | nS | 现象学 | VPN→CB 单突触电导（全部 ACh 兴奋性） |
| tau_v_ms | 8.0 | ms | 现象学 | VPN 一级突触动力学 |
| i_v_base_pa | 60.0 | pA | 校准 | VPN 未建模背景基流（暗率 1-10 Hz 窗口内） |
| i_c_base_pa | 60.0 | pA | 校准 | 中央脑目标细胞基流（同窗口） |
| ou_sigma_vpn_cb_pa | 40.0 | pA | 现象学 | VPN/CB 的 OU 噪声（与 MID 同量级） |
| positions | soma | - | 假设 | VPN/CB 用胞体位置近似突触位置（syn-points 13 GB 未做子集扫描；偶极长度因此保守偏短） |

> 电路为可选扩展：仅当 circuit 带 vpn_ids/cb_ids 键时build_stack 才构建这两层（exp016 评估用，未进主管线导出）。

## 人脑四球壳 + 电极（思想实验几何）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| scale | 400.0 | x | 选定 | 网络放大倍数标称值（exp010 定：单叶铺满大脑）；exp015 起按网络实际半径自适应 scale = min(400, 0.98·r_脑/r_max)——双叶原生几何 692 um 在 ×400 下宽 277 mm 装不进脑球，自动降到 ~×213（原生相对几何保持不变） |
| radii_mm | [78.0, 80.0, 85.0, 92.0] | mm | 文献 | 脑/CSF/颅骨/头皮半径（成人文献值；±1 mm 只动幅值 ±1%） |
| sigmas_sm | [0.33, 1.79, 0.013, 0.33] | S/m | 文献 | 各层电导率；±20% 的幅值包络 ×0.82–1.06（scripts/head_sensitivity.py） |
| elec_rel_radius | 0.985 | of r_scalp | 选定 | 电极深度（贴表面差 1.7%，可忽略） |
| occipital_pole_frac | 0.9 | of r_brain | 选定 | T4/T5 极贴近球壁的程度 |
| occipital_margin_frac | 0.98 | of r_brain | 选定 | 安全边距（实际余量 1.5 mm） |
| n_elec | 17 | count | 选定 | 头皮电极数（10-20 系统量级） |
| kernel_n_terms | 60 | terms | 数值 | 勒让德截断；网络深度处误差 2.5e-10 |

> **DEFERRED（当前最大不确定项）**：回流电流几何——三候选（t-bar 邻近 / 神经突跨区 / 均值位置）核范数差 ~400×，exp012/012b 未能判别（需先做 LMC 动力学/inverted-Cl 电导）。绝对幅值因此未标定；相对量（波形/拓扑/相位）不受影响。

## 背景人脑 EEG 与检测论

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| alpha_hz | 10.0 | Hz | 文献 | 闭眼 α 峰（后区主导） |
| alpha_amp_uv | 30.0 | uV | 文献 | 枕区峰值量级（最不利：与信号同侧） |
| envelope_period_ms | 700.0 | ms | 选定 | α 涨落包络半周期 |
| aperiodic_uv | 3.0 | uV | 文献 | 1/f 成分（谱指数 1） |
| sensor_uv | 1.5 | uV | 文献 | 放大器白噪声 |
| snr_band_hz | [2.0, 20.0] | Hz | 选定 | 带限 SNR 分析窗（避开 α 峰的经典 EEG 带） |
| dprime_target | 2.0 | sd | 选定 | 检测阈值 |

> 当前结果：宽带 842 / 带限 3019 次试验（d'=2）。频带选择定量重要：果蝇信号大量能量在 >20 Hz。

## 已知的开放超参数（数值待定）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| return_current_geometry | neurite | - | 校准 | 已闭合（exp013 终扫）：突触后回流沿 L 细胞远端突起分布；在文献一致点（暗 −38.8mV、深度 20.6mV）off/on 锚点仅 neurite 通过（25 配置统计 neurite 8 过 / 其余 0）。绝对幅值标定 = 主管线值 ×1.7 |
| lmc_e_cl_mv | -70.0 | mV | 校准 | 已由文献反解闭合（exp013）：E_Cl≈−70mV 常规极性 + 去极化漏 v_K≈−25mV；详见 lamina_mechanistic 节 |
| r16_phototransduction_heterogeneity | None | - | 待定 | R1-6 个体差异（低优先） |

## 机制板层（exp013，文献反解；主管线采纳待末轮校准）

| 参数 | 值 | 单位 | 状态 | 说明 |
|---|---|---|---|---|
| e_cl_mv | -70.0 | mV | 校准 | 组胺氯通道翻转电位：由暗 −38.4mV + 光深度 10–25mV + 释放比 ~4 双电导稳态反解（常规极性；'inverted-Cl' 说法适用于其他昆虫中间神经元，不适用 L1/2） |
| v_k_lmc_mv | -25.0 | mV | 校准 | LMC 漏通道翻转电位（去极化侧），同一反解 |
| l_dark_target_mv | -38.4 | mV | 文献 | L1/2 暗静息（Rusanen & Weckström 2016: −38.4 ± 3.2 mV） |
| light_depth_window_mv | [10.0, 25.0] | mV | 文献 | 光反应深度窗（Laughlin/Hardie 经典值） |
| g_unit_hist_ns | 0.16 | nS | 校准 | 每突触单位组胺电导峰值；第 5 轮按主管线（左叶）突触质量重定标（exp013 双叶电路为 0.25——电导与每细胞突触量成反比） |
| l_release_map_mv | [-58.0, -30.0] | mV | 校准 | L 输出端释放率映射 v_L→r_L（第 5 轮校准对象）；暗态 L（−38.8mV）释放 ~0.64，亮态（超极化）趋 0 |
| g_unit_lm_ns | 0.04 | nS | 校准 | L→Mi/Tm 分级电导（第 5 轮校准；E_rev 按数据集符号 0/−80mV） |
| release_map_mv | [-59.0, -25.0] | mV | 校准 | v_R→释放率线性映射 [lo, hi]；暗紧张释放约 0.1–0.2 |

> exp013 已验证机制（暗电位/超极化方向/off 瞬变可达）；主管线采纳 = R 分级 + 组胺分级释放 + L 分级 + L→Mi/Tm 分级释放，替换 exp009 双反号代理与 I_L_BASE hack，需第 5 轮校准与 ON/OFF 复验。
