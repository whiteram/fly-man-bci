"""Hyperparameter registry: every fixed number in the pipeline, with
provenance, uncertainty and sensitivity. Single source of truth for
ffbm.pipeline.CAL and ffbm.vizprep background defaults; regenerates
docs/PARAMS.md so the config and its documentation cannot drift.

Status vocabulary:
  dataset      directly from MaleCNS v1.0 (peer-reviewed, most reliable)
  literature   standard value from published work (fly or human EEG)
  calibrated   fitted by scripts/calibrate_working_point.py (round 4,
               score 0.0 = all metrics in literature windows)
  phenomenol.  hand-built to reproduce a measured waveform feature
  assumed      working assumption, untested / untestable here
  numerical    algorithmic choice, validated (convergence/cross-checks)
  deferred     known-open item; value not yet determined (see notes)

Regenerate the documentation:   python -m ffbm.params
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for downstream imports)

# name: (value, unit, status, note)
SECTIONS: dict = {
    "circuit": {
        "_title": "电路（区域可选装配，exp015 起）",
        "lobes": ("both", "L/R", "chosen",
                  "exp015 起双叶接入（原生几何、同侧连线；跨叶边=0 实测）"
                  "——右眼 R1-6 追踪不完整（1,112 vs 左 2,265），受支配 "
                  "L1/2 突触质量中位数两侧一致（148 vs 153），工作点直接迁移"),
        "regions_default": ({"visual_bilateral": True, "vpn_central": False},
                            "-", "chosen",
                            "ffbm.regions 区域开关默认值：每个脑区独立启停，"
                            "关闭的区域不构建种群/突触/前向核（零计算零内存）。"
                            "已知区域见 ffbm.regions.known_regions()；"
                            "导出可用 --regions 覆盖"),
        "lobe_split_iters": (20, "iterations", "numerical",
                             "x 轴 2-means 收敛轮数（20 轮内必然收敛）"),
        "rhabd_offset_um": (23.5, "um", "calibrated",
                            "光感受器偶极长度（小网膜方向外推，exp002 对 ERG 标定）；"
                            "双叶各自朝本侧眼（镜像）"),
    },
    "neuron_lif": {
        "_title": "LIF 神经元（量级取自果蝇文献，未逐细胞拟合）",
        "v_rest": (-70.0, "mV", "literature", "全层统一"),
        "v_th": (-50.0, "mV", "literature", "全层统一（阈值-静息 20 mV）"),
        "v_reset": (-70.0, "mV", "literature", "等于静息"),
        "r_in": (0.1, "GOhm", "literature",
                 "全层统一输入电阻；注意它把 pA 噪声换算成 mV 波动"),
        "_status": "literature",
        "tau_ms": {"R": (10.0, "ms"), "L": (20.0, "ms"),
                   "MID": (10.0, "ms"), "T45": (10.0, "ms")},
        "t_refrac_ms": {"R": (3.0, "ms"), "L": (2.0, "ms"),
                        "MID": (2.0, "ms"), "T45": (2.0, "ms")},
        "_note_tau": "tau/refrac 为文献量级；放电率可信度由工作点校准兜底，"
                     "而非逐细胞参数。",
    },
    "phototransduction": {
        "_title": "光转导（现象学：双级低通 + 慢适应）",
        "tau_stage_ms": (10.0, "ms", "phenomenol.",
                         "每级 10 ms，两级共 ~20 ms 感受器潜伏期"
                         "（exp002 强度系列 / exp004 闪烁融合对形）"),
        "tau_adapt_ms": (800.0, "ms", "phenomenol.", "慢亮度适应时间常数"),
        "adapt_sag": (0.30, "fraction", "phenomenol.",
                      "光平台期 30% 下垂（对 ERG 平台/瞬变比）"),
        "_note": "非对称变体（tau_fall=2.5 ms，exp003）已实现但主管线未启用——"
                 "exp012b 证明它在 exp009 符号下不改变 off 瞬变。",
    },
    "synapses": {
        "_title": "突触（RL/LM 电流基；MT 电导基）",
        "gain_rl_pa_per_syn": (12.0, "pA", "calibrated",
                               "R→L 每突触每脉冲电流峰值（符号来自数据集组胺 −1）"),
        "gain_lm_pa_per_syn": (20.0, "pA", "calibrated",
                               "L→Mi/Tm 每突触每脉冲（符号来自数据集）"),
        "tau_rl_ms": (5.0, "ms", "literature", "R→L 突触电流衰减"),
        "tau_lm_ms": (5.0, "ms", "literature", "L→Mi/Tm"),
        "g_unit_mt_ns": (0.02, "nS", "calibrated",
                         "MT 每突触峰值电导；在 v=−70 mV 时等效 1.4 pA/突触"),
        "e_rev_exc_mv": (0.0, "mV", "literature", "乙酰胆碱类"),
        "e_rev_inh_mv": (-80.0, "mV", "assumed",
                         "抑制性翻转电位（GABA/GluCl 文献范围 −70~−85 的中值；"
                         "影响抑制强度 ~±20%）"),
        "_status": "phenomenol.",
        "mt_tau_s_ms": {"Mi1": (25.0, "ms"), "Tm3": (8.0, "ms"),
                        "Mi4": (10.0, "ms"), "Mi9": (15.0, "ms"),
                        "Tm1": (20.0, "ms"), "Tm2": (8.0, "ms"),
                        "Tm4": (10.0, "ms"), "Tm9": (20.0, "ms")},
        "_note_mt": "MT 分化动力学（exp005 KINETICS.differentiated）："
                    "为方向选择性涌现设计的每类型时间常数；宽为 ±5 ms 时"
                    "DS 显著减弱（exp006）。",
    },
    "working_point": {
        "_title": "工作点（第 4 轮校准，score 0.0）",
        "i_r_base_pa": (150.0, "pA", "calibrated", "R 基线电流（黑暗期由 OU 噪声驱动 ~2.7 Hz）"),
        "i_l_base_pa": (250.0, "pA", "calibrated",
                        "LMC 暗去极化基流（exp009 的 ON/OFF 架构支柱；"
                        "机制正确版 = inverted-Cl 组胺电导，deferred）"),
        "i_mid_base_pa": (90.0, "pA", "calibrated",
                          "Mi/Tm 基流（黑暗期 19 Hz）"),
        "i_t4_base_pa": (175.0, "pA", "calibrated",
                         "T4 未建模输入背景（黑暗期 7 Hz；见 §6 结构发现）"),
        "_status": "calibrated",
        "ou_sigma_pa": {"R": (50.0, "pA"), "L": (40.0, "pA"),
                        "MID": (40.0, "pA"), "T45": (60.0, "pA")},
        "ou_tau_ms": (8.0, "ms", "assumed",
                      "OU 相关时间 ≈ 突触时间尺度；膜波动 2.7–4 mV 的目标"
                      "由此达成，tau 在 5–12 ms 内结果不敏感"),
        "_note": "全部黑暗/闪光期放电率同时落文献窗（R 2.7 / L 12.5 / "
                 "MID 19.0 / T4 7.0 / T5 3.0 Hz）。敏感性：I_*_BASE ±10 pA "
                 "约改变对应层黑暗率数 Hz（平滑梯度，非刀锋）。",
    },
    "delays": {
        "_title": "传导/突触延迟",
        "syn_delay_ms": (1.0, "ms", "literature", "化学突触固定延迟"),
        "v_axon_um_per_ms": (300.0, "um/ms", "literature",
                             "0.3 m/s，小轴突传导速度"),
        "jitter_ms": (0.5, "ms", "assumed", "±均匀抖动；影响微小（实测校准点不变）"),
    },
    "vpn_central": {
        "_title": "扩展脑区（exp016 VPN / exp017 全 CNS，可选层）",
        "_status": "phenomenol.",
        "vpn_classes": (["LC17", "LC12", "LC10a", "LC10d", "LC11", "LC18",
                         "LPLC2", "LLPC1", "LC9", "LC16"], "-", "chosen",
                        "CB 突触量前 10 的 VPN 类（承担 3.17M VPN→CB "
                        "突触中的 1.35M；全部胆碱能；跨度 326-561 um）"),
        "g_unit_m2v_ns": (0.02, "nS", "phenomenol.",
                          "MID/T45→VPN 单突触电导（全部 ACh 兴奋性）"),
        "g_unit_v2c_ns": (0.02, "nS", "phenomenol.",
                          "VPN→CB 单突触电导（全部 ACh 兴奋性）"),
        "g_unit_cx_ns": (0.004, "nS", "phenomenol.",
                         "exp017 扩展区域（ol_rest/central/vnc）单突触"
                         "电导；复发环路增益保守取小，防爆发作"),
        "tau_v_ms": (8.0, "ms", "phenomenol.", "扩展区域一级突触动力学"),
        "i_v_base_pa": (120.0, "pA", "calibrated",
                        "VPN 未建模背景基流（暗率窗口 1-10 Hz）"),
        "i_c_base_pa": (60.0, "pA", "calibrated",
                        "exp016 中央脑目标细胞基流（同窗口）"),
        "i_olr_base_pa": (60.0, "pA", "calibrated",
                          "exp017 其余视叶基流（同窗口）"),
        "i_cen_base_pa": (60.0, "pA", "calibrated",
                          "exp017 中央脑（含 VPN/下行）基流（同窗口）"),
        "i_vnc_base_pa": (60.0, "pA", "calibrated",
                          "exp017 腹索 VNC 基流（同窗口）"),
        "i_orn_base_pa": (60.0, "pA", "assumed",
                          "exp019 嗅觉受体神经元自发电流（新鲜空气 ORN "
                          "自发放；与 CEN 同工作点窗口）"),
        "i_grn_base_pa": (60.0, "pA", "assumed",
                          "exp019 味觉受体神经元基流（静息；同窗口）"),
        "i_pro_base_pa": (60.0, "pA", "assumed",
                          "exp020 本体感受神经元基流（静息肢体；同窗口）"),
        "i_tc_base_pa": (60.0, "pA", "assumed",
                          "exp022 体表触觉感受神经元基流（静息；同窗口）"),
        "i_joa_base_pa": (60.0, "pA", "assumed",
                          "exp021 Johnston 器听觉神经元基流（静息空气；"
                          "同窗口）"),
        "i_th_base_pa": (60.0, "pA", "assumed",
                         "exp023 温度感受神经元基流（静息温度；同窗口）"),
        "i_hy_base_pa": (60.0, "pA", "assumed",
                         "exp023 湿度感受神经元基流（静息湿度；同窗口）"),
        "ou_sigma_vpn_cb_pa": (40.0, "pA", "phenomenol.",
                               "扩展区域的 OU 噪声（与 MID 同量级）"),
        "sign_mapping": ("consensus_nt", "-", "assumed",
                          "ACh→+1；GABA/Glu→−1；多巴胺/5-HT/章鱼胺/"
                          "未知→+1（多数为兴奋/调质，文档化为假设）"),
        "positions": ("soma", "-", "assumed",
                      "扩展区域用胞体位置近似突触位置（syn-points 13 GB "
                      "未做子集扫描；偶极长度因此保守偏短）"),
        "_note": "区域为可选扩展：由 ffbm.regions 的区域开关装配，开启时"
                 "电路带 extra_pops/extra_edges 规范、build_stack 才构建"
                 "这些层。VNC 组标记 forward=False（参与仿真、不进头壳"
                 "前向核——腹索在 ×N 下位于头球之外）。",
    },
    "stimulus": {
        "_title": "自然刺激协议（viz/export_data.py 消费）",
        "seed": (42, "int", "chosen",
                 "1/f 噪声与纹理生成的随机种子"),
        "t_epochs": ([("dark", 0.0, 1500.0), ("flicker", 1500.0, 4500.0),
                      ("drift+", 4500.0, 6000.0), ("drift-", 6000.0, 7500.0),
                      ("band+", 7500.0, 9000.0), ("band-", 9000.0, 10500.0)],
                     "ms", "chosen",
                     "自然刺激时段表 (名称, 起, 止)：黑暗基线 → 全场 1/f "
                     "闪烁 → 1/f 纹理沿 T4/T5 偏好方向漂移 → 时间反演 → "
                     "DS 尺度带通纹理 → 其反演；页面时段色条同步此表"),
        "stim_contrast": (2.0, "x", "chosen",
                          "亮度调制对比度（1 基底上的乘性幅度；亮度下限 "
                          "0.05 防负值）"),
        "i_lum": (150.0, "pA", "calibrated",
                  "亮度增量 → R 光电流的整体增益（exp002 强度系列 / "
                  "exp004 闪烁融合的标定工作点）"),
        "drift_speed": (0.1, "um/ms", "chosen",
                        "纹理漂移速度（沿 T4/T5 偏好方向 e_ds，六角轴回归）"),
        "lam_um": (22.0, "um", "literature",
                   "板层柱间距——DS 运动检测的特征空间尺度（带通纹理的"
                   "通带 1/45–1/15 um⁻¹ 由此设定）"),
    },
    "head_model": {
        "_title": "人脑四球壳 + 电极（思想实验几何）",
        "scale": (400.0, "x", "chosen",
                  "网络放大倍数标称值（exp010 定：单叶铺满大脑）；exp015 起"
                  "按网络实际半径自适应 scale = min(400, 0.98·r_脑/r_max)"
                  "——双叶原生几何 692 um 在 ×400 下宽 277 mm 装不进脑球，"
                  "自动降到 ~×213（原生相对几何保持不变）"),
        "radii_mm": ([78.0, 80.0, 85.0, 92.0], "mm", "literature",
                     "脑/CSF/颅骨/头皮半径（成人文献值；±1 mm 只动幅值 ±1%）"),
        "sigmas_sm": ([0.33, 1.79, 0.013, 0.33], "S/m", "literature",
                      "各层电导率；±20% 的幅值包络 ×0.82–1.06"
                      "（scripts/head_sensitivity.py）"),
        "elec_rel_radius": (0.985, "of r_scalp", "chosen",
                            "电极深度（贴表面差 1.7%，可忽略）"),
        "occipital_pole_frac": (0.90, "of r_brain", "chosen",
                                "T4/T5 极贴近球壁的程度"),
        "occipital_margin_frac": (0.98, "of r_brain", "chosen",
                                  "安全边距（实际余量 1.5 mm）"),
        "n_elec": (17, "count", "chosen", "头皮电极数（10-20 系统量级）"),
        "elec_face_excl_deg": (38.0, "deg", "chosen",
                               "电极面孔排除锥（相对面轴的最小角；排除眼/鼻/嘴，"
                               "保留额极 Fpz 一带的帽区——55° 会把额区最佳电极"
                               "也排除掉，d'=2 试验数从 ~128 涨到 ~3784）"),
        "elec_neck_excl_deg": (40.0, "deg", "chosen",
                               "电极颈部排除锥（相对颈轴的最小角；电极不落在"
                               "耳下颈部）"),
        "head_form_calib": ({"scale": 3.2, "rotX": 0.0, "rotY": -17.0,
                             "rotZ": 3.0, "offF_mm": 77.0, "offU_mm": 160.0,
                             "offR_mm": -48.0}, "-", "calibrated",
                             "人头形态（LeePerrySmith 扫描，CC-BY）手动校准"
                             "超参数（2026-09-14 用户目测标定）：缩放相对"
                             "种子（模型宽=脑球直径）；旋转为世界轴外旋"
                             "（俯仰/偏航/翻滚）；平移沿面/上/右轴 mm。"
                             "旋转与缩放以颅腔中心（耳点中点）为支点"),
        "kernel_n_terms": (60, "terms", "numerical",
                           "勒让德截断；网络深度处误差 2.5e-10"),
        "_note_deferred": "**DEFERRED（当前最大不确定项）**：回流电流几何——"
                          "三候选（t-bar 邻近 / 神经突跨区 / 均值位置）核范数差 "
                          "~400×，exp012/012b 未能判别（需先做 LMC 动力学/"
                          "inverted-Cl 电导）。绝对幅值因此未标定；相对量"
                          "（波形/拓扑/相位）不受影响。",
    },
    "background_eeg": {
        "_title": "背景人脑 EEG 与检测论",
        "alpha_hz": (10.0, "Hz", "literature", "闭眼 α 峰（后区主导）"),
        "alpha_amp_uv": (30.0, "uV", "literature", "枕区峰值量级（最不利：与信号同侧）"),
        "envelope_period_ms": (700.0, "ms", "chosen", "α 涨落包络半周期"),
        "aperiodic_uv": (3.0, "uV", "literature", "1/f 成分（谱指数 1）"),
        "sensor_uv": (1.5, "uV", "literature", "放大器白噪声"),
        "snr_band_hz": ([2.0, 20.0], "Hz", "chosen",
                        "带限 SNR 分析窗（避开 α 峰的经典 EEG 带）"),
        "dprime_target": (2.0, "sd", "chosen", "检测阈值"),
        "_note": "当前结果：宽带 842 / 带限 3019 次试验（d'=2）。频带选择"
                 "定量重要：果蝇信号大量能量在 >20 Hz。",
    },
    "open_items": {
        "_title": "已知的开放超参数（数值待定）",
        "return_current_geometry": ("neurite", "-", "calibrated",
                                    "已闭合（exp013 终扫）：突触后回流沿 L 细胞远端"
                                    "突起分布；在文献一致点（暗 −38.8mV、深度 "
                                    "20.6mV）off/on 锚点仅 neurite 通过（25 配置"
                                    "统计 neurite 8 过 / 其余 0）。绝对幅值标定 "
                                    "= 主管线值 ×1.7"),
        "lmc_e_cl_mv": (-70.0, "mV", "calibrated",
                        "已由文献反解闭合（exp013）：E_Cl≈−70mV 常规极性 + "
                        "去极化漏 v_K≈−25mV；详见 lamina_mechanistic 节"),
        "r16_phototransduction_heterogeneity": (None, "-", "deferred",
                                                "R1-6 个体差异（低优先）"),
    },
    "lamina_mechanistic": {
        "_title": "机制板层（exp013，文献反解；主管线采纳待末轮校准）",
        "e_cl_mv": (-70.0, "mV", "calibrated",
                    "组胺氯通道翻转电位：由暗 −38.4mV + 光深度 10–25mV + "
                    "释放比 ~4 双电导稳态反解（常规极性；'inverted-Cl' "
                    "说法适用于其他昆虫中间神经元，不适用 L1/2）"),
        "v_k_lmc_mv": (-25.0, "mV", "calibrated",
                       "LMC 漏通道翻转电位（去极化侧），同一反解"),
        "l_dark_target_mv": (-38.4, "mV", "literature",
                             "L1/2 暗静息（Rusanen & Weckström 2016: "
                             "−38.4 ± 3.2 mV）"),
        "light_depth_window_mv": ([10.0, 25.0], "mV", "literature",
                                  "光反应深度窗（Laughlin/Hardie 经典值）"),
        "g_unit_hist_ns": (0.16, "nS", "calibrated",
                           "每突触单位组胺电导峰值；第 5 轮按主管线（左叶）"
                           "突触质量重定标（exp013 双叶电路为 0.25——电导与每"
                           "细胞突触量成反比）"),
        "l_release_map_mv": ([-58.0, -30.0], "mV", "calibrated",
                             "L 输出端释放率映射 v_L→r_L（第 5 轮校准对象）；"
                             "暗态 L（−38.8mV）释放 ~0.64，亮态（超极化）趋 0"),
        "g_unit_lm_ns": (0.04, "nS", "calibrated",
                         "L→Mi/Tm 分级电导（第 5 轮校准；E_rev 按数据集符号 "
                         "0/−80mV）"),
        "release_map_mv": ([-59.0, -25.0], "mV", "calibrated",
                           "v_R→释放率线性映射 [lo, hi]；暗紧张释放约 0.1–0.2"),
        "_note": "exp013 已验证机制（暗电位/超极化方向/off 瞬变可达）；"
                 "主管线采纳 = R 分级 + 组胺分级释放 + L 分级 + L→Mi/Tm "
                 "分级释放，替换 exp009 双反号代理与 I_L_BASE hack，"
                 "需第 5 轮校准与 ON/OFF 复验。",
    },
}


def cal() -> dict:
    """Assemble the pipeline CAL dict from the registry (single source)."""
    wp = SECTIONS["working_point"]
    lam = SECTIONS["lamina_mechanistic"]
    lif = SECTIONS["neuron_lif"]
    syn = SECTIONS["synapses"]
    dly = SECTIONS["delays"]
    vpn = SECTIONS["vpn_central"]
    return {
        "I_R_BASE": wp["i_r_base_pa"][0],
        "I_L_BASE": wp["i_l_base_pa"][0],
        "I_MID_BASE": wp["i_mid_base_pa"][0],
        "I_T4_BASE": wp["i_t4_base_pa"][0],
        "GAIN_RL": syn["gain_rl_pa_per_syn"][0],
        "GAIN_LM": syn["gain_lm_pa_per_syn"][0],
        "TAU_RL": syn["tau_rl_ms"][0],
        "TAU_LM": syn["tau_lm_ms"][0],
        "G_UNIT_MT": syn["g_unit_mt_ns"][0],
        "E_REV_EXC": syn["e_rev_exc_mv"][0],
        "E_REV_INH": syn["e_rev_inh_mv"][0],
        "OU": {k: v[0] for k, v in wp["ou_sigma_pa"].items()},
        "OU_TAU_MS": wp["ou_tau_ms"][0],
        "SYN_DELAY_MS": dly["syn_delay_ms"][0],
        "V_AXON_UM_PER_MS": dly["v_axon_um_per_ms"][0],
        "DELAY_JITTER_MS": dly["jitter_ms"][0],
        "LIF": {k: (lif["tau_ms"][k][0], lif["t_refrac_ms"][k][0])
                for k in ("R", "L", "MID", "T45")},
        "RIN_GOHM": lif["r_in"][0],
        "MID_TAU_S": {k: v[0] for k, v in syn["mt_tau_s_ms"].items()},
        "LAMINA_MECHANISTIC": True,
        "E_CL_MV": lam["e_cl_mv"][0],
        "V_K_LMC_MV": lam["v_k_lmc_mv"][0],
        "G_UNIT_HIST_NS": lam["g_unit_hist_ns"][0],
        "R_RELEASE_MAP_MV": list(lam["release_map_mv"][0]),
        "L_RELEASE_MAP_MV": list(lam["l_release_map_mv"][0]),
        "G_UNIT_LM_NS": lam["g_unit_lm_ns"][0],
        # exp016/017 optional regions (built only when the region
        # switches in ffbm.regions are on)
        "VPN_CLASSES": list(vpn["vpn_classes"][0]),
        "G_UNIT_M2V": vpn["g_unit_m2v_ns"][0],
        "G_UNIT_V2C": vpn["g_unit_v2c_ns"][0],
        "G_UNIT_CX": vpn["g_unit_cx_ns"][0],
        "TAU_V_MS": vpn["tau_v_ms"][0],
        "I_V_BASE": vpn["i_v_base_pa"][0],
        "I_C_BASE": vpn["i_c_base_pa"][0],
        "I_OLR_BASE": vpn["i_olr_base_pa"][0],
        "I_CEN_BASE": vpn["i_cen_base_pa"][0],
        "I_VNC_BASE": vpn["i_vnc_base_pa"][0],
        "I_ORN_BASE": vpn["i_orn_base_pa"][0],
        "I_GRN_BASE": vpn["i_grn_base_pa"][0],
        "I_PRO_BASE": vpn["i_pro_base_pa"][0],
        "I_JOA_BASE": vpn["i_joa_base_pa"][0],
        "I_TC_BASE": vpn["i_tc_base_pa"][0],
        "I_TH_BASE": vpn["i_th_base_pa"][0],
        "I_HY_BASE": vpn["i_hy_base_pa"][0],
        "OU_VPN_CB": vpn["ou_sigma_vpn_cb_pa"][0],
    }


def bg_defaults() -> dict:
    b = SECTIONS["background_eeg"]
    return {"alpha_hz": b["alpha_hz"][0],
            "alpha_amp_uv": b["alpha_amp_uv"][0],
            "aperiodic_uv": b["aperiodic_uv"][0],
            "sensor_uv": b["sensor_uv"][0],
            "envelope_period_ms": b["envelope_period_ms"][0],
            "seed": 2026}


def elec_defaults() -> dict:
    h = SECTIONS["head_model"]
    return {"face_excl_deg": h["elec_face_excl_deg"][0],
            "neck_excl_deg": h["elec_neck_excl_deg"][0]}


_STATUS_ZH = {"dataset": "数据集", "literature": "文献", "calibrated": "校准",
              "phenomenol.": "现象学", "assumed": "假设", "numerical": "数值",
              "chosen": "选定", "deferred": "待定"}


def to_markdown() -> str:
    lines = ["# 超参数注册表（自动生成，勿手编）", "",
             "来源：`src/ffbm/params.py`——运行 `python -m ffbm.params` 再生成。",
             "状态含义：数据集（MaleCNS 直接给出）｜文献（文献值）｜校准"
             "（第 4 轮扫描拟合）｜现象学（对测得波形特征手工构造）｜假设"
             "（工作假设）｜数值（算法选择，已验证）｜选定（设计选择）｜"
             "待定（开放项）。", ""]
    for key, sec in SECTIONS.items():
        lines.append(f"## {sec.get('_title', key)}")
        lines.append("")
        lines.append("| 参数 | 值 | 单位 | 状态 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for k, v in sec.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    lines.append(f"| {k}.{k2} | {v2[0]} | {v2[1]} | "
                                 f"{_STATUS_ZH.get(sec_status(sec), '')} | |")
            else:
                val, unit, status, note = v
                lines.append(f"| {k} | {val} | {unit} | "
                             f"{_STATUS_ZH.get(status, status)} | {note} |")
        for k, v in sec.items():
            if k.startswith("_note") or k.startswith("_title"):
                if k != "_title":
                    lines.append("")
                    lines.append(f"> {v}")
        lines.append("")
    return "\n".join(lines)


def sec_status(sec) -> str:
    return sec.get("_status", "literature")


def main():
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "docs" / "PARAMS.md"
    out.write_text(to_markdown(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
