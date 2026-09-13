"""Experiment 009: data-driven synaptic signs through the lamina — the ON/OFF
functional architecture.

The dataset's neurotransmitter annotations give, for every edge, the correct
sign, and they overturn exp001-008's lamina wiring:

  R1-R6 -> L1/L2/L3  histamine, inhibitory (-1)   [was excitatory]
  L1 -> Mi1/Tm3/Mi4  glutamate, inhibitory (-1)   [was unsigned / +]
  L2 -> Tm1/2/4/Mi9  acetylcholine, excitatory (+1)
  L3 -> Mi9/Tm9/...  acetylcholine, excitatory (+1)

With the classic LMC dark-depolarized state (endogenous base drive; light
removes it via R's histamine), this double inversion reproduces the textbook
ON/OFF split without any hand-coded polarity:

  light -> R fires -> inhibits L1 -> L1 stops inhibiting Mi1/Tm3
        -> disinhibition -> T4 ON pathway
  light -> R fires -> inhibits L2 -> Tm1/2/4 lose excitation
  off    -> L2 returns to dark state -> Tm's re-driven -> T5 OFF pathway

Protocol: flash (500-3500 ms) at 1x/2x, left-lobe cascade, rates per cell
type (dark / on / off windows), eye+lamina+medulla electrodes under the
homogeneous and sealed-head kernels, ERG window metrics.

Run from repository root:
    python experiments/exp009_lamina_polarity/run.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp005_medulla_ds"))

from ffbm.forward import SealedHeadPairField, StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation

import run as exp005

OUT = Path(__file__).resolve().parent / "outputs"

DT = exp005.DT
T_END = 4500.0
FLASH_ON, FLASH_OFF = 500.0, 3500.0
LEVELS_PA = (350.0, 700.0)
SEED = 42

# operating-point drives: the classic LMC dark-depolarized state (all three
# L types) and an endogenous Mi/Tm drive held in check by L1's inhibition
I_L_BASE = 250.0      # pA -> ~25 Hz dark L rate
I_MID_BASE = 220.0    # pA -> fires when disinhibited, silenced when inhibited
GAIN_RL = 12.0         # pA per synapse (sign from data: histamine, -1)
GAIN_LM = 20.0
GAIN_MT = 5.0
TAU_RL, TAU_LM = 5.0, 5.0
KIN = exp005.KINETICS["differentiated"]

RATE_TYPES = ("R", "L1", "L2", "L3", "Mi1", "Tm3", "Mi9", "Tm1", "Tm2",
              "Tm4", "Tm9", "T4", "T5")


def lockout_metrics(phi_nV, t_ms):
    """ERG window metrics (baseline / on-peak / plateau / off-peak)."""
    m = lambda t0, t1: (t_ms >= t0) & (t_ms < t1)
    base = float(np.median(phi_nV[m(100, FLASH_ON - 100)]))
    on = float(phi_nV[m(FLASH_ON, FLASH_ON + 200)].min())
    plateau = float(np.median(phi_nV[m(FLASH_ON + 500, FLASH_OFF - 200)]))
    off_win = phi_nV[m(FLASH_OFF, FLASH_OFF + 400)]
    off = float(off_win.max())
    return {"baseline": round(base, 1), "peak_on": round(on, 1),
            "plateau": round(plateau, 1), "peak_off": round(off, 1),
            "off_overshoot": round(off - base, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = exp005.build_circuit()
    pp, qq = circuit["pre_pos"], circuit["post_pos"]

    r_ids = circuit["r_ids"]
    l_ids = circuit["l_ids"]
    mid_ids = circuit["mid_ids"]
    t45_ids = circuit["t45_ids"]
    n_r, n_l, n_mid, n_t45 = (len(r_ids), len(l_ids), len(mid_ids),
                              len(t45_ids))
    r_type = circuit["r_type"]
    l_type = circuit["l_type"]
    mid_type = circuit["mid_type"]
    t45_type = circuit["t45_type"]

    l_index = np.full(int(l_ids.max()) + 1, -1, dtype=np.int64)
    l_index[l_ids] = np.arange(n_l)
    mid_index = np.full(int(mid_ids.max()) + 1, -1, dtype=np.int64)
    mid_index[mid_ids] = np.arange(n_mid)
    t45_index = np.full(int(t45_ids.max()) + 1, -1, dtype=np.int64)
    t45_index[t45_ids] = np.arange(n_t45)
    type_masks = {
        "R": np.ones(n_r, dtype=bool),
        **{ty: l_type == ty for ty in ("L1", "L2", "L3")},
        **{ty: mid_type == ty for ty in
           ("Mi1", "Tm3", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9")},
        "T4": np.array([str(s).startswith("T4") for s in t45_type]),
        "T5": np.array([str(s).startswith("T5") for s in t45_type]),
    }

    # edges with DATA signs: R->L histamine(-), L->MID per-nt, MT as before
    e_rl = circuit["e_rl"].copy()
    e_rl["weight"] = e_rl["weight"] * e_rl["sign"]        # all -1 (histamine)
    e_lm = circuit["e_lm"].copy()
    e_lm["weight"] = e_lm["weight"] * e_lm["sign"]        # L1 -, L2/L3 +
    e_mt = circuit["e_mt"]

    r_pos = np.array([pp[b] for b in r_ids])
    l_pos = np.array([qq[b] for b in l_ids])
    t45_pos = np.array([pp[b] for b in t45_ids])
    u_eye = r_pos.mean(axis=0) - t45_pos.mean(axis=0)
    u_eye /= np.linalg.norm(u_eye)

    eye_elec = (r_pos.mean(axis=0)
                + (float(np.max((r_pos - r_pos.mean(axis=0)) @ u_eye)) + 20.0)
                * u_eye)
    lam_elec = l_pos.mean(axis=0) + 30.0 * u_eye
    med_elec = t45_pos.mean(axis=0) + 30.0 * (
        (t45_pos.mean(axis=0) - l_pos.mean(axis=0))
        / np.linalg.norm(t45_pos.mean(axis=0) - l_pos.mean(axis=0)))
    electrodes = np.array([eye_elec, lam_elec, med_elec])
    center = np.vstack([r_pos, l_pos, t45_pos]).mean(axis=0)
    r_max = np.linalg.norm(np.vstack([r_pos, l_pos, t45_pos, electrodes])
                           - center, axis=1).max()
    r1 = 1.05 * r_max + 10.0

    def make_syn_and_kernel(e_sub, target_index, n_post, gain, tau_s,
                            sealed=False):
        pre = e_sub["body_pre"].to_numpy(np.int64)
        post = e_sub["body_post"].to_numpy(np.int64)
        weight = e_sub["weight"].to_numpy(np.float32)
        syn = ExponentialSynapses(pre, post, weight, target_index, dt=DT,
                                  gain=gain, tau_s=tau_s, n_post=n_post)
        order = np.lexsort((pre, post))
        pr = np.array([pp[b] for b in pre[order]])
        po = np.array([qq[b] for b in post[order]])
        if sealed:
            f = SealedHeadPairField(pr, po, electrodes, center=center, r1=r1,
                                    r2=1.3 * r1, sigma1=exp005.SIGMA,
                                    sigma2=0.01 * exp005.SIGMA)
        else:
            f = StaticPairField(pr, po, electrodes, sigma=exp005.SIGMA)
        return syn, f

    photo = exp005.PhotoCascadeVector(n_r, DT)
    rhabd = r_pos + 23.5 * u_eye
    field_photo_h = StaticPairField(r_pos, rhabd, electrodes,
                                    sigma=exp005.SIGMA)
    field_photo_s = SealedHeadPairField(r_pos, rhabd, electrodes,
                                        center=center, r1=r1, r2=1.3 * r1,
                                        sigma1=exp005.SIGMA,
                                        sigma2=0.01 * exp005.SIGMA)

    t_ms = np.arange(int(T_END / DT) // 2) * 1.0
    results = {}
    for level in LEVELS_PA:
        syn_rl, f_rl_h = make_syn_and_kernel(e_rl, l_index, n_l, GAIN_RL,
                                             TAU_RL)
        _, f_rl_s = make_syn_and_kernel(e_rl, l_index, n_l, GAIN_RL, TAU_RL,
                                        sealed=True)
        syn_lm, f_lm_h = make_syn_and_kernel(e_lm, mid_index, n_mid, GAIN_LM,
                                             TAU_LM)
        _, f_lm_s = make_syn_and_kernel(e_lm, mid_index, n_mid, GAIN_LM,
                                        TAU_LM, sealed=True)
        syn_mt, fs_mt_h, fs_mt_s = {}, {}, {}
        for mt in exp005.MID_TYPES:
            syn_mt[mt], fs_mt_h[mt] = make_syn_and_kernel(
                e_mt[mt], t45_index, n_t45, GAIN_MT, KIN[mt])
            _, fs_mt_s[mt] = make_syn_and_kernel(
                e_mt[mt], t45_index, n_t45, GAIN_MT, KIN[mt], sealed=True)

        rng = np.random.default_rng(SEED)
        pop_r = LIFPopulation(n_r, DT, tau_m=exp005.R_TAU,
                              t_refrac=exp005.R_REF, R_m=exp005.RIN)
        pop_l = LIFPopulation(n_l, DT, tau_m=exp005.L_TAU,
                              t_refrac=exp005.L_REF, R_m=exp005.RIN)
        pop_mid = LIFPopulation(n_mid, DT, tau_m=exp005.MID_TAU,
                                t_refrac=exp005.MID_REF, R_m=exp005.RIN)
        pop_t45 = LIFPopulation(n_t45, DT, tau_m=exp005.T45_TAU,
                                t_refrac=exp005.T45_REF, R_m=exp005.RIN)
        photo = exp005.PhotoCascadeVector(n_r, DT)
        l_base = np.full(n_l, I_L_BASE)
        mid_base = np.full(n_mid, I_MID_BASE)

        n_steps = int(T_END / DT)
        n_field = n_steps // 2
        counts = {ty: np.zeros(n_field) for ty in RATE_TYPES}
        phi_h = np.zeros((n_field, 3))
        phi_s = np.zeros((n_field, 3))

        print(f"simulating flash level {level:.0f} pA ...")
        for k in range(n_steps):
            t = k * DT
            inc = np.full(n_r, level) if FLASH_ON <= t < FLASH_OFF \
                else np.zeros(n_r)
            inc_f = photo.step(inc)
            i_r = exp005.I_R_BASE + inc_f + rng.normal(
                0, exp005.NOISE_SD["R"], n_r)
            sp_r = pop_r.step(i_r)
            i_l = l_base + syn_rl.to_neuron_current() + rng.normal(
                0, exp005.NOISE_SD["L"], n_l)
            sp_l = pop_l.step(i_l)
            i_mid = mid_base + syn_lm.to_neuron_current() + rng.normal(
                0, exp005.NOISE_SD["MID"], n_mid)
            sp_mid = pop_mid.step(i_mid)
            i_t45 = sum(s.to_neuron_current() for s in syn_mt.values()) \
                + rng.normal(0, exp005.NOISE_SD["T45"], n_t45)
            sp_t45 = pop_t45.step(i_t45)

            syn_rl.step(r_ids[sp_r])
            syn_lm.step(l_ids[sp_l])
            spiked_mid = mid_ids[sp_mid]
            for s in syn_mt.values():
                s.step(spiked_mid)

            if k % 2 == 0:
                j = k // 2
                for ty in RATE_TYPES:
                    m = type_masks[ty]
                    if ty == "R":
                        counts[ty][j] = sp_r.sum()
                    elif ty in ("L1", "L2", "L3"):
                        counts[ty][j] = sp_l[m].sum()
                    elif ty in ("Mi1", "Tm3", "Mi9", "Tm1", "Tm2", "Tm4",
                                "Tm9"):
                        counts[ty][j] = sp_mid[m].sum()
                    else:
                        counts[ty][j] = sp_t45[m].sum()
                i_photo = exp005.I_R_BASE + inc_f
                y_sum_h = f_rl_h.field_timeseries(syn_rl.y[None, :])[0] \
                    + f_lm_h.field_timeseries(syn_lm.y[None, :])[0]
                y_sum_s = f_rl_s.field_timeseries(syn_rl.y[None, :])[0] \
                    + f_lm_s.field_timeseries(syn_lm.y[None, :])[0]
                for mt in exp005.MID_TYPES:
                    y_sum_h += fs_mt_h[mt].field_timeseries(
                        syn_mt[mt].y[None, :])[0]
                    y_sum_s += fs_mt_s[mt].field_timeseries(
                        syn_mt[mt].y[None, :])[0]
                phi_h[j] = y_sum_h + field_photo_h.field_timeseries(
                    i_photo[None, :])[0]
                phi_s[j] = y_sum_s + field_photo_s.field_timeseries(
                    i_photo[None, :])[0]

        sizes = {ty: int(type_masks[ty].sum()) for ty in RATE_TYPES}
        rates = {ty: counts[ty] * 1000.0 / sizes[ty] for ty in RATE_TYPES}

        def wr(ty, t0, t1):
            m = (t_ms >= t0) & (t_ms < t1)
            return round(float(rates[ty][m].mean()), 1)

        rate_table = {ty: {"dark": wr(ty, 100, FLASH_ON),
                           "on": wr(ty, FLASH_ON + 300, FLASH_OFF - 300),
                           "off": wr(ty, FLASH_OFF + 100, FLASH_OFF + 800)}
                      for ty in RATE_TYPES}
        results[level] = {"rates": rate_table,
                          "phi_hom_nV": phi_h * 1e9,
                          "phi_seal_nV": phi_s * 1e9,
                          "rates_traces": rates}
        print(json.dumps(rate_table, indent=1))

    # ---- summary ----
    summary = {"config": {
        "flash_ms": [FLASH_ON, FLASH_OFF], "levels_pA": list(LEVELS_PA),
        "i_l_base_pA": I_L_BASE, "i_mid_base_pA": I_MID_BASE,
        "gains": {"RL": GAIN_RL, "LM": GAIN_LM, "MT": GAIN_MT},
        "signs": "dataset sign column (R->L histamine -, L1->Mi/Tm glutamate"
                 " -, L2/L3->Tm ACh +)",
        "n_R": n_r, "n_L": n_l, "n_mid": n_mid, "n_t45": n_t45,
    }, "levels": {}}
    for level, res in results.items():
        summary["levels"][f"{level:.0f}pA"] = {
            "rates_hz": res["rates"],
            "erg_eye_hom_nV": lockout_metrics(res["phi_hom_nV"][:, 0],
                                              t_ms),
            "erg_eye_seal_nV": lockout_metrics(res["phi_seal_nV"][:, 0],
                                               t_ms),
            "medulla_hom_nV": lockout_metrics(res["phi_hom_nV"][:, 2], t_ms),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["levels"]["350pA"]["erg_eye_hom_nV"],
                     indent=1))

    # ---- figures ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    res = results[LEVELS_PA[0]]
    for ty, color in (("R", "k"), ("L1", "tab:blue"), ("L2", "tab:red"),
                      ("Mi1", "tab:green"), ("Tm2", "tab:orange")):
        axes[0].plot(t_ms, res["rates_traces"][ty], color=color, lw=0.7,
                     label=ty)
    axes[0].axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(fontsize=8, ncols=5)
    axes[0].set_title("exp009: data-driven signs — LMC hyperpolarizing "
                      "responses, disinhibition cascade (1x flash)")
    for i, (name, sig) in enumerate((("homogeneous", "phi_hom_nV"),
                                     ("sealed head", "phi_seal_nV"))):
        axes[1].plot(t_ms, res[sig][:, 0] / 1000.0, label=f"eye {name}")
    axes[1].axvspan(FLASH_ON, FLASH_OFF, color="orange", alpha=0.12)
    axes[1].set_ylabel("eye electrode (uV)")
    axes[1].set_xlabel("time (ms)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_polarity.png", dpi=130)
    print(f"\nfigures and summary written to {OUT}")


if __name__ == "__main__":
    main()
