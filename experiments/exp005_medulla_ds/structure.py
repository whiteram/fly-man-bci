"""exp005 Part A: structure — column offsets of T4/T5 inputs (delay-line
blueprint) and circuit extraction stats.

T4/T5 have no assigned hex column, but each T4 in column c receives its Mi1
input from Mi1 cells of column c (1 Mi1 per column, 4 T4 per column), so the
column of a T4/T5 cell is inferred by majority vote over its dominant
presynaptic partners. The per-input-type distribution of hex(post) - hex(pre)
is the anatomical substrate a delay-and-compare mechanism would need.

Run from repository root:
    python experiments/exp005_medulla_ds/structure.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata

OUT = Path(__file__).resolve().parent / "outputs"

INPUT_T4 = ("Mi1", "Tm3", "Mi4", "Mi9")
INPUT_T5 = ("Tm1", "Tm2", "Tm4", "Tm9")
MID = INPUT_T4 + INPUT_T5
T45 = tuple(f"T{x}{y}" for x in "45" for y in "abcd")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = fdata.load_visual_nodes()
    edges = fdata.load_visual_edges()
    ann = fdata.load_annotations()
    types = nodes.set_index("bodyId")["type"]

    hex1 = ann.set_index("bodyId")["assignedOlHex1"]
    e = edges.assign(tp=edges["body_pre"].map(types),
                     tq=edges["body_post"].map(types))

    # Tm3 (0% hex coverage) and Tm4 (50%) get their column by majority vote of
    # their dominant presynaptic partners (L1 -> Tm3, L2 -> Tm4; both ~1:1
    # per column), filling gaps before use.
    hex_full = hex1.to_dict()
    for target, voter in (("Tm3", "L1"), ("Tm4", "L2")):
        s = e[(e["tp"] == voter) & (e["tq"] == target)].dropna(
            subset=["tp", "tq"])
        voted = (s.assign(h=s["body_pre"].map(hex_full))
                 .dropna(subset=["h"])
                 .groupby("body_post")["h"]
                 .agg(lambda x: x.mode().iloc[0]))
        filled = sum(1 for b in voted.index if hex_full.get(b) is None
                     or np.isnan(hex_full.get(b, np.nan)))
        hex_full.update(voted.to_dict())
        print(f"column fill {target} <- {voter} vote: {len(voted):,} cells "
              f"({filled} newly assigned)")

    # --- infer T4/T5 columns by majority vote of dominant partners ---
    # T4 <- Mi1 vote; T5 <- Tm9 vote (both ~1 cell/column, strong weights)
    stats = {}
    offsets = {}
    offsets_per_subtype = {}
    hex_of_by_pg = {}
    for post_group, voter in (("T4", "Mi1"), ("T5", "Tm9")):
        sub = e[(e["tp"] == voter) & e["tq"].str.startswith(post_group)]
        sub = sub.dropna(subset=["tp", "tq"])
        hex_pre = sub["body_pre"].map(hex_full)
        keep = hex_pre.notna()
        sub = sub[keep]
        hex_pre = hex_pre[keep]
        mode_hex = (sub.assign(h=hex_pre)
                    .groupby("body_post")["h"]
                    .agg(lambda s: s.mode().iloc[0]))
        vote_frac = (sub.assign(h=hex_pre)
                     .groupby("body_post")["h"]
                     .agg(lambda s: float((s == s.mode().iloc[0]).mean())))
        stats[post_group] = {
            "n_cells_with_column": int(len(mode_hex)),
            "median_vote_agreement": round(float(vote_frac.median()), 3),
        }
        hex_of = dict(hex_full)
        hex_of.update(mode_hex.to_dict())
        hex_of_by_pg[post_group] = hex_of

        # offset distribution per input type (pooled over subtypes and
        # separately per T4a-d / T5a-d — the per-subtype differential slow-vs-
        # fast offset is the delay-line signature)
        offsets[post_group] = {}
        for mt in (INPUT_T4 if post_group == "T4" else INPUT_T5):
            s = e[(e["tp"] == mt) & e["tq"].str.startswith(post_group)]
            s = s.dropna(subset=["tp", "tq"])
            s = s.assign(dh=s["body_post"].map(hex_of)
                         - s["body_pre"].map(hex_of)).dropna(subset=["dh"])
            dh = s["dh"].to_numpy(int)
            w = s["weight"].to_numpy(float)
            vals, counts = np.unique(dh, return_counts=True)
            offsets[post_group][mt] = {
                "n_edges": int(len(dh)),
                "weighted_mean_dhex": round(float(np.average(dh, weights=w)), 3),
                "frac_same_column": round(float((dh == 0).mean()), 3),
                "hist_dhex": {int(k): int(v) for k, v in zip(vals, counts)},
            }
            per_sub = {}
            for sub in (f"{post_group}{c}" for c in "abcd"):
                ss = s[s["tq"] == sub]
                per_sub[sub] = (round(float(np.average(
                    ss["dh"], weights=ss["weight"])), 3) if len(ss) else None)
            offsets_per_subtype[f"{mt}->{post_group}"] = per_sub
    print(json.dumps({"column_inference": stats, "offsets": offsets,
                      "offsets_per_subtype": offsets_per_subtype},
                     indent=1)[:2000])

    # --- per-lobe check: the two lobes' hex u-axes are mirrored in physical
    # x, so pooled offsets could in principle cancel; verify per lobe ---
    sites = fdata.load_neuron_sites()
    pre_pos = fdata.site_positions(sites, "PreSyn")
    mid_bodies = nodes.loc[nodes["type"].isin(MID), "bodyId"]
    mid_bodies = [b for b in mid_bodies if b in pre_pos]
    mx = np.array([pre_pos[b][0] for b in mid_bodies])
    lobe_mid, _ = (lambda t: (np.percentile(t, [25, 75]).astype(float), None))(mx)
    lab_mid = np.abs(mx - lobe_mid[0]) < np.abs(mx - lobe_mid[1])
    for _ in range(20):
        lobe_mid = np.array([mx[lab_mid].mean(), mx[~lab_mid].mean()])
        lab_mid = np.abs(mx - lobe_mid[0]) < np.abs(mx - lobe_mid[1])
    lobe_of = dict(zip(mid_bodies, lab_mid.tolist()))  # True = lobe A
    midline_ab = float(np.mean(lobe_mid))
    per_lobe = {}
    for pg, voter in (("T4", "Mi1"), ("T5", "Tm9")):
        sub = e[(e["tp"] == voter) & e.tq.str.startswith(pg)].dropna(
            subset=["tp", "tq"])
        hex_pre = sub.body_pre.map(hex_full)
        sub = sub[hex_pre.notna()]
        mode_hex = (sub.assign(h=hex_pre[hex_pre.notna()])
                    .groupby("body_post")["h"].agg(lambda x: x.mode().iloc[0]))
        hex_of2 = dict(hex_full)
        hex_of2.update(mode_hex.to_dict())
        for lobe_name, lobe_val in (("lobeA", True), ("lobeB", False)):
            for mt in (INPUT_T4 if pg == "T4" else INPUT_T5):
                s = e[(e["tp"] == mt) & e.tq.str.startswith(pg)]
                s = s.dropna(subset=["tp", "tq"])
                s = s[s["body_pre"].map(lobe_of) == lobe_val]
                s = s.assign(dh=s.body_post.map(hex_of2)
                             - s.body_pre.map(hex_of2)).dropna(subset=["dh"])
                per_lobe[f"{mt}->{pg}/{lobe_name}"] = {
                    st: (round(float(np.average(
                        s.loc[s["tq"] == st, "dh"],
                        weights=s.loc[s["tq"] == st, "weight"])), 3)
                        if (s["tq"] == st).any() else None)
                    for st in (f"{pg}{c}" for c in "abcd")}
    print("per-lobe weighted mean dhex:")
    print(json.dumps(per_lobe, indent=1))

    # --- 2D offsets (hex1 = u, hex2 = v): weighted-mean offset VECTOR per
    # input type x subtype in lobe A — the direction-carrying blueprint.
    # hex2 is the second hex-grid axis (79% coverage in Mi/Tm types). ---
    hex2 = ann.set_index("bodyId")["assignedOlHex2"]
    h2f = hex2.to_dict()
    for target, voter in (("Tm3", "L1"), ("Tm4", "L2")):
        s2 = e[(e["tp"] == voter) & (e["tq"] == target)].dropna(
            subset=["tp", "tq"])
        v2 = (s2.assign(h=s2["body_pre"].map(h2f)).dropna(subset=["h"])
              .groupby("body_post")["h"].agg(lambda x: x.mode().iloc[0]))
        h2f.update(v2.to_dict())
    offsets_2d = {}
    for pg, voter in (("T4", "Mi1"), ("T5", "Tm9")):
        sub = e[(e["tp"] == voter) & e.tq.str.startswith(pg)].dropna(
            subset=["tp", "tq"])
        v2 = (sub.assign(h=sub["body_pre"].map(h2f)).dropna(subset=["h"])
              .groupby("body_post")["h"].agg(lambda x: x.mode().iloc[0]))
        h2_of = dict(h2f)
        h2_of.update(v2.to_dict())
        h1_of = dict(hex_of_by_pg[pg]) if pg in hex_of_by_pg else {}
        for mt in (INPUT_T4 if pg == "T4" else INPUT_T5):
            s2 = e[(e["tp"] == mt) & e.tq.str.startswith(pg)]
            s2 = s2.dropna(subset=["tp", "tq"])
            s2 = s2[[lobe_of.get(b, False) is True and
                     pre_pos.get(b) is not None and
                     pre_pos[b][0] < midline_ab
                     for b in s2["body_pre"]]]
            entry = {}
            for st in (f"{pg}{c}" for c in "abcd"):
                ss = s2[s2["tq"] == st]
                du = ss["body_post"].map(h1_of) - ss["body_pre"].map(hex_full)
                dv = ss["body_post"].map(h2_of) - ss["body_pre"].map(h2_of)
                w = ss["weight"]
                ok = np.isfinite(du) & np.isfinite(dv) & np.isfinite(w)
                if ok.sum() > 100:
                    mu = float(np.average(du[ok], weights=w[ok]))
                    mv = float(np.average(dv[ok], weights=w[ok]))
                    entry[st] = {
                        "du": round(mu, 3), "dv": round(mv, 3),
                        "mag": round(float(np.hypot(mu, mv)), 3),
                        "angle_deg": round(
                            float(np.degrees(np.arctan2(mv, mu))), 1),
                        "n_edges": int(ok.sum()),
                    }
                else:
                    entry[st] = None
            offsets_2d[f"{mt}->{pg}"] = entry
    print("2D offsets (lobe A):")
    print(json.dumps(offsets_2d, indent=1))

    (OUT / "structure_offsets.json").write_text(json.dumps(
        {"column_inference": stats, "offsets": offsets,
         "offsets_per_subtype": offsets_per_subtype,
         "offsets_per_lobe": per_lobe,
         "offsets_2d_lobeA": offsets_2d}, indent=1))

    # --- figure: offset histograms + per-subtype slow-fast differential ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, post_group in zip(axes[:2], ("T4", "T5")):
        for mt, color in zip(offsets[post_group],
                             ("tab:blue", "tab:orange", "tab:green",
                              "tab:red")):
            hist = offsets[post_group][mt]["hist_dhex"]
            ks = sorted(int(k) for k in hist)
            ws = [hist[k] for k in ks]
            ax.plot(ks, np.array(ws) / sum(ws), "o-", color=color, label=mt)
        ax.set_xlabel("hex(post) - hex(pre)")
        ax.set_ylabel("fraction of edges")
        ax.set_title(f"{post_group} inputs (pooled)")
        ax.legend()
    slow = {"T4": "Mi1", "T5": "Tm1"}
    fast = {"T4": "Tm3", "T5": "Tm4"}
    for pg in ("T4", "T5"):
        subs = [f"{pg}{c}" for c in "abcd"]
        diff = [offsets_per_subtype[f"{slow[pg]}->{pg}"][s]
                - offsets_per_subtype[f"{fast[pg]}->{pg}"][s] for s in subs]
        axes[2].bar([i - 0.18 if pg == "T4" else i + 0.18
                     for i in range(4)], diff,
                    width=0.36, label=pg)
    axes[2].set_xticks(range(4), ["a", "b", "c", "d"])
    axes[2].set_xlabel("subtype")
    axes[2].set_ylabel("dhex(slow) - dhex(fast)")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_title("slow-vs-fast input offset (delay-line)")
    axes[2].legend()
    fig.suptitle("exp005: anatomical input offsets of T4/T5")
    fig.tight_layout()
    fig.savefig(OUT / "fig_offsets.png", dpi=130)
    print(f"figure -> {OUT / 'fig_offsets.png'}")


if __name__ == "__main__":
    main()
