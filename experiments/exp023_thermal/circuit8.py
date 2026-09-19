"""exp023: thermal/hygrosensory region builder (temperature + humidity).

Adds the two environmental senses as one switchable ffbm.regions
region with TWO populations:

  thermal   TH   thermosensory receptor neurons (25 cells: TRN_VP1m
                 11 hot-pathway, TRN_VP2 7, TRN_VP3a 6, TRN_VP3b 1;
                 rootSide L 12 / R 13 -- the antenna's anterior-
                 organizer thermosensory pathway)
            HY   hygrosensory receptor neurons (66 cells: HRN_VP4 28
                 dry, HRN_VP1d 18, HRN_VP5 12, HRN_VP1l 8 moist;
                 L 29 / R 36 / unknown 1)

Wiring (connectome-weights, weight >= 5, consensus-nt signs, same
convention as exp019-022): the hot/cold and dry/moist afferents
target central brain cells inside CEN:

  TH_C   TH->CEN   thermal afferent dipole (forward=True)
  HY_C   HY->CEN   hygrosensory afferent dipole (forward=True)
  TH_R / HY_R      local recurrence (forward=False; small)

Positions: mean PRESYNAPTIC terminal sites (all 91 have one), the
exp019-022 sensory convention.

Stimulus interface: chem_groups per pop map "<type>@<rootSide>" ->
compact indices; chem channels address e.g. "TRN_VP1m*@L" (hot left)
or "HRN_VP4*@R" (dry right).  Temperature coding: SLOW tonic ramps
(smoothstep seconds-scale), the natural envelope for TRN tonic firing
-- the shared PhotoCascade handles the edge smoothing.

Run from repository root:
    python experiments/exp023_thermal/circuit8.py    # report
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata
from ffbm.params import cal as params_cal

OUT = Path(__file__).resolve().parent / "outputs"
W_MIN = 5
CAL = params_cal()

_ID_BASE = 6_000_000_000          # above exp022's 5e9 TC band
_CACHE = {}


def _prep():
    if "ann" not in _CACHE:
        ann = fdata.load_annotations()
        sites = fdata.load_neuron_sites()
        _CACHE["ann"] = ann
        _CACHE["sites"] = fdata.site_positions(sites, "PreSyn")
        t = pf.read_table(fdata.RAW / "connectome-weights.feather")
        pre = t.column("body_pre").combine_chunks().to_numpy(
            zero_copy_only=False)
        post = t.column("body_post").combine_chunks().to_numpy(
            zero_copy_only=False)
        wt = t.column("weight").combine_chunks().to_numpy(
            zero_copy_only=False)
        m = wt >= W_MIN
        _CACHE["w"] = (pre[m], post[m], wt[m])
        nt = pd.read_feather(fdata.RAW / "body-neurotransmitters.feather")
        sig = nt.set_index("body")["consensus_nt"].map(
            lambda s: -1.0 if isinstance(s, str)
            and s.lower() in ("gaba", "glutamate") else 1.0)
        _CACHE["sig"] = sig
        _CACHE["next_id"] = _ID_BASE
    return _CACHE["ann"], _CACHE["sites"]


def _signs_for(raw_ids):
    return pd.Series(raw_ids).map(_CACHE["sig"]).fillna(1.0).to_numpy(
        np.float32)


def _region_raw_ids(ann, mask):
    soma = _CACHE["soma"]
    return np.array(sorted(ann.loc[mask & ann["bodyId"].isin(soma),
                                  "bodyId"].tolist()), dtype=np.int64)


# (pop name, class filter, i_base key)
_POPS = (("TH", "thermosensory", "I_TH_BASE"),
         ("HY", "hygrosensory", "I_HY_BASE"))


def build_thermal(circuit):
    ann, sites = _prep()
    if "soma" not in _CACHE:
        _CACHE["soma"] = fdata.neuron_positions(ann)
        sc = ann["superclass"].fillna("")
        _CACHE["mask_CEN"] = sc.str.startswith("cb_") \
            | (sc == "descending_neuron")
    pops = circuit.get("extra_pops") or {}
    chem_groups = circuit.get("chem_groups") or {}
    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for pop, cls_name, i_key in _POPS:
        raw = np.array(sorted(
            ann.loc[ann["class"] == cls_name, "bodyId"].tolist()),
            dtype=np.int64)
        missing = [int(b) for b in raw if int(b) not in sites]
        assert not missing, f"{len(missing)} {pop} cells w/o terminals"

        base = _CACHE["next_id"]
        remap = {int(old): base + i for i, old in enumerate(raw)}
        _CACHE["next_id"] = base + len(raw)
        _CACHE[f"raw_{pop}"] = raw
        _CACHE[f"remap_{pop}"] = remap
        ids_sorted = np.array(sorted(remap.values()), dtype=np.int64)
        for old, new in remap.items():
            p = sites[old]
            pp[new] = p
            qq[new] = p
        pops[pop] = {"ids": ids_sorted, "tau_ms": 10.0,
                     "t_refrac_ms": 2.0, "i_base": i_key,
                     "ou_sigma": CAL["OU_VPN_CB"]}

        sub = ann.loc[ann["bodyId"].isin(set(raw.tolist())),
                      ["bodyId", "type", "rootSide"]].copy()
        sub["compact"] = sub["bodyId"].map(remap)
        groups = {}
        for (tname, side), grp in sub.groupby(
                [sub["type"].fillna("?"), sub["rootSide"].fillna("?")]):
            groups[f"{tname}@{side}"] = np.searchsorted(
                ids_sorted, grp["compact"].to_numpy(np.int64))
        chem_groups[pop] = groups

        def edge(gname, post_pop, forward, raw=raw, remap=remap,
                 ids_sorted=ids_sorted):
            pre_w, post_w, wt = _CACHE["w"]
            if post_pop == "CEN":
                raw_post = _region_raw_ids(ann, _CACHE["mask_CEN"])
                compact = circuit["extra_pops"]["CEN"]["ids"]
            else:
                raw_post = raw
                compact = ids_sorted
            assert len(raw_post) == len(compact)
            rmap = dict(zip(raw_post.tolist(), compact.tolist()))
            mm = np.isin(pre_w, raw) & np.isin(post_w, raw_post)
            e = pd.DataFrame({
                "body_pre": pd.Series(pre_w[mm]).map(remap).to_numpy(
                    np.int64),
                "body_post": pd.Series(post_w[mm]).map(rmap).to_numpy(
                    np.int64),
                "weight": wt[mm].astype(np.float32),
                "sign": _signs_for(pre_w[mm])})
            e = e.sort_values(["body_pre", "body_post"]).reset_index(
                drop=True)
            edges = circuit.get("extra_edges") or {}
            edges[gname] = {"pre": (pop,), "post": post_pop, "table": e,
                            "tau_s": CAL["TAU_V_MS"],
                            "g_unit": "G_UNIT_CX", "forward": forward}
            circuit["extra_edges"] = edges
            return len(e), float(e["weight"].sum())

        k, s = edge(f"{pop}_R", pop, forward=False)
        print(f"  {pop}_R {pop}->{pop}: {k:,} pairs / {s:,.0f} syn")
        k, s = edge(f"{pop}_C", "CEN", forward=True)
        print(f"  {pop}_C {pop}->CEN: {k:,} pairs / {s:,.0f} syn")
        print(f"thermal: {len(raw):,} {pop} cells "
              f"({len(groups)} chem groups)")
    circuit["extra_pops"] = pops
    circuit["chem_groups"] = chem_groups
    return circuit


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "exp015_circuit", ROOT / "experiments" / "exp015_bilateral"
        / "circuit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    circuit = mod.build_bilateral_circuit()
    c3 = importlib.util.spec_from_file_location(
        "exp017_circuit3", ROOT / "experiments" / "exp017_full_cns"
        / "circuit3.py")
    m3 = importlib.util.module_from_spec(c3)
    c3.loader.exec_module(m3)
    circuit = m3.build_ol_rest(circuit)
    circuit = m3.build_central_brain(circuit)
    circuit = m3.build_vnc(circuit)
    circuit = build_thermal(circuit)
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()},
           "chem_groups": {p: len(g) for p, g in
                           circuit["chem_groups"].items()}}
    (OUT / "circuit8_report.json").write_text(json.dumps(rep, indent=1))
    big_th = sorted(circuit["chem_groups"]["TH"].items(),
                    key=lambda kv: -len(kv[1]))[:5]
    big_hy = sorted(circuit["chem_groups"]["HY"].items(),
                    key=lambda kv: -len(kv[1]))[:5]
    print("largest TH groups:", {k: len(v) for k, v in big_th})
    print("largest HY groups:", {k: len(v) for k, v in big_hy})
    print(json.dumps({k: rep[k] for k in ("cells", "edges")}, indent=1))
