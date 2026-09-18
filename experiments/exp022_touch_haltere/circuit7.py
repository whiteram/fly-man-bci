"""exp022: tactile region builder (body-surface touch channel).

  touch   TC   mechanosensory_tactile sensory neurons (2,558 cells:
               bristles over the body surface; types SNta*, sides
               rootSide L/R ~ 1264/1294).  All 2,558 have PreSyn
               terminal sites (positions = mean PreSyn site, the
               exp019/020/021 sensory convention).

NOTE on haltere: 201/205 of the haltere-subclass cells are ALREADY
inside the proprioception region (they are campaniform/chordotonal
sensilla by class) -- a separate haltere builder would double-count
them, so the haltere channel is deliberately NOT built here; its
gyroscopic feedback is simulated within PRO (exp020).

Wiring (connectome-weights, w>=5, consensus-nt signs):

  TC_R  TC->TC   local (forward=False)
  TC_C  TC->CEN  ascending tactile afferents (forward=True -- the
                 scalp-visible dipole; 1,346 pairs / 17,192 syn)
  TC_V  TC->VNC  main grooming/postural reflex arcs (forward=False,
                 only when vnc is ON; ~83k pairs / ~969k syn)

Stimulus interface: chem_groups["TC"] has BOTH "<type>@<side>" and
"<SUBCLASS-UPPER>@<side>" keys (e.g. "SNta29@L", "LEG@L", "NOTUM@R"),
so a chem channel can address a body part with patterns like "*@L".

Run from repository root:
    python experiments/exp022_touch_haltere/circuit7.py   # report
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

_ID_BASE = 5_000_000_000          # above exp021's 4e9 JOA band
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


def build_touch(circuit):
    ann, sites = _prep()
    if "soma" not in _CACHE:
        _CACHE["soma"] = fdata.neuron_positions(ann)
        sc = ann["superclass"].fillna("")
        _CACHE["mask_CEN"] = sc.str.startswith("cb_") \
            | (sc == "descending_neuron")
        _CACHE["mask_VNC"] = sc.str.startswith("vnc_") | sc.isin(
            ["ascending_neuron", "sensory_ascending",
             "sensory_ascending_tbc", "vnc_tbc", "vnc_sensory_tbc",
             "efferent_ascending"])
    m = ann["class"].fillna("") == "mechanosensory_tactile"
    raw = np.array(sorted(ann.loc[m & ann["bodyId"].isin(sites),
                                  "bodyId"].tolist()), dtype=np.int64)

    base = _CACHE["next_id"]
    remap = {int(old): base + i for i, old in enumerate(raw)}
    _CACHE["next_id"] = base + len(raw)
    _CACHE["raw_TC"] = raw
    _CACHE["remap_TC"] = remap
    ids_sorted = np.array(sorted(remap.values()), dtype=np.int64)

    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for old, new in remap.items():
        p = sites[old]
        pp[new] = p
        qq[new] = p
    pops = circuit.get("extra_pops") or {}
    pops["TC"] = {"ids": ids_sorted, "tau_ms": 10.0, "t_refrac_ms": 2.0,
                  "i_base": "I_TC_BASE", "ou_sigma": CAL["OU_VPN_CB"]}
    circuit["extra_pops"] = pops

    sub = ann.loc[ann["bodyId"].isin(set(raw.tolist())),
                  ["bodyId", "type", "subclass", "rootSide"]].copy()
    sub["compact"] = sub["bodyId"].map(remap)
    groups = {}
    for (tname, side), grp in sub.groupby(
            [sub["type"].fillna("?"), sub["rootSide"].fillna("?")]):
        groups[f"{tname}@{side}"] = np.searchsorted(
            ids_sorted, grp["compact"].to_numpy(np.int64))
    for (sname, side), grp in sub.groupby(
            [sub["subclass"].fillna("?"), sub["rootSide"].fillna("?")]):
        groups[f"{sname.upper()}@{side}"] = np.searchsorted(
            ids_sorted, grp["compact"].to_numpy(np.int64))
    chem_groups = circuit.get("chem_groups") or {}
    chem_groups["TC"] = groups
    circuit["chem_groups"] = chem_groups

    def edge(gname, post_pop, forward):
        pre_w, post_w, wt = _CACHE["w"]
        if post_pop == "CEN":
            raw_post = _region_raw_ids(ann, _CACHE["mask_CEN"])
            compact = circuit["extra_pops"]["CEN"]["ids"]
        elif post_pop == "VNC":
            raw_post = _region_raw_ids(ann, _CACHE["mask_VNC"])
            compact = circuit["extra_pops"]["VNC"]["ids"]
        else:
            raw_post = _CACHE["raw_TC"]
            compact = ids_sorted
        assert len(raw_post) == len(compact)
        rmap = dict(zip(raw_post.tolist(), compact.tolist()))
        mm = np.isin(pre_w, raw) & np.isin(post_w, raw_post)
        e = pd.DataFrame({
            "body_pre": pd.Series(pre_w[mm]).map(remap).to_numpy(np.int64),
            "body_post": pd.Series(post_w[mm]).map(rmap).to_numpy(np.int64),
            "weight": wt[mm].astype(np.float32),
            "sign": _signs_for(pre_w[mm])})
        e = e.sort_values(["body_pre", "body_post"]).reset_index(drop=True)
        edges = circuit.get("extra_edges") or {}
        edges[gname] = {"pre": ("TC",), "post": post_pop, "table": e,
                        "tau_s": CAL["TAU_V_MS"], "g_unit": "G_UNIT_CX",
                        "forward": forward}
        circuit["extra_edges"] = edges
        return len(e), float(e["weight"].sum())

    k, s = edge("TC_R", "TC", forward=False)
    print(f"  TC_R TC->TC: {k:,} pairs / {s:,.0f} syn")
    k, s = edge("TC_C", "CEN", forward=True)
    print(f"  TC_C TC->CEN: {k:,} pairs / {s:,.0f} syn")
    if "VNC" in (circuit.get("extra_pops") or {}):
        k, s = edge("TC_V", "VNC", forward=False)
        print(f"  TC_V TC->VNC: {k:,} pairs / {s:,.0f} syn")
    print(f"touch: {len(raw):,} TC cells")
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
    circuit = build_touch(circuit)
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()},
           "tc_groups": len(circuit["chem_groups"]["TC"])}
    (OUT / "circuit7_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("cells", "edges",
                                          "tc_groups")}, indent=1))
