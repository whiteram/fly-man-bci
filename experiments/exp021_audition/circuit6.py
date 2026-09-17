"""exp021: auditory region builder (JO ear -> AMMC/brain pathway).

Adds the hearing channel as a switchable ffbm.regions region:

  audition   JOA   Johnston-organ auditory sensory neurons -- the
                   subclass "auditory" subset of the JO- types
                   (115 cells: JO-A1/A2/A3/A4, JO-B1/B2/B3, JO-CA1/2,
                  JO-*-unclear; rootSide L 63 / R 52). The other JO
                   subclasses (wind_gravity / grooming) stay out --
                   this region models COURTSHIP-SONG HEARING.

Wiring (connectome-weights, weight >= 5, consensus-nt signs, same
convention as exp019): JO-A terminals land in the antennal-mechano-
sensory-and-motor-center zone and partner with cb_intrinsic cells
(SAD001, CB1076, SAD051, GNG636, AMMC0xx, ... all inside CEN):

  JOA_R  JOA->JOA  local (38 pairs; forward=False)
  JOA_C  JOA->CEN  the auditory afferent dipole (1,625 pairs /
                   24,728 syn; forward=True)

JO-A cells have NO somaLocation (peripheral somata in the antenna);
positions = mean PRESYNAPTIC terminal sites (all 115 have one), the
exp019/020 sensory convention.

Stimulus interface: circuit["chem_groups"]["JOA"] maps
"<type>@<rootSide>" -> compact index array; a chem_inputs.json channel
drives e.g. "JO-A*@L". The near-field sound carrier (pulse song
150-300 Hz / sine song ~150 Hz) is far above the LIF pop's following
capacity (tau 10 ms), so the drive encodes the song AMPLITUDE ENVELOPE
(documented modeling choice) through the shared PhotoCascade
transduction stage.

Run from repository root:
    python experiments/exp021_audition/circuit6.py    # report
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

_ID_BASE = 4_000_000_000          # above exp020's 3e9 PRO band
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


def build_audition(circuit):
    ann, sites = _prep()
    if "soma" not in _CACHE:
        _CACHE["soma"] = fdata.neuron_positions(ann)
        sc = ann["superclass"].fillna("")
        _CACHE["mask_CEN"] = sc.str.startswith("cb_") \
            | (sc == "descending_neuron")
    m = ann["subclass"].fillna("") == "auditory"
    raw = np.array(sorted(ann.loc[m, "bodyId"].tolist()), dtype=np.int64)
    # dipole positions come from the PreSyn terminal arbor; 26/115 JO-A
    # cells appear in the connectome only as postsynaptic partners (no
    # PreSyn sites) -- they stay out of the pop, same rule as regions
    # requiring a soma
    drop = [int(b) for b in raw if int(b) not in sites]
    raw = np.array([b for b in raw if int(b) in sites], dtype=np.int64)
    if drop:
        print(f"  JOA: {len(drop)} cells without PreSyn terminals "
              f"excluded ({len(raw)} kept)")

    base = _CACHE["next_id"]
    remap = {int(old): base + i for i, old in enumerate(raw)}
    _CACHE["next_id"] = base + len(raw)
    _CACHE["raw_JOA"] = raw
    _CACHE["remap_JOA"] = remap
    ids_sorted = np.array(sorted(remap.values()), dtype=np.int64)

    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for old, new in remap.items():
        p = sites[old]
        pp[new] = p
        qq[new] = p
    pops = circuit.get("extra_pops") or {}
    pops["JOA"] = {"ids": ids_sorted, "tau_ms": 10.0, "t_refrac_ms": 2.0,
                   "i_base": "I_JOA_BASE", "ou_sigma": CAL["OU_VPN_CB"]}
    circuit["extra_pops"] = pops

    # chem_groups: "<type>@<rootSide>" -> indices into ids_sorted
    # (kept cells only -- dropped cells must not appear here)
    sub = ann.loc[ann["bodyId"].isin(set(raw.tolist())),
                  ["bodyId", "type", "rootSide"]].copy()
    sub["compact"] = sub["bodyId"].map(remap)
    groups = {}
    for (tname, side), grp in sub.groupby(
            [sub["type"].fillna("?"), sub["rootSide"].fillna("?")]):
        groups[f"{tname}@{side}"] = np.searchsorted(
            ids_sorted, grp["compact"].to_numpy(np.int64))
    chem_groups = circuit.get("chem_groups") or {}
    chem_groups["JOA"] = groups
    circuit["chem_groups"] = chem_groups

    def edge(gname, post_pop, forward):
        pre_w, post_w, wt = _CACHE["w"]
        if post_pop == "CEN":
            raw_post = _region_raw_ids(ann, _CACHE["mask_CEN"])
            compact = circuit["extra_pops"]["CEN"]["ids"]
        else:
            raw_post = _CACHE["raw_JOA"]
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
        edges[gname] = {"pre": ("JOA",), "post": post_pop, "table": e,
                        "tau_s": CAL["TAU_V_MS"], "g_unit": "G_UNIT_CX",
                        "forward": forward}
        circuit["extra_edges"] = edges
        return len(e), float(e["weight"].sum())

    k, s = edge("JOA_R", "JOA", forward=False)
    print(f"  JOA_R JOA->JOA: {k:,} pairs / {s:,.0f} syn")
    k, s = edge("JOA_C", "CEN", forward=True)
    print(f"  JOA_C JOA->CEN: {k:,} pairs / {s:,.0f} syn")
    print(f"audition: {len(raw):,} JO-A cells")
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
    circuit = m3.build_ol_rest(circuit)      # central_brain needs OLR
    circuit = m3.build_central_brain(circuit)
    circuit = m3.build_vnc(circuit)
    circuit = build_audition(circuit)
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()},
           "joa_groups": len(circuit["chem_groups"]["JOA"])}
    (OUT / "circuit6_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("cells", "edges", "joa_groups")},
                     indent=1))
