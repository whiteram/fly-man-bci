"""exp020: proprioceptive region builder (motor sense / reafference).

Adds the body-sense input channel as a switchable ffbm.regions region:

  proprioception  PRO  leg/wing proprioceptive sensory neurons -- the
                       union of class mechanosensory_proprioceptive
                       (1,454: SNpp leg / SApp wing), chordotonal-organ
                       subclass (425, stretch/vibration), campaniform-
                       sensilla subclass (426, cuticle load) and hair-
                       plate types (113, joint position) = 2,418 cells
                       after de-duplication

Wiring (connectome-weights, weight >= 5, consensus-nt signs, same
convention as exp017/019): one edge group per POSTSYNAPTIC id space,

  PRO_R  PRO->PRO   (local; forward=False)
  PRO_C  PRO->CEN   (ascending + cb partners -- the scalp-visible
                     reafference dipole; forward=True)
  PRO_V  PRO->VNC   (main reflex arc: vnc_intrinsic/vnc_motor; only
                     when the vnc region is ON; forward=False)

Positions: mean PRESYNAPTIC terminal sites (all 2,418 have one), the
same convention as exp019 -- the current source of the PRO->partner
synapse.

Stimulus interface: circuit["chem_groups"]["PRO"] maps
"<annotation type>@<somaSide>" -> compact index array, so a
chem_inputs.json channel can address e.g. "SNpp*@L" (left-leg
proprioceptors) or "SApp*@R"; the drive reuses the generic chem_fn
machinery (transduction cascade included) -- no pipeline changes.

Motor-imagery mapping (bci-style acquisition lives in run.py): a short
reafference pattern on one leg group = the sensory consequence of the
intended movement; DN (descending neurons, somata in CEN, forward
kernels ON) are the motor-plan layer whose activity the scalp EEG can
carry -- decoding a movement class from EEG without any body attached
is the fly analog of movement-imagery decoding.

Run from repository root:
    python experiments/exp020_proprioception/circuit5.py    # report
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

_ID_BASE = 3_000_000_000          # above raw ids (max 1.57e9) and
                                  # clear of exp019's 2e9 band
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


def _pro_mask(ann):
    cls = ann["class"].fillna("")
    sub = ann["subclass"].fillna("")
    typ = ann["type"].fillna("")
    return (cls == "mechanosensory_proprioceptive") \
        | (sub == "chordotonal organ") \
        | (sub == "campaniform sensilla") \
        | typ.str.contains("hair plate", case=False)


def _region_raw_ids(ann, mask):
    soma = _CACHE["soma"]
    return np.array(sorted(ann.loc[mask & ann["bodyId"].isin(soma),
                                  "bodyId"].tolist()), dtype=np.int64)


def build_proprioception(circuit):
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
    mask = _pro_mask(ann)
    raw = np.array(sorted(ann.loc[mask, "bodyId"].tolist()),
                   dtype=np.int64)
    missing = [int(b) for b in raw if int(b) not in sites]
    assert not missing, f"{len(missing)} PRO cells without terminals"

    base = _CACHE["next_id"]
    remap = {int(old): base + i for i, old in enumerate(raw)}
    _CACHE["next_id"] = base + len(raw)
    _CACHE["raw_PRO"] = raw
    _CACHE["remap_PRO"] = remap

    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for old, new in remap.items():
        p = sites[old]
        pp[new] = p
        qq[new] = p
    ids_sorted = np.array(sorted(remap.values()), dtype=np.int64)
    pops = circuit.get("extra_pops") or {}
    pops["PRO"] = {"ids": ids_sorted, "tau_ms": 10.0, "t_refrac_ms": 2.0,
                   "i_base": "I_PRO_BASE", "ou_sigma": CAL["OU_VPN_CB"]}
    circuit["extra_pops"] = pops

    # chem_groups: "<type>@<rootSide>" -> indices into ids_sorted
    # (somaSide is empty for peripheral sensory cells; rootSide carries
    # the L/R split: 735 L / 718 R / 1 unknown -- the movement-class
    # axis of the motor-imagery paradigm)
    sub = ann.loc[mask, ["bodyId", "type", "rootSide"]].copy()
    sub["compact"] = sub["bodyId"].map(remap)
    groups = {}
    for (tname, side), grp in sub.groupby(
            [sub["type"].fillna("?"), sub["rootSide"].fillna("?")]):
        groups[f"{tname}@{side}"] = np.searchsorted(
            ids_sorted, grp["compact"].to_numpy(np.int64))
    chem_groups = circuit.get("chem_groups") or {}
    chem_groups["PRO"] = groups
    circuit["chem_groups"] = chem_groups

    # edge groups
    def edge(gname, post_pop, forward):
        pre_w, post_w, wt = _CACHE["w"]
        if post_pop == "CEN":
            raw_post = _region_raw_ids(ann, _CACHE["mask_CEN"])
            compact = circuit["extra_pops"]["CEN"]["ids"]
        elif post_pop == "VNC":
            raw_post = _region_raw_ids(ann, _CACHE["mask_VNC"])
            compact = circuit["extra_pops"]["VNC"]["ids"]
        else:
            raw_post = _CACHE["raw_PRO"]
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
        edges[gname] = {"pre": ("PRO",), "post": post_pop, "table": e,
                        "tau_s": CAL["TAU_V_MS"], "g_unit": "G_UNIT_CX",
                        "forward": forward}
        circuit["extra_edges"] = edges
        return len(e), float(e["weight"].sum())

    k, s = edge("PRO_R", "PRO", forward=False)
    print(f"  PRO_R PRO->PRO: {k:,} pairs / {s:,.0f} syn")
    k, s = edge("PRO_C", "CEN", forward=True)
    print(f"  PRO_C PRO->CEN: {k:,} pairs / {s:,.0f} syn")
    if "VNC" in (circuit.get("extra_pops") or {}):
        k, s = edge("PRO_V", "VNC", forward=False)
        print(f"  PRO_V PRO->VNC: {k:,} pairs / {s:,.0f} syn")
    else:
        print("  PRO_V skipped (vnc region OFF)")
    print(f"proprioception: {len(raw):,} PRO cells")
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
    circuit = build_proprioception(circuit)
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()},
           "pro_groups": {f"{t}@{s}": len(v)
                          for (t, s), v in []},
           "n_groups": len(circuit["chem_groups"]["PRO"])}
    (OUT / "circuit5_report.json").write_text(json.dumps(rep, indent=1))
    big = sorted(circuit["chem_groups"]["PRO"].items(),
                 key=lambda kv: -len(kv[1]))[:8]
    print("largest PRO groups:",
          {k: len(v) for k, v in big})
    print(json.dumps({k: rep[k] for k in ("cells", "edges")},
                     indent=1))
