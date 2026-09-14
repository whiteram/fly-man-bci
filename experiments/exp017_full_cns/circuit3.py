"""exp017: full-CNS region builders (ol_rest / central_brain / vnc).

Adds everything that is not the 5-layer visual cascade, as three
independently switchable regions (ffbm.regions):

  ol_rest      remaining optic-lobe cells (74,375 with soma):
               all VIS-superclass cells outside the cascade types
  central_brain cb_* + descending + visual_projection VPNs (33,480)
  vnc          vnc_* + ascending classes (15,367); SIMULATED but kept
               out of the scalp forward kernels ("forward": False) --
               the fly's spinal cord has no place inside the human head
               sphere, and sources outside r1 would break the 4-sphere
               kernel

Wiring (connectome-weights, weight >= 5, signs from consensus
neurotransmitter: ACh -> +1, GABA/Glu -> -1, modulators/unknown -> +1,
documented as assumed): each region receives one edge group per
PRESYNAPTIC id space (core cascade / ol_rest / central / vnc) so the
compact per-region id remaps can never collide inside a group:

  OLR_V core->ol_rest    OLR_R ol_rest->ol_rest
  CEN_V core->central    CEN_R ol_rest->central   CEN_C central->central
  VNC_C central->vnc     VNC_R vnc->vnc           ASC_R vnc->central

Positions: soma locations (same um frame as the synapse sites). Working
point: i_base values are CAL-key strings resolved per run (grid-friendly);
the round values live in params.vpn_central and are re-calibrated by
experiments/exp017_full_cns/run.py.

Run from repository root:
    python experiments/exp017_full_cns/circuit3.py      # report
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
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

from ffbm import data as fdata
from ffbm.params import cal as params_cal

OUT = Path(__file__).resolve().parent / "outputs"
W_MIN = 5
CAL = params_cal()

CASCADE_TYPES = ("R1-R6", "L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9",
                 "Tm1", "Tm2", "Tm4", "Tm9")

# module-level caches (one heavy load per process)
_CACHE = {}


def _region_masks(ann):
    sc = ann["superclass"].fillna("")
    types = ann["type"]
    m_vis = sc.isin(["ol_intrinsic", "visual_projection", "ol_sensory",
                     "visual_centrifugal"])
    casc = (m_vis & types.isin(CASCADE_TYPES)) \
        | types.str.match(r"^T[45][abcd]$", na=False)
    return {
        "OLR": m_vis & ~casc,
        "CEN": sc.str.startswith("cb_") | (sc == "descending_neuron"),
        "VNC": sc.str.startswith("vnc_") | sc.isin(
            ["ascending_neuron", "sensory_ascending",
             "sensory_ascending_tbc", "vnc_tbc", "vnc_sensory_tbc",
             "efferent_ascending"]),
    }


def _signs_for(pre_ids, ann):
    """consensus neurotransmitter -> sign (+1 ACh/modulator, -1 GABA/Glu)."""
    nt = pd.read_feather(fdata.RAW / "body-neurotransmitters.feather")
    sig = nt.set_index("body")["consensus_nt"].map(
        lambda s: -1.0 if isinstance(s, str)
        and s.lower() in ("gaba", "glutamate") else 1.0)
    return pre_ids.map(sig).fillna(1.0).to_numpy(np.float32)


def _weights_filtered():
    """(pre, post, weight) arrays at w>=5 -- cached per process."""
    if "w" not in _CACHE:
        t = pf.read_table(fdata.RAW / "connectome-weights.feather")
        pre = t.column("body_pre").combine_chunks().to_numpy(
            zero_copy_only=False)
        post = t.column("body_post").combine_chunks().to_numpy(
            zero_copy_only=False)
        wt = t.column("weight").combine_chunks().to_numpy(
            zero_copy_only=False)
        m = wt >= W_MIN
        _CACHE["w"] = (pre[m], post[m], wt[m])
    return _CACHE["w"]


def _add_region(circuit, name, mask, ann, soma):
    """Append one region: compact ids, soma positions, i_base CAL keys."""
    ids_all = ann.loc[mask & ann["bodyId"].isin(soma), "bodyId"]
    ids = np.array(sorted(ids_all.tolist()), dtype=np.int64)
    base = _CACHE["next_id"]
    remap = {int(old): base + i for i, old in enumerate(ids)}
    _CACHE["next_id"] = base + len(ids)
    _CACHE[f"ids_{name}"] = ids
    _CACHE[f"remap_{name}"] = remap
    _CACHE[f"mask_{name}"] = mask

    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for old, new in remap.items():
        p = soma[old]
        pp[new] = p
        qq[new] = p
    i_key = {"OLR": "I_OLR_BASE", "CEN": "I_CEN_BASE",
             "VNC": "I_VNC_BASE"}[name]
    ou = {"OLR": CAL["OU_VPN_CB"], "CEN": CAL["OU_VPN_CB"],
          "VNC": CAL["OU_VPN_CB"]}[name]
    pops = circuit.get("extra_pops") or {}
    pops[name] = {"ids": np.array(sorted(remap.values()), dtype=np.int64),
                  "tau_ms": 10.0, "t_refrac_ms": 2.0,
                  "i_base": i_key, "ou_sigma": ou}
    circuit["extra_pops"] = pops
    return len(ids)


def _edge_group(circuit, gname, pre_ids_original, post_name,
                pre_space=None, forward=True):
    """One edge group with ALL pre ids in ONE id space (caller ensures).

    pre_ids_original: ORIGINAL body ids of the presynaptic population
    (filter against the raw connectome); pre_space: region name when the
    pre ids must be remapped into that region's compact band so spike
    delivery matches extra_pops ids (None = core cascade, ids stay raw).
    """
    pre, post, wt = _weights_filtered()
    m = np.isin(pre, pre_ids_original) & np.isin(
        post, _CACHE[f"ids_{post_name}"])
    remap = _CACHE[f"remap_{post_name}"]
    pre_col = (pd.Series(pre[m]).map(_CACHE[f"remap_{pre_space}"])
               .to_numpy(np.int64) if pre_space else pre[m])
    e = pd.DataFrame({
        "body_pre": pre_col,
        "body_post": pd.Series(post[m]).map(remap).to_numpy(np.int64),
        "weight": wt[m].astype(np.float32),
        "sign": _signs_for(pd.Series(pre[m]), _CACHE["ann"])})
    e = e.sort_values(["body_pre", "body_post"]).reset_index(drop=True)
    edges = circuit.get("extra_edges") or {}
    edges[gname] = {"pre": None, "pre_space": pre_space or "CORE",
                    "post": post_name,
                    "table": e, "tau_s": CAL["TAU_V_MS"],
                    "g_unit": "G_UNIT_CX", "forward": forward}
    circuit["extra_edges"] = edges
    return len(e), float(e["weight"].sum())


def _resolve_pre(circuit):
    """Fill the pre pop-name lists once ids exist (idempotent)."""
    core = set(circuit["r_ids"].tolist()) | set(circuit["l_ids"].tolist()) \
        | set(circuit["mid_ids"].tolist()) | set(circuit["t45_ids"].tolist())
    spaces = {"CORE": core}
    for reg in ("OLR", "CEN", "VNC"):
        if f"ids_{reg}" in _CACHE:
            spaces[reg] = set(_CACHE[f"ids_{reg}"].tolist())
    for gname, spec in circuit["extra_edges"].items():
        if "pre_space" not in spec:
            continue                      # already resolved (idempotent)
        space = spec.pop("pre_space")
        assert space in spaces, f"{gname}: unknown pre space {space}"
        # core pre space spans R/L/MID/T45 -- deliver from all four
        spec["pre"] = (("R", "L", "MID", "T45") if space == "CORE"
                       else (space,))


def build_ol_rest(circuit):
    ann, soma = _prep()
    n = _add_region(circuit, "OLR", _CACHE["masks"]["OLR"], ann, soma)
    core = set(circuit["r_ids"].tolist()) | set(circuit["l_ids"].tolist()) \
        | set(circuit["mid_ids"].tolist()) | set(circuit["t45_ids"].tolist())
    _edge_group(circuit, "OLR_V", np.array(sorted(core)), "OLR")
    _resolve_pre(circuit)
    print(f"ol_rest: {n:,} cells")
    return circuit


def build_central_brain(circuit):
    ann, soma = _prep()
    n = _add_region(circuit, "CEN", _CACHE["masks"]["CEN"], ann, soma)
    core = set(circuit["r_ids"].tolist()) | set(circuit["l_ids"].tolist()) \
        | set(circuit["mid_ids"].tolist()) | set(circuit["t45_ids"].tolist())
    _edge_group(circuit, "CEN_V", np.array(sorted(core)), "CEN")
    _edge_group(circuit, "CEN_R", _CACHE["ids_OLR"], "CEN",
                pre_space="OLR")
    _edge_group(circuit, "CEN_C", _CACHE["ids_CEN"], "CEN",
                pre_space="CEN")
    _resolve_pre(circuit)
    print(f"central_brain: {n:,} cells")
    return circuit


def build_vnc(circuit):
    ann, soma = _prep()
    n = _add_region(circuit, "VNC", _CACHE["masks"]["VNC"], ann, soma)
    _edge_group(circuit, "VNC_C", _CACHE["ids_CEN"], "VNC",
                pre_space="CEN", forward=False)
    _edge_group(circuit, "VNC_R", _CACHE["ids_VNC"], "VNC",
                pre_space="VNC", forward=False)
    _edge_group(circuit, "ASC_R", _CACHE["ids_VNC"], "CEN",
                pre_space="VNC", forward=False)
    _resolve_pre(circuit)
    print(f"vnc: {n:,} cells")
    return circuit


def _prep():
    if "ann" not in _CACHE:
        _CACHE["ann"] = fdata.load_annotations()
        _CACHE["soma"] = fdata.neuron_positions(_CACHE["ann"])
        _CACHE["masks"] = _region_masks(_CACHE["ann"])
        _CACHE["next_id"] = 10_000_000      # clear of VPN's 1..3k band
    return _CACHE["ann"], _CACHE["soma"]


def build_full_cns(circuit):
    build_ol_rest(circuit)
    build_central_brain(circuit)
    build_vnc(circuit)
    return circuit


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "exp015_circuit", ROOT / "experiments" / "exp015_bilateral"
        / "circuit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    circuit = build_full_cns(mod.build_bilateral_circuit())
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()}}
    tot_pairs = sum(v["pairs"] for v in rep["edges"].values())
    tot_syn = sum(v["syn"] for v in rep["edges"].values())
    print(f"extra pairs {tot_pairs:,} / synapses {tot_syn:,}")
    (OUT / "circuit3_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
