"""exp019: chemosensory region builders (olfactory / gustatory).

Adds the two remaining sensory INPUT channels of the thought
experiment as independently switchable regions (ffbm.regions):

  olfactory   ORN   olfactory receptor neurons, class "olfactory"
                    (2,639: 2,455 antennal nerve + 184 maxillary-labial;
                    types ORN_<glomerulus>, e.g. ORN_DA1 = Or67d/cVA)
  gustatory   GRN   gustatory receptor neurons, class "gustatory"
                    (1,428; types carry the ORGAN, not the quality:
                    LgLG* labellum, WG* wing, claw_tpGRN tarsus)

Wiring (connectome-weights, weight >= 5, consensus-nt signs; same
convention as exp017): one edge group per POSTSYNAPTIC id space so the
compact remaps can never collide inside a group,

  ORN_R  ORN->ORN   (local, 9.7k syn; forward=False: peripheral axons)
  ORN_C  ORN->CEN   (the antennal-lobe circuit: ALLN+ALPN+ALON+ALIN
                     are cb_intrinsic, i.e. already simulated inside the
                     central_brain region -- ORN terminals in the AL are
                     the presynaptic side of that dipole; forward=True)
  GRN_R  GRN->GRN   (local, 12.4k syn; forward=False)
  GRN_C  GRN->CEN   (SEZ circuit: SEZPN + cb_* partners; forward=True)
  GRN_V  GRN->VNC   (tarsal/abdominal reflex arcs; forward=False, only
                     when the vnc region is ON)

Both sensory classes have NO soma positions in MaleCNS (somaLocation
empty for peripheral somata); positions come from the mean PRESYNAPTIC
synapse-site positions (fdata.site_positions, "PreSyn") -- the central
terminal arbor, which is exactly the current source of the ORN/GRN
dipole onto its central partner. All 2,639 ORN and 1,428 GRN have one.

Stimulus interface: builders store circuit["chem_groups"][pop] =
{annotation type -> compact index array} so viz/export_data.py
--chem-input can turn a chem_inputs.json channel spec (type pattern +
pulse train) into per-neuron current increments (see
docs/SENSORY_INPUTS.md).

Run from repository root:
    python experiments/exp019_chemosense/circuit4.py      # report
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

# compact id band: above every raw MaleCNS bodyId (max 1.57e9) and
# clear of exp017's 10_000_000..10_200_000 band
_ID_BASE = 2_000_000_000

_CACHE = {}


def _prep():
    """One heavy load per process: annotations, terminal positions,
    filtered weights, neurotransmitter signs."""
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
    sig = _CACHE["sig"]
    return pd.Series(raw_ids).map(sig).fillna(1.0).to_numpy(np.float32)


def _region_raw_ids(ann, mask):
    """Sorted raw ids of a region, EXACTLY the list exp017 used
    (soma-bearing cells only) -- lets us reconstruct exp017's raw ->
    compact remap from the cached circuit's sorted compact ids."""
    soma = _CACHE["soma"]
    return np.array(sorted(ann.loc[mask & ann["bodyId"].isin(soma),
                                  "bodyId"].tolist()), dtype=np.int64)


def _add_sensory_pop(circuit, name, class_name, i_key):
    """Append one sensory-input population: compact ids, terminal
    positions, chem_groups (annotation type -> index array)."""
    ann, sites = _prep()
    m = ann["class"] == class_name
    raw = np.array(sorted(ann.loc[m, "bodyId"].tolist()), dtype=np.int64)
    missing = [int(b) for b in raw if int(b) not in sites]
    assert not missing, f"{name}: {len(missing)} cells without terminals"
    base = _CACHE["next_id"]
    remap = {int(old): base + i for i, old in enumerate(raw)}
    _CACHE["next_id"] = base + len(raw)
    _CACHE[f"raw_{name}"] = raw
    _CACHE[f"remap_{name}"] = remap
    _CACHE[f"ids_{name}"] = np.array(sorted(remap.values()),
                                     dtype=np.int64)

    pp, qq = circuit["pre_pos"], circuit["post_pos"]
    for old, new in remap.items():
        p = sites[old]
        pp[new] = p
        qq[new] = p
    pops = circuit.get("extra_pops") or {}
    pops[name] = {"ids": _CACHE[f"ids_{name}"],
                  "tau_ms": 10.0, "t_refrac_ms": 2.0,
                  "i_base": i_key, "ou_sigma": CAL["OU_VPN_CB"]}
    circuit["extra_pops"] = pops

    # chem_groups: annotation type -> indices into pops[name]["ids"]
    # (sorted compact ids -- index == position in the sorted array);
    # side-qualified duplicates "<type>@<rootSide>" are added for the
    # stereo paradigms (bci/stereo) -- type-only keys stay so existing
    # chem entries (ORN_DA1, LgLG*, claw_tpGRN, ...) keep matching
    sub = ann.loc[m, ["bodyId", "type", "rootSide"]].copy()
    sub["compact"] = sub["bodyId"].map(remap)
    groups = {}
    for tname, grp in sub.groupby("type"):
        groups[str(tname)] = np.searchsorted(
            _CACHE[f"ids_{name}"], grp["compact"].to_numpy(np.int64))
    for (tname, sd), grp in sub.groupby(
            [sub["type"].fillna("?"), sub["rootSide"].fillna("?")]):
        groups[f"{tname}@{sd}"] = np.searchsorted(
            _CACHE[f"ids_{name}"], grp["compact"].to_numpy(np.int64))
    chem_groups = circuit.get("chem_groups") or {}
    chem_groups[name] = groups
    circuit["chem_groups"] = chem_groups
    return len(raw)


def _edge_group(circuit, gname, pre_raw_ids, post_pop, pre_space,
                forward):
    """One edge group pre (sensory pop, compact) -> post (compact).

    The connectome table is in RAW body ids, so the post filter uses
    the RAW id list; only the table columns carry compact remaps."""
    pre, post, wt = _CACHE["w"]
    if post_pop in ("CEN", "VNC"):
        # exp017 built the CEN/VNC remap in ITS process cache; the
        # mapping is reconstructible because compact ids were assigned
        # in sorted-raw order: zip(sorted raw, sorted compact)
        raw_post = _region_raw_ids(_CACHE["ann"], _CACHE["mask_CEN"]
                                   if post_pop == "CEN"
                                   else _CACHE["mask_VNC"])
        compact = circuit["extra_pops"][post_pop]["ids"]
        assert len(raw_post) == len(compact), \
            f"{post_pop}: raw/compact length mismatch"
        rmap = dict(zip(raw_post.tolist(), compact.tolist()))
    else:
        raw_post = _CACHE[f"raw_{post_pop}"]
        rmap = _CACHE[f"remap_{post_pop}"]
    m = np.isin(pre, pre_raw_ids) & np.isin(post, raw_post)
    post_col = pd.Series(post[m]).map(rmap)
    pre_col = pd.Series(pre[m]).map(_CACHE[f"remap_{pre_space}"])
    e = pd.DataFrame({
        "body_pre": pre_col.to_numpy(np.int64),
        "body_post": post_col.to_numpy(np.int64),
        "weight": wt[m].astype(np.float32),
        "sign": _signs_for(pre[m])})
    e = e.sort_values(["body_pre", "body_post"]).reset_index(drop=True)
    edges = circuit.get("extra_edges") or {}
    edges[gname] = {"pre": (pre_space,), "post": post_pop,
                    "table": e, "tau_s": CAL["TAU_V_MS"],
                    "g_unit": "G_UNIT_CX", "forward": forward}
    circuit["extra_edges"] = edges
    return len(e), float(e["weight"].sum())


def _prep_masks():
    """exp017's CEN/VNC masks + soma dict (for remap reconstruction)."""
    ann, _ = _prep()
    if "mask_CEN" not in _CACHE:
        _CACHE["soma"] = fdata.neuron_positions(ann)
        sc = ann["superclass"].fillna("")
        _CACHE["mask_CEN"] = sc.str.startswith("cb_") \
            | (sc == "descending_neuron")
        _CACHE["mask_VNC"] = sc.str.startswith("vnc_") | sc.isin(
            ["ascending_neuron", "sensory_ascending",
             "sensory_ascending_tbc", "vnc_tbc", "vnc_sensory_tbc",
             "efferent_ascending"])
    return ann


def build_olfactory(circuit):
    ann = _prep_masks()
    n = _add_sensory_pop(circuit, "ORN", "olfactory", "I_ORN_BASE")
    k, s = _edge_group(circuit, "ORN_R", _CACHE["raw_ORN"], "ORN",
                       "ORN", forward=False)
    print(f"  ORN_R ORN->ORN: {k:,} pairs / {s:,.0f} syn")
    k, s = _edge_group(circuit, "ORN_C", _CACHE["raw_ORN"], "CEN",
                       "ORN", forward=True)
    print(f"  ORN_C ORN->CEN: {k:,} pairs / {s:,.0f} syn")
    print(f"olfactory: {n:,} ORN")
    return circuit


def build_gustatory(circuit):
    ann = _prep_masks()
    n = _add_sensory_pop(circuit, "GRN", "gustatory", "I_GRN_BASE")
    k, s = _edge_group(circuit, "GRN_R", _CACHE["raw_GRN"], "GRN",
                       "GRN", forward=False)
    print(f"  GRN_R GRN->GRN: {k:,} pairs / {s:,.0f} syn")
    k, s = _edge_group(circuit, "GRN_C", _CACHE["raw_GRN"], "CEN",
                       "GRN", forward=True)
    print(f"  GRN_C GRN->CEN: {k:,} pairs / {s:,.0f} syn")
    if "VNC" in (circuit.get("extra_pops") or {}):
        k, s = _edge_group(circuit, "GRN_V", _CACHE["raw_GRN"], "VNC",
                           "GRN", forward=False)
        print(f"  GRN_V GRN->VNC: {k:,} pairs / {s:,.0f} syn")
    else:
        print("  GRN_V skipped (vnc region OFF)")
    print(f"gustatory: {n:,} GRN")
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
    circuit = m3.build_vnc(circuit)          # exercises the GRN_V group
    circuit = build_olfactory(circuit)
    circuit = build_gustatory(circuit)
    rep = {"cells": {n: len(s["ids"]) for n, s in
                     circuit["extra_pops"].items()},
           "edges": {g: {"pairs": len(s["table"]),
                         "syn": int(s["table"]["weight"].sum()),
                         "post": s["post"], "forward": s["forward"]}
                     for g, s in circuit["extra_edges"].items()},
           "chem_groups": {p: {t: len(v) for t, v in g.items()}
                           for p, g in circuit["chem_groups"].items()}}
    (OUT / "circuit4_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
