"""Synthetic-circuit tests for the region-optional extra_pops /
extra_edges mechanism (ffbm.pipeline) and the ffbm.regions switch."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import pipeline as fp
from ffbm import regions as freg


def _tiny_circuit(with_region=True):
    """Minimal visual cascade (2 R, 2 L, 3 MID, 2 T45) + optional
    region X fed by MID/T45 and region Y fed by X."""
    ids = {"r": np.array([10, 11]), "l": np.array([20, 21]),
           "mid": np.array([30, 31, 32]), "t45": np.array([40, 41])}
    rng = np.random.default_rng(0)
    pos = {b: rng.uniform(0, 10, 3) for group in ids.values() for b in group}

    def tab(pre, post, w=5.0):
        return pd.DataFrame({"body_pre": np.asarray(pre, dtype=np.int64),
                             "body_post": np.asarray(post, dtype=np.int64),
                             "weight": np.full(len(pre), w, np.float32),
                             "sign": np.ones(len(pre), np.float32)})

    circuit = {
        "r_ids": ids["r"], "l_ids": ids["l"], "mid_ids": ids["mid"],
        "t45_ids": ids["t45"],
        "r_type": np.array(["R1-R6"] * 2), "l_type": np.array(["L1"] * 2),
        "mid_type": np.array(["Mi1"] * 2 + ["Tm3"]),
        "t45_type": np.array(["T4a", "T5a"]),
        "e_rl": tab([10, 11], [20, 21]), "e_lm": tab([20, 21], [30, 31]),
        "e_mt": {"Mi1": tab([30, 31], [40, 41]),
                 "Tm3": tab([32], [41]),
                 "Mi4": tab([30], [40]).iloc[0:0],
                 "Mi9": tab([30], [40]).iloc[0:0],
                 "Tm1": tab([30], [40]).iloc[0:0],
                 "Tm2": tab([30], [40]).iloc[0:0],
                 "Tm4": tab([30], [40]).iloc[0:0],
                 "Tm9": tab([30], [40]).iloc[0:0]},
        "pre_pos": dict(pos), "post_pos": dict(pos),
    }
    if with_region:
        for b in (50, 51, 60, 61, 62):
            pos[b] = rng.uniform(0, 10, 3)
        circuit["pre_pos"] = dict(pos)
        circuit["post_pos"] = dict(pos)
        circuit["extra_pops"] = {
            "X": {"ids": np.array([50, 51]), "tau_ms": 10.0,
                  "t_refrac_ms": 2.0, "i_base": 500.0,
                  "ou_sigma": 40.0},
            "Y": {"ids": np.array([60, 61, 62]), "tau_ms": 10.0,
                  "t_refrac_ms": 2.0, "i_base": 0.0, "ou_sigma": 40.0},
        }
        circuit["extra_edges"] = {
            "MIDX": {"pre": ("MID", "T45"), "post": "X",
                     "table": tab([30, 40], [50, 51], w=50.0),
                     "tau_s": 8.0, "g_unit": 0.05},
            "XY": {"pre": ("X",), "post": "Y",
                   "table": tab([50, 51], [60, 61], w=50.0),
                   "tau_s": 8.0, "g_unit": 0.05},
        }
    return circuit


def test_extra_region_builds_and_propagates():
    circuit = _tiny_circuit(True)
    seen = {}
    cal = dict(fp.CAL)

    def record(j, k, t, st, inc_f, sp):
        seen.setdefault("spikes", []).append(sp)
        seen["st"] = st

    fp.simulate(circuit, cal, lambda t: 0.0, seed=0, t_end_ms=200.0,
                on_sample=record)
    st = seen["st"]
    assert "X" in st["pops"] and "Y" in st["pops"]
    assert "MIDX" in st["syn"] and "XY" in st["syn"]
    spikes = seen["spikes"]
    assert "X" in spikes[0] and "Y" in spikes[0]
    # X is strongly driven (i_base 500 pA) -> must spike; Y only via X
    assert any(s["X"].any() for s in spikes)
    assert sum(s["Y"].sum() for s in spikes) <= sum(s["X"].sum()
                                                    for s in spikes) * 3


def test_no_region_is_zero_cost():
    circuit = _tiny_circuit(False)
    st = fp.simulate(circuit, dict(fp.CAL), lambda t: 0.0, seed=0,
                     t_end_ms=50.0)
    assert st["extra_pops"] == {}
    assert all("X" not in k for k in st["syn"])
    assert "X" not in st["pops"]


def test_regions_registry_switch():
    assert freg.DEFAULT_REGIONS["visual_bilateral"] is True
    assert freg.DEFAULT_REGIONS["vpn_central"] is False
    # unknown region names are rejected
    try:
        freg.build_circuit({"hypothalamus": True})
        raised = False
    except ValueError:
        raised = True
    assert raised
