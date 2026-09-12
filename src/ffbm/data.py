"""Loaders for raw MaleCNS v1.0 files and derived tables."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

# official male-cns v1.0 flat-connectome files (public, no token)
RAW_URLS = {
    "body-annotations.feather":
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters.feather":
        "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights.feather":
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "syn-points.feather":
        "syn-points-male-cns-v1.0-minconf-0.5.feather",
}
URL_PREFIX = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
              "connectome-data/flat-connectome/")

VISUAL_SUPERCLASSES = (
    "ol_intrinsic",        # optic lobe intrinsic (lamina/medulla/lobula/lobula plate)
    "visual_projection",   # VPN: optic lobe -> central brain
    "ol_sensory",          # photoreceptors
    "visual_centrifugal",  # central brain -> optic lobe feedback
)

# somaLocation entries are voxel indices of the aligned 8 nm EM volume
VOXEL_UM = 0.008


def load_annotations() -> pd.DataFrame:
    return pd.read_feather(RAW / "body-annotations.feather")


def load_visual_nodes() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "visual_nodes.parquet")


def load_visual_edges() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "visual_edges.parquet")


def load_neuron_sites() -> pd.DataFrame:
    """Per-(body, kind) mean synapse-site positions (um) and site counts."""
    return pd.read_parquet(DERIVED / "neuron_sites.parquet")


def site_positions(sites: pd.DataFrame, kind: str) -> dict[int, np.ndarray]:
    """bodyId -> mean synapse-site position (um) for one role ('PreSyn'/'PostSyn')."""
    s = sites[sites["kind"] == kind]
    return dict(zip(s["body"].astype(int).tolist(),
                    s[["x_um", "y_um", "z_um"]].to_numpy()))


def neuron_positions(ann: pd.DataFrame) -> dict[int, np.ndarray]:
    """bodyId -> soma (x, y, z) in micrometers (from somaLocation voxels)."""
    sl = ann[ann["somaLocation"].notna()]
    coords = np.array(
        [ast.literal_eval(s) if isinstance(s, str) else s for s in sl["somaLocation"]],
        dtype=float,
    )
    coords *= VOXEL_UM
    return dict(zip(sl["bodyId"].astype(int).tolist(), coords))
