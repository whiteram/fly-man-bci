"""Aggregate syn-point coordinates per neuron -> data/derived/neuron_sites.parquet.

Streams the 12.7 GB syn-points feather via pyarrow (memory-mapped), groups by
(body, kind) with the Arrow-native group_by (no pandas materialization), and
writes one row per (body, role) with mean site position in micrometers and the
site count. Run from repository root:

    python scripts/build_neuron_sites.py
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as ft
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffbm.data import RAW, DERIVED, VOXEL_UM


def main():
    src = RAW / "syn-points.feather"
    print(f"reading {src} (memory-mapped) ...")
    t = ft.read_table(src, memory_map=True, columns=["x", "y", "z", "kind", "body"])
    print(f"{t.num_rows:,} sites")

    print("grouping by (body, kind) ...")
    gb = t.group_by(["body", "kind"]).aggregate(
        [("x", "mean"), ("y", "mean"), ("z", "mean"), ("x", "count")]
    )
    df = gb.to_pandas()
    df = df.rename(
        columns={
            "x_mean": "x_um", "y_mean": "y_um", "z_mean": "z_um",
            "x_count": "n_sites",
        }
    )
    for c in ("x_um", "y_um", "z_um"):
        df[c] = df[c] * VOXEL_UM
    df["body"] = df["body"].astype("int64")
    df["kind"] = df["kind"].astype(str)

    out = DERIVED / "neuron_sites.parquet"
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df):,} rows -> {out}")
    print(df.groupby("kind")["body"].count().to_string())


if __name__ == "__main__":
    main()
