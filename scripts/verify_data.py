"""Verify data/ integrity: raw files against official GCS sizes, derived files
against measured row-count facts (HANDOFF §4).

Catches the two failure modes we have actually hit: interrupted copies leaving
missing/partial raw files, and stale derived tables after a rebuild. Run from
repository root:
    python scripts/verify_data.py
Exits non-zero if anything fails.
"""

import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffbm.data import RAW, DERIVED, RAW_URLS, URL_PREFIX

# row counts measured on the completed pipeline (HANDOFF §4, do not edit
# unless data/ was intentionally rebuilt from a different source)
DERIVED_EXPECTED_ROWS = {
    "visual_nodes.parquet": 105_265,
    "visual_edges.parquet": 12_490_360,
    "neuron_sites.parquet": 89_411_645,
}


def check_raw() -> bool:
    ok = True
    for name, remote in RAW_URLS.items():
        dest = RAW / name
        if not dest.exists():
            print(f"FAIL raw/{name}: missing")
            ok = False
            continue
        local = dest.stat().st_size
        req = urllib.request.Request(URL_PREFIX + remote, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_size = int(resp.headers["Content-Length"])
        if local == remote_size:
            print(f"PASS raw/{name}: {local:,} bytes (matches official)")
        else:
            print(f"FAIL raw/{name}: {local:,} bytes, official {remote_size:,}"
                  f" ({local - remote_size:+,})")
            ok = False
    return ok


def check_derived() -> bool:
    ok = True
    for name, expected in DERIVED_EXPECTED_ROWS.items():
        path = DERIVED / name
        if not path.exists():
            print(f"FAIL derived/{name}: missing")
            ok = False
            continue
        n = pq.ParquetFile(path).metadata.num_rows
        if n == expected:
            print(f"PASS derived/{name}: {n:,} rows")
        else:
            print(f"FAIL derived/{name}: {n:,} rows, expected {expected:,}")
            ok = False
    rl = DERIVED / "rl_sites.parquet"
    if rl.exists():
        n = pq.ParquetFile(rl).metadata.num_rows
        print(f"INFO derived/rl_sites.parquet: {n:,} rows (exp002 cache)")
    return ok


if __name__ == "__main__":
    ok = check_raw() and check_derived()
    print("data verification:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
