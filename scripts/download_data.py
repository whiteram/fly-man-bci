"""Download the three MaleCNS v1.0 raw files (~1.1 GB total) into data/raw/.

Public GCS links, no token required. Files already present are skipped unless
--force is given.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffbm.data import RAW, RAW_URLS, URL_PREFIX


def download(name: str, remote: str, force: bool = False) -> None:
    dest = RAW / name
    if dest.exists() and not force:
        print(f"skip (exists): {dest.name}")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    url = URL_PREFIX + remote
    print(f"downloading {url}")
    result = subprocess.run(["curl", "-sL", "--fail", "-o", str(dest), url])
    if result.returncode != 0:
        raise SystemExit(f"download failed: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name, remote in RAW_URLS.items():
        download(name, remote, args.force)
    print("done.")
