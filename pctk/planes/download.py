"""Helpers to fetch the real FETAL_PLANES_DB.

The dataset's canonical home is Zenodo (record 3904280); it is also mirrored on
Kaggle. Both require a (free) account:

  * **Kaggle** needs an API token at ``~/.kaggle/kaggle.json`` (Kaggle ->
    Account -> Create New API Token), then this uses the ``kaggle`` CLI.
  * **Zenodo** is a direct download of a ~2 GB zip.

These are thin, dependency-light wrappers; nothing here trains or looks at sex.
"""

from __future__ import annotations

import os
import subprocess
import sys

ZENODO_RECORD = "3904280"
ZENODO_FILE = "FETAL_PLANES_ZENODO.zip"
DEFAULT_KAGGLE_SLUG = "fatemehsarhaddi/fetal-planes-db"  # a known mirror


def download_kaggle(dest: str, slug: str = DEFAULT_KAGGLE_SLUG,
                    unzip: bool = True) -> str:
    """Download a Kaggle dataset with the ``kaggle`` CLI into ``dest``."""
    os.makedirs(dest, exist_ok=True)
    token = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(token) and not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError(
            "Kaggle credentials not found. Create an API token at "
            "https://www.kaggle.com/settings -> 'Create New API Token' and save "
            "it to ~/.kaggle/kaggle.json (chmod 600).")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", dest]
    if unzip:
        cmd.append("--unzip")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def download_zenodo(dest: str, record: str = ZENODO_RECORD,
                    filename: str = ZENODO_FILE) -> str:
    """Download the FETAL_PLANES_DB zip directly from Zenodo (~2 GB)."""
    import urllib.request

    os.makedirs(dest, exist_ok=True)
    url = f"https://zenodo.org/record/{record}/files/{filename}?download=1"
    out = os.path.join(dest, filename)
    print(f"downloading {url}\n  -> {out}  (this is large)")

    def _hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, out, reporthook=_hook)
    print()
    return out
