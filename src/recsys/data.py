"""Download, cache and load MovieLens 100K (GroupLens).

The dataset licence forbids redistribution, so it is fetched at runtime from the official
GroupLens server into ``data/`` and never committed.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_MD5 = "0e33842e24a9c977be4e0107933c0723"
_FILES = ("ml-100k/u.data", "ml-100k/u.item")


@dataclass(frozen=True)
class Interactions:
    """Ratings as parallel arrays plus item titles.

    ``user`` and ``item`` are contiguous 0-based indices; ``titles[i]`` is the title of item ``i``.
    """

    user: np.ndarray
    item: np.ndarray
    rating: np.ndarray
    timestamp: np.ndarray
    titles: np.ndarray
    n_users: int
    n_items: int

    def __len__(self) -> int:
        return int(self.user.size)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_ml100k(data_dir: str | Path = "data") -> Path:
    """Download and extract MovieLens 100K once; return the directory holding ``u.data``.

    Subsequent calls use the cached copy. Raises ``RuntimeError`` with a clear message when the
    archive is missing and the network is unavailable.
    """
    data_dir = Path(data_dir)
    target = data_dir / "ml-100k"
    if all((data_dir / f).exists() for f in _FILES):
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-100k.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(ML100K_URL, timeout=60) as resp, tmp.open("wb") as out:
                while block := resp.read(1 << 20):
                    out.write(block)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download MovieLens 100K from {ML100K_URL}. The first run needs "
                f"internet access; afterwards the copy cached in '{data_dir}/' is used. "
                f"Cause: {exc!r}"
            ) from exc
        tmp.replace(archive)
    digest = _md5(archive)
    if digest != ML100K_MD5:
        archive.unlink()
        raise RuntimeError(f"ml-100k.zip checksum mismatch ({digest}); deleted, please retry.")
    with zipfile.ZipFile(archive) as zf:
        for name in _FILES:
            zf.extract(name, data_dir)
    return target


def load_ml100k(data_dir: str | Path = "data") -> Interactions:
    """Load MovieLens 100K ratings (943 users, 1,682 movies) with 0-based indices and titles."""
    root = download_ml100k(data_dir)
    ratings = pd.read_csv(
        root / "u.data",
        sep="\t",
        header=None,
        names=["user", "item", "rating", "timestamp"],
        dtype="int64",
    )
    items = pd.read_csv(
        root / "u.item",
        sep="|",
        header=None,
        usecols=[0, 1],
        names=["item", "title"],
        dtype={"item": "int64", "title": "object"},
        encoding="latin-1",
    )
    n_items = int(items["item"].max())
    titles = np.full(n_items, "", dtype=object)
    titles[items["item"].to_numpy() - 1] = items["title"].to_numpy(dtype=object)
    return Interactions(
        user=ratings["user"].to_numpy() - 1,
        item=ratings["item"].to_numpy() - 1,
        rating=ratings["rating"].to_numpy().astype(np.int8),
        timestamp=ratings["timestamp"].to_numpy(),
        titles=titles,
        n_users=int(ratings["user"].max()),
        n_items=n_items,
    )
