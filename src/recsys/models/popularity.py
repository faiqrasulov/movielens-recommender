"""Popularity baselines: all-time most popular and recent (time-windowed) popularity."""

from __future__ import annotations

import numpy as np

from recsys.models.base import Recommender
from recsys.split import TrainData

DAY = 86_400


class MostPopular(Recommender):
    """Recommend the items with the most training positives to everyone."""

    name = "MostPopular"

    def fit(self, train: TrainData) -> MostPopular:
        self.counts_ = np.asarray(train.X.sum(axis=0), dtype=np.float64).ravel()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.counts_, (len(users), 1))


class RecentPopular(Recommender):
    """Items most often liked in the ``window_days`` before each user's recommendation time.

    Only training events strictly before ``train.asof[user]`` are counted, so this baseline never
    looks into the future, even under a per-user split. Ties (for example items with no recent
    positives) are broken by the item's count over all history before ``asof``.
    ``window_days=None`` counts all history before ``asof``.
    """

    name = "RecentPopular"

    def __init__(self, window_days: float | None = 30) -> None:
        self.window_days = window_days

    def fit(self, train: TrainData) -> RecentPopular:
        order = np.argsort(train.timestamp, kind="stable")
        self.times_ = train.timestamp[order]
        self.items_ = train.item[order]
        self.asof_ = train.asof
        self.n_items_ = train.n_items
        return self

    def _counts(self, lo: int, hi: int) -> np.ndarray:
        return np.bincount(self.items_[lo:hi], minlength=self.n_items_).astype(np.float64)

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.empty((len(users), self.n_items_))
        cache: dict[tuple[int, int], np.ndarray] = {}
        for row, u in enumerate(users):
            hi = int(np.searchsorted(self.times_, self.asof_[u], side="left"))
            lo = 0
            if self.window_days is not None:
                start = self.asof_[u] - self.window_days * DAY
                lo = int(np.searchsorted(self.times_, start, side="left"))
            if (lo, hi) not in cache:
                tie_break = self._counts(0, hi) / (hi + 1.0)  # in [0, 1): never beats a count
                cache[(lo, hi)] = self._counts(lo, hi) + tie_break
            out[row] = cache[(lo, hi)]
        return out
