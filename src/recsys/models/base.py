"""Common interface of all recommenders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from recsys.split import TrainData

EpochCallback = Callable[[int, "Recommender"], None]


class Recommender(ABC):
    """Fit on training positives, then return a dense score for every (user, item) pair."""

    name: str = "recommender"

    @abstractmethod
    def fit(self, train: TrainData) -> Recommender:
        """Learn from ``train`` and return ``self``."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Scores of shape ``(len(users), n_items)``; higher means recommend first."""
