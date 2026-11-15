"""Item-based k-nearest-neighbours with shrunk cosine similarity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.models.base import Recommender
from recsys.split import TrainData


def shrunk_cosine(X: sp.spmatrix, shrink: float = 0.0) -> np.ndarray:
    """Item-item cosine similarity ``c_ij / (|i| |j| + shrink)`` with a zero diagonal.

    ``c_ij`` counts users who liked both items; the shrinkage term pulls similarities that rest
    on few co-occurrences towards zero.
    """
    C = np.asarray((X.T @ X).todense(), dtype=np.float64)
    norms = np.sqrt(np.diag(C))
    S = C / (np.outer(norms, norms) + shrink + 1e-12)
    np.fill_diagonal(S, 0.0)
    return S


def keep_top_k(S: np.ndarray, k: int) -> np.ndarray:
    """Keep, for every target item (column), only its ``k`` most similar items."""
    n = S.shape[0]
    if k >= n - 1:
        return S.copy()
    rows = np.argpartition(-S, k - 1, axis=0)[:k]
    out = np.zeros_like(S)
    cols = np.broadcast_to(np.arange(n), rows.shape)
    out[rows, cols] = S[rows, cols]
    return out


class ItemKNN(Recommender):
    """Score ``s_uj = sum_{i in history(u)} sim_k(i, j)`` (Sarwar et al. 2001; Deshpande &
    Karypis 2004), where ``sim_k`` keeps the ``k`` nearest neighbours of each item ``j``."""

    name = "ItemKNN"

    def __init__(self, k: int = 100, shrink: float = 10.0) -> None:
        self.k = k
        self.shrink = shrink

    def fit(self, train: TrainData) -> ItemKNN:
        self.X_ = train.X.tocsr()
        self.W_ = keep_top_k(shrunk_cosine(self.X_, self.shrink), self.k)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.W_)
