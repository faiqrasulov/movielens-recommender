"""EASE^R: embarrassingly shallow autoencoder (Steck, 2019), closed form."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.models.base import Recommender
from recsys.split import TrainData


def ease_weights(X: sp.spmatrix, lam: float) -> np.ndarray:
    """Solve ``min_B ||X - XB||_F^2 + lam ||B||_F^2`` subject to ``diag(B) = 0``.

    With ``P = (X^T X + lam I)^{-1}`` the solution is ``B_ij = -P_ij / P_jj`` (``i != j``).
    """
    G = np.asarray((X.T @ X).todense(), dtype=np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    return B


class EASE(Recommender):
    """Linear item-item model; the score is ``x_u B``, so every score decomposes exactly into
    one contribution per item in the user's history."""

    name = "EASE"

    def __init__(self, lam: float = 300.0) -> None:
        self.lam = lam

    def fit(self, train: TrainData) -> EASE:
        self.X_ = train.X.tocsr()
        self.W_ = ease_weights(self.X_, self.lam)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.W_)
