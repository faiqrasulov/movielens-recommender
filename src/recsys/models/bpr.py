"""BPR-MF (Rendle et al., 2009): pairwise ranking loss, vectorised mini-batch SGD in NumPy."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from recsys.models.base import EpochCallback, Recommender
from recsys.split import TrainData


def _scatter_add(target: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    """``target[idx] += values`` with repeated indices summed (a fast ``np.add.at``)."""
    m = sp.csr_matrix(
        (np.ones(idx.size), (idx, np.arange(idx.size))), shape=(target.shape[0], idx.size)
    )
    target += np.asarray(m @ values)


class BPR(Recommender):
    """Matrix factorisation with item bias trained on sampled (user, positive, negative) triples.

    Maximises ``sum ln sigmoid(x_ui - x_uj) - reg |theta|^2`` with ``x_ui = p_u . q_i + b_i``.
    Each epoch visits every training positive once (shuffled); the negative item is drawn
    uniformly from the items the user has not liked.
    """

    name = "BPR"

    def __init__(
        self,
        factors: int = 64,
        lr: float = 0.05,
        reg: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        init_std: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_std = init_std
        self.seed = seed

    def fit(self, train: TrainData, callback: EpochCallback | None = None) -> BPR:
        X = train.X.tocoo()
        n_users, n_items = X.shape
        users, items = X.row.astype(np.int64), X.col.astype(np.int64)
        liked = train.X.toarray() > 0  # dense lookup table for rejecting false negatives
        rng = np.random.default_rng(self.seed)
        f = self.factors
        # The item bias is stored as an extra item column matched by a constant-1 user column,
        # so one dot product gives p_u . q_i + b_i and one scatter updates factors and bias.
        P = np.ones((n_users, f + 1))
        P[:, :f] = rng.normal(0.0, self.init_std, (n_users, f))
        Q = np.zeros((n_items, f + 1))
        Q[:, :f] = rng.normal(0.0, self.init_std, (n_items, f))
        self.user_factors_, self.item_factors_, self.item_bias_ = P[:, :f], Q[:, :f], Q[:, f]
        self._P, self._Q = P, Q
        self.loss_history_: list[float] = []

        for epoch in range(self.epochs):
            order = rng.permutation(users.size)
            total = 0.0
            for start in range(0, order.size, self.batch_size):
                batch = order[start : start + self.batch_size]
                u, i = users[batch], items[batch]
                j = rng.integers(0, n_items, batch.size)
                clash = liked[u, j]
                for _ in range(50):  # bounded, in case a user has liked (nearly) every item
                    if not clash.any():
                        break
                    j[clash] = rng.integers(0, n_items, int(clash.sum()))
                    clash = liked[u, j]
                pu, diff = P[u], Q[i] - Q[j]
                x = np.einsum("bf,bf->b", pu, diff)
                total += float(np.sum(np.logaddexp(0.0, -x)))
                g = (self.lr * expit(-x))[:, None]  # lr * d ln sigmoid(x) / dx
                decay = self.lr * self.reg
                grad_u = g * diff - decay * pu
                grad_u[:, f] = 0.0  # the constant-1 column stays fixed
                grad_i = g * pu - decay * Q[i]
                grad_j = -g * pu - decay * Q[j]
                _scatter_add(P, u, grad_u)
                _scatter_add(Q, np.concatenate([i, j]), np.concatenate([grad_i, grad_j]))
            self.loss_history_.append(total / users.size)
            if callback is not None:
                callback(epoch + 1, self)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self._P[users] @ self._Q.T
