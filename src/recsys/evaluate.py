"""Turn scores into top-k lists (excluding seen items) and evaluate them, with bootstrap CIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.metrics import (
    RANKING_METRICS,
    average_popularity,
    catalog_coverage,
    novelty,
)
from recsys.models.base import Recommender
from recsys.split import Task


def top_k(scores: np.ndarray, k: int, exclude: sp.csr_matrix | None = None) -> np.ndarray:
    """Indices of the ``k`` highest scores per row, best first; ``exclude`` marks items that
    must never be returned. Ties go to the lower item index (stable sort), so the output does
    not depend on the NumPy version."""
    scores = np.array(scores, dtype=np.float64)
    if exclude is not None:
        coo = exclude.tocoo()
        scores[coo.row, coo.col] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def recommend(
    model: Recommender, users: np.ndarray, seen: sp.csr_matrix, k: int = 10, batch: int = 512
) -> np.ndarray:
    """Top-k recommendations for ``users`` that exclude every item in ``seen``."""
    out = np.empty((len(users), k), dtype=np.int64)
    for start in range(0, len(users), batch):
        u = users[start : start + batch]
        out[start : start + len(u)] = top_k(model.score(u), k, seen[u])
    return out


@dataclass(frozen=True)
class Evaluation:
    """Per-user ranking metrics plus the recommendation lists they were computed from."""

    users: np.ndarray
    recs: np.ndarray
    per_user: dict[str, np.ndarray]

    def mean(self, metric: str) -> float:
        return float(np.mean(self.per_user[metric]))


def evaluate(model: Recommender, task: Task, k: int = 10) -> Evaluation:
    """Recommend to every evaluated user of ``task`` and compute the ranking metrics."""
    recs = recommend(model, task.users, task.seen, k)
    relevant = task.truth[task.users].toarray() > 0
    per_user = {name: fn(recs, relevant) for name, fn in RANKING_METRICS.items()}
    return Evaluation(users=task.users, recs=recs, per_user=per_user)


def beyond_accuracy(ev: Evaluation, task: Task) -> dict[str, float]:
    """Catalogue coverage, average popularity and novelty of the recommendation lists.

    Popularity is measured on the fitting data: the share of users who liked each item."""
    counts = np.asarray(task.train.X.sum(axis=0)).ravel()
    n_users = task.train.n_users
    return {
        "coverage": catalog_coverage(ev.recs, task.train.n_items),
        "avg_popularity": average_popularity(ev.recs, counts, n_users),
        "novelty": novelty(ev.recs, counts, n_users),
    }


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling users with replacement."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, (n_boot, values.size))
    means = values[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """Bootstrap the mean per-user difference ``a - b`` (same users, resampled jointly).

    ``p_value`` is the two-sided share of bootstrap means on the other side of zero."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, (n_boot, diff.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    p = 2.0 * min(np.mean(means <= 0.0), np.mean(means >= 0.0))
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p))}
