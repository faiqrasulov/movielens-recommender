"""Top-N ranking metrics and beyond-accuracy metrics, implemented from scratch.

Ranking metrics take ``recs`` (users x k item indices, best first) and ``relevant`` (a boolean
users x items matrix of held-out positives) and return one value per user.
"""

from __future__ import annotations

import numpy as np


def hits_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Boolean users x k matrix: is the item at each rank relevant?"""
    return np.take_along_axis(np.asarray(relevant, dtype=bool), recs, axis=1)


def precision_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Share of the k recommended items that are relevant."""
    return hits_at_k(recs, relevant).mean(axis=1)


def recall_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Share of the user's relevant items that appear in the top k."""
    n_rel = np.asarray(relevant, dtype=bool).sum(axis=1)
    return hits_at_k(recs, relevant).sum(axis=1) / np.maximum(n_rel, 1)


def hit_rate_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """1 if at least one relevant item is in the top k."""
    return hits_at_k(recs, relevant).any(axis=1).astype(np.float64)


def ndcg_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Binary-relevance NDCG: ``DCG = sum_r hit_r / log2(r + 1)`` over ranks ``r = 1..k``,
    normalised by the DCG of a perfect list with ``min(k, n_relevant)`` hits."""
    hits = hits_at_k(recs, relevant)
    k = recs.shape[1]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discount
    n_rel = np.minimum(np.asarray(relevant, dtype=bool).sum(axis=1), k)
    idcg = np.concatenate([[0.0], np.cumsum(discount)])[n_rel]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def average_precision_at_k(recs: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """AP@k: mean of precision@r over the ranks r of hits, divided by ``min(k, n_relevant)``.
    Its mean over users is MAP@k."""
    hits = hits_at_k(recs, relevant).astype(np.float64)
    k = recs.shape[1]
    precision_at_r = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    n_rel = np.minimum(np.asarray(relevant, dtype=bool).sum(axis=1), k)
    return np.divide((precision_at_r * hits).sum(axis=1), n_rel,
                     out=np.zeros(len(recs)), where=n_rel > 0)


RANKING_METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k list."""
    return float(np.unique(recs).size / n_items)


def average_popularity(recs: np.ndarray, item_counts: np.ndarray, n_users: int) -> float:
    """Mean popularity of recommended items, as the share of users who liked the item in
    training (Abdollahpouri et al. 2019's ARP, normalised to [0, 1])."""
    return float(np.mean(np.asarray(item_counts)[recs]) / n_users)


def novelty(recs: np.ndarray, item_counts: np.ndarray, n_users: int) -> float:
    """Mean self-information ``-log2 p(i)`` of recommended items in bits, with
    ``p(i) = (count_i + 1) / (n_users + 1)``; higher means less popular, more novel items."""
    p = (np.asarray(item_counts, dtype=np.float64)[recs] + 1.0) / (n_users + 1.0)
    return float(np.mean(-np.log2(p)))
