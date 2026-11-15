import numpy as np
import pytest

from recsys.metrics import (
    average_popularity,
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)

# Three users, catalogue of 8 items, k = 4.
RECS = np.array([[0, 1, 2, 3], [5, 6, 7, 4], [0, 1, 2, 3]])
RELEVANT = np.zeros((3, 8), dtype=bool)
RELEVANT[0, [1, 3, 7]] = True  # hits at ranks 2 and 4, one relevant item missed
RELEVANT[1, [5]] = True  # hit at rank 1, the only relevant item
RELEVANT[2, [6]] = True  # no hit


def test_precision_recall_hit_rate_known_answers():
    np.testing.assert_allclose(precision_at_k(RECS, RELEVANT), [2 / 4, 1 / 4, 0])
    np.testing.assert_allclose(recall_at_k(RECS, RELEVANT), [2 / 3, 1, 0])
    np.testing.assert_allclose(hit_rate_at_k(RECS, RELEVANT), [1, 1, 0])


def test_ndcg_known_answer():
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)  # min(k, 3 relevant) = 3 ideal hits
    np.testing.assert_allclose(ndcg_at_k(RECS, RELEVANT), [dcg / idcg, 1.0, 0.0])


def test_average_precision_known_answer():
    # user 0: precision 1/2 at rank 2 and 2/4 at rank 4, normalised by min(4, 3) = 3
    np.testing.assert_allclose(average_precision_at_k(RECS, RELEVANT), [(0.5 + 0.5) / 3, 1, 0])


def test_ndcg_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    scores = rng.random((50, 30))
    relevant = rng.random((50, 30)) < 0.15
    relevant[relevant.sum(axis=1) == 0, 0] = True
    recs = np.argsort(-scores, axis=1)[:, :10]
    ours = ndcg_at_k(recs, relevant)
    ref = [sklearn_metrics.ndcg_score(relevant[[u]].astype(float), scores[[u]], k=10)
           for u in range(50)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_beyond_accuracy_metrics():
    counts = np.array([10, 5, 0, 1, 2, 3, 4, 7])
    recs = np.array([[0, 1], [0, 2]])
    assert catalog_coverage(recs, 8) == pytest.approx(3 / 8)
    assert average_popularity(recs, counts, n_users=10) == pytest.approx((10 + 5 + 10 + 0) / 40)
    expected = np.mean(-np.log2((counts[recs] + 1) / 11))
    assert novelty(recs, counts, n_users=10) == pytest.approx(expected)
