import numpy as np
import pytest
import scipy.sparse as sp

from recsys.evaluate import (
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
    recommend,
    top_k,
)
from recsys.explain import contributions, explain_recommendations
from recsys.models import EASE, ItemKNN, MostPopular
from recsys.split import final_task, per_user_temporal_split


def test_top_k_excludes_seen_and_breaks_ties_by_index():
    scores = np.array([[5.0, 9.0, 9.0, 1.0, 9.0]])
    seen = sp.csr_matrix(np.array([[0, 1, 0, 0, 0]]))
    np.testing.assert_array_equal(top_k(scores, 3, seen), [[2, 4, 0]])
    np.testing.assert_array_equal(top_k(scores, 2), [[1, 2]])


def test_recommendations_never_contain_seen_items(toy_data):
    task = final_task(toy_data, per_user_temporal_split(toy_data))
    for model in (MostPopular(), EASE(lam=10.0)):
        model.fit(task.train)
        recs = recommend(model, task.users, task.seen, k=10, batch=7)  # several batches
        seen = task.seen[task.users].toarray() > 0
        assert not np.take_along_axis(seen, recs, axis=1).any()
        assert all(len(set(r)) == 10 for r in recs)


def test_evaluate_beats_popularity_on_structured_data(toy_data):
    task = final_task(toy_data, per_user_temporal_split(toy_data))
    pop = evaluate(MostPopular().fit(task.train), task)
    ease = evaluate(EASE(lam=5.0).fit(task.train), task)
    assert ease.mean("ndcg") > pop.mean("ndcg")
    assert set(pop.per_user) == {"precision", "recall", "ndcg", "map", "hit_rate"}


def test_bootstrap_intervals():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 400)
    lo, hi = bootstrap_ci(x, n_boot=1000)
    se = x.std(ddof=1) / np.sqrt(x.size)
    assert lo < x.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.15)
    res = paired_bootstrap(x + 0.5, x, n_boot=500)
    assert res["diff"] == pytest.approx(0.5)
    assert res["ci_low"] == pytest.approx(0.5) and res["p_value"] == 0.0
    noise = paired_bootstrap(x, x + rng.normal(0, 0.01, x.size), n_boot=500)
    assert noise["ci_low"] < 0 < noise["ci_high"]


@pytest.mark.parametrize("model", [EASE(lam=5.0), ItemKNN(k=10, shrink=5.0)])
def test_explanations_add_up_to_the_score(toy_data, model):
    task = final_task(toy_data, per_user_temporal_split(toy_data))
    model.fit(task.train)
    user = int(task.users[0])
    recs = recommend(model, np.array([user]), task.seen, k=3)[0]
    scores = model.score(np.array([user]))[0]
    for entry in explain_recommendations(model, user, recs, toy_data.titles):
        history, contrib = contributions(model, user, entry["item"])
        assert entry["score"] == pytest.approx(scores[entry["item"]])
        assert contrib.sum() == pytest.approx(scores[entry["item"]])
        assert entry["because"][0]["contribution"] == pytest.approx(contrib.max())
        assert entry["because"][0]["item"] in history
