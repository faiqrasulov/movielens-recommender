import numpy as np
import pytest
import scipy.sparse as sp

from recsys.models import BPR, EASE, ImplicitALS, ItemKNN, MostPopular, RecentPopular
from recsys.models.als import implicit_objective, solve_cg, solve_exact
from recsys.models.ease import ease_weights
from recsys.models.itemknn import keep_top_k, shrunk_cosine
from recsys.split import TrainData

DAY = 86_400


def _train(X, timestamps=None, asof=None):
    X = sp.csr_matrix(X, dtype=np.float32)
    coo = X.tocoo()
    ts = np.zeros(coo.nnz, dtype=np.int64) if timestamps is None else np.asarray(timestamps)
    asof = np.full(X.shape[0], 10**12) if asof is None else np.asarray(asof)
    return TrainData(X=X, user=coo.row.astype(np.int64), item=coo.col.astype(np.int64),
                     timestamp=ts, asof=asof)


def _block_data(rng, n_users=60, n_items=45, groups=3, density=0.6, hide=0.25):
    """Users like only items of their own group; a share of those likes is hidden."""
    ug, ig = np.arange(n_users) % groups, np.arange(n_items) % groups
    same = ug[:, None] == ig[None, :]
    liked = same & (rng.random((n_users, n_items)) < density)
    hidden = liked & (rng.random(liked.shape) < hide)
    return liked & ~hidden, hidden, same


def _auc_hidden_vs_other_groups(scores, train, hidden, same):
    """Share of (hidden in-group item, out-of-group item) pairs ranked correctly."""
    aucs = []
    for u in range(scores.shape[0]):
        pos, neg = scores[u, hidden[u]], scores[u, ~same[u] & ~train[u]]
        if pos.size and neg.size:
            aucs.append(np.mean(pos[:, None] > neg[None, :]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------- EASE
def test_ease_matches_columnwise_ridge_regressions(rng):
    X = (rng.random((30, 12)) < 0.3).astype(float)
    lam = 2.5
    B = ease_weights(sp.csr_matrix(X), lam)
    for j in range(12):  # independent reference: ridge of column j on all other columns
        others = np.delete(np.arange(12), j)
        A = X[:, others]
        b = np.linalg.solve(A.T @ A + lam * np.eye(11), A.T @ X[:, j])
        np.testing.assert_allclose(B[others, j], b, atol=1e-10)
        assert B[j, j] == 0.0


# ---------------------------------------------------------------- item-kNN
def test_itemknn_toy_similarities_and_scores():
    X = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1],
                  [1, 0, 0, 1]], dtype=float)
    # co-counts: c01=2, c02=1, c03=1, c12=2, c13=1, c23=1; item norms sqrt(3, 3, 2, 2)
    S = shrunk_cosine(sp.csr_matrix(X), shrink=1.0)
    assert S[0, 1] == pytest.approx(2 / (3 + 1))
    assert S[1, 2] == pytest.approx(2 / (np.sqrt(6) + 1))
    assert S[2, 3] == pytest.approx(1 / (2 + 1))
    assert np.allclose(np.diag(S), 0) and np.allclose(S, S.T)
    W = keep_top_k(S, k=1)  # each target item (column) keeps its single nearest neighbour
    assert np.all((W > 0).sum(axis=0) == 1)
    # nearest neighbours: item 0 -> 1 (0.5), 1 -> 2 (0.58 > 0.5), 2 -> 1, 3 -> 2 (1/3)
    assert np.array_equal(np.argmax(W, axis=0), [1, 2, 1, 2])
    assert W[1, 0] == S[1, 0] and W[0, 1] == 0.0
    model = ItemKNN(k=1, shrink=1.0).fit(_train(X))
    np.testing.assert_allclose(model.score(np.arange(4)), X @ W)


# ---------------------------------------------------------------- iALS
def _dense_objective(R, U, V, reg, alpha):
    P = (R > 0).astype(float)
    C = 1 + alpha * R
    return np.sum(C * (P - U @ V.T) ** 2) + reg * (np.sum(U**2) + np.sum(V**2))


def test_als_objective_and_exact_solve_match_dense_reference(rng):
    R = (rng.random((15, 10)) < 0.3).astype(float)
    U, V = rng.normal(size=(15, 4)), rng.normal(size=(10, 4))
    reg, alpha = 0.7, 3.0
    Rs = sp.csr_matrix(R)
    assert implicit_objective(Rs, U, V, reg, alpha) == pytest.approx(
        _dense_objective(R, U, V, reg, alpha))
    X = solve_exact(Rs, V, reg, alpha)
    for u in range(15):  # weighted ridge regression with the full diagonal C_u
        C = np.diag(1 + alpha * R[u])
        ref = np.linalg.solve(V.T @ C @ V + reg * np.eye(4), V.T @ C @ (R[u] > 0))
        np.testing.assert_allclose(X[u], ref, atol=1e-10)
    # conjugate gradient converges to the same solution (f = 4 steps are exact in theory)
    np.testing.assert_allclose(solve_cg(Rs, V, np.zeros((15, 4)), reg, alpha, steps=20), X,
                               atol=1e-8)


@pytest.mark.parametrize("solver", ["exact", "cg"])
def test_als_objective_decreases_and_recovers_planted_blocks(rng, solver):
    train, hidden, same = _block_data(rng)
    model = ImplicitALS(factors=3, reg=0.1, alpha=5.0, iterations=12, solver=solver)
    model.fit(_train(train))
    loss = np.array(model.loss_history_)
    assert np.all(np.diff(loss) <= 1e-9 * loss[0])  # monotone (ALS / CG never go uphill)
    assert loss[-1] < 0.6 * loss[0]
    auc = _auc_hidden_vs_other_groups(model.score(np.arange(train.shape[0])), train, hidden,
                                      same)
    assert auc > 0.99


def test_als_recovers_exact_low_rank_matrix(rng):
    # With alpha = 0 every pair has weight 1, so ALS fits the binary matrix by least squares;
    # a rank-2 block matrix is reproduced almost exactly with rank-2 factors.
    M = np.kron(np.eye(2), np.ones((5, 4)))
    model = ImplicitALS(factors=2, reg=1e-6, alpha=0.0, iterations=30, solver="exact").fit(
        _train(M))
    np.testing.assert_allclose(model.score(np.arange(10)), M, atol=1e-3)


# ---------------------------------------------------------------- BPR
def test_bpr_learns_blocks_and_loss_decreases(rng):
    train, hidden, same = _block_data(rng)
    model = BPR(factors=8, lr=0.05, reg=0.001, epochs=40, batch_size=64, seed=1)
    model.fit(_train(train))
    loss = model.loss_history_
    assert loss[-1] < 0.5 * loss[0]
    auc = _auc_hidden_vs_other_groups(model.score(np.arange(train.shape[0])), train, hidden,
                                      same)
    assert auc > 0.95
    again = BPR(factors=8, lr=0.05, reg=0.001, epochs=40, batch_size=64, seed=1)
    np.testing.assert_array_equal(again.fit(_train(train)).score(np.arange(3)),
                                  model.score(np.arange(3)))


# ---------------------------------------------------------------- popularity
def test_most_popular_counts():
    X = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
    scores = MostPopular().fit(_train(X)).score(np.array([0, 2]))
    np.testing.assert_array_equal(scores, [[3, 1, 1], [3, 1, 1]])


def test_recent_popular_never_looks_past_asof():
    # item 0 liked on day 1, item 1 on days 8 and 9, item 2 on day 20 (after user 0's asof)
    X = np.zeros((4, 3))
    X[[1, 2, 3, 0], [0, 1, 1, 2]] = 1
    coo = sp.csr_matrix(X).tocoo()
    day = {(1, 0): 1, (2, 1): 8, (3, 1): 9, (0, 2): 20}
    ts = np.array([day[(r, c)] * DAY for r, c in zip(coo.row, coo.col, strict=True)])
    asof = np.array([10, 30, 30, 30]) * DAY
    train = _train(X, ts, asof)
    s = RecentPopular(window_days=5).fit(train).score(np.array([0, 1]))
    assert s[0, 2] == 0  # the day-20 event is in user 0's future
    assert s[0, 1] > s[0, 0]  # days 8-9 are inside the 5-day window, day 1 is not
    assert 0 < s[0, 0] < 1  # outside the window: only the all-history tie-break remains
    s15 = RecentPopular(window_days=15).fit(train).score(np.array([1]))
    assert s15[0, 2] >= 1 and s15[0, 1] < 1  # user 1 (day 30): only day 20 is in [15, 30)
    s_all = RecentPopular(window_days=None).fit(train).score(np.array([0]))
    assert s_all[0, 1] > s_all[0, 0] > s_all[0, 2] == 0


def test_ease_scores_are_history_times_weights(rng):
    X = (rng.random((20, 9)) < 0.4).astype(float)
    model = EASE(lam=3.0).fit(_train(X))
    np.testing.assert_allclose(model.score(np.arange(20)), X @ model.W_, atol=1e-12)
