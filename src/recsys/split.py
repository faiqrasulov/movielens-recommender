"""Leakage-aware train / validation / test splits and the matrices built from them.

Three protocols, all with the same three-way structure (validation for tuning, test only for
the final report):

* :func:`per_user_temporal_split` (main protocol): each user's most recent ratings are held out.
  Every training rating of a user is *strictly earlier* than all of that user's validation
  ratings, which are strictly earlier than all of their test ratings.
* :func:`global_time_split`: two global cutoffs; the model never sees anything that happened
  after the moment it is evaluated. Strictest, but in MovieLens 100K most users rate everything
  in one session, so only a small number of users have both a history and a future.
* :func:`random_split_like`: the classic random hold-out, with exactly the same number of
  held-out ratings per user as a reference split, so the only difference is the time ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.data import Interactions

POSITIVE_THRESHOLD = 4  # rating >= 4 counts as a positive ("watched and liked")


@dataclass(frozen=True)
class Split:
    """Indices into an :class:`Interactions` object for each part of a split.

    ``valid_asof`` / ``test_asof`` hold, per user, the time at which that user's validation /
    test recommendations are made. Time-aware models may only use training events before it.
    """

    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    valid_asof: np.ndarray
    test_asof: np.ndarray


@dataclass(frozen=True)
class TrainData:
    """What a model is allowed to see when it is fitted.

    ``X`` is the binary user x item matrix of training positives; ``user``, ``item`` and
    ``timestamp`` list the same positive events (for time-aware baselines); ``asof[u]`` is the
    time at which user ``u`` receives recommendations.
    """

    X: sp.csr_matrix
    user: np.ndarray
    item: np.ndarray
    timestamp: np.ndarray
    asof: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.X.shape[1])


@dataclass(frozen=True)
class Task:
    """One evaluation round: fit on ``train``, recommend to ``users``, score against ``truth``.

    ``seen`` holds every rating (any value) in the fitting data; those items are never
    recommended again. ``truth`` holds the held-out positives.
    """

    train: TrainData
    seen: sp.csr_matrix
    truth: sp.csr_matrix
    users: np.ndarray


def _user_groups(data: Interactions) -> list[np.ndarray]:
    """Indices of each user's ratings sorted by time (ties broken by a fixed random key)."""
    tie_break = np.random.default_rng(0).permutation(len(data))
    order = np.lexsort((tie_break, data.timestamp, data.user))
    bounds = np.searchsorted(data.user[order], np.arange(data.n_users + 1))
    return [order[bounds[u] : bounds[u + 1]] for u in range(data.n_users)]


def _cut_last_n(times: np.ndarray, n: int) -> int:
    """Position where the last ``n`` ratings start, moved earlier so that no rating before the
    cut shares a timestamp with a rating after it (MovieLens stores ratings in batches)."""
    if times.size <= n:
        return 0
    tau = times[times.size - n]
    return int(np.searchsorted(times, tau, side="left"))


def per_user_temporal_split(data: Interactions, n_holdout: int = 5) -> Split:
    """Hold out each user's last ``n_holdout`` ratings for test and the ``n_holdout`` before
    those for validation (timestamp ties go to the held-out side, so the boundary is strict)."""
    train, valid, test = [], [], []
    valid_asof = np.zeros(data.n_users, dtype=np.int64)
    test_asof = np.zeros(data.n_users, dtype=np.int64)
    for u, g in enumerate(_user_groups(data)):
        if g.size == 0:
            continue
        t = data.timestamp[g]
        c_test = _cut_last_n(t, n_holdout)
        c_valid = _cut_last_n(t[:c_test], n_holdout)
        train.append(g[:c_valid])
        valid.append(g[c_valid:c_test])
        test.append(g[c_test:])
        test_asof[u] = t[c_test] if c_test < t.size else t[-1] + 1
        valid_asof[u] = t[c_valid] if c_valid < c_test else test_asof[u]
    return Split(
        name="per_user_temporal",
        train=np.sort(np.concatenate(train)),
        valid=np.sort(np.concatenate(valid)),
        test=np.sort(np.concatenate(test)),
        valid_asof=valid_asof,
        test_asof=test_asof,
    )


def global_time_split(
    data: Interactions, valid_quantile: float = 0.8, test_quantile: float = 0.9
) -> Split:
    """Split at two global time cutoffs: train < T_valid <= validation < T_test <= test."""
    t_valid = int(np.quantile(data.timestamp, valid_quantile, method="lower"))
    t_test = int(np.quantile(data.timestamp, test_quantile, method="lower"))
    if not t_valid < t_test:
        raise ValueError("validation cutoff must precede the test cutoff")
    ts = data.timestamp
    return Split(
        name="global_temporal",
        train=np.flatnonzero(ts < t_valid),
        valid=np.flatnonzero((ts >= t_valid) & (ts < t_test)),
        test=np.flatnonzero(ts >= t_test),
        valid_asof=np.full(data.n_users, t_valid, dtype=np.int64),
        test_asof=np.full(data.n_users, t_test, dtype=np.int64),
    )


def random_split_like(data: Interactions, reference: Split, seed: int = 0) -> Split:
    """Random hold-out with the same per-user validation / test sizes as ``reference``."""
    rng = np.random.default_rng(seed)
    n_valid = np.bincount(data.user[reference.valid], minlength=data.n_users)
    n_test = np.bincount(data.user[reference.test], minlength=data.n_users)
    train, valid, test = [], [], []
    for u, g in enumerate(_user_groups(data)):
        g = rng.permutation(g)
        test.append(g[: n_test[u]])
        valid.append(g[n_test[u] : n_test[u] + n_valid[u]])
        train.append(g[n_test[u] + n_valid[u] :])
    # No time ordering: recommendations are "made" after the last rating in the data.
    end = np.full(data.n_users, int(data.timestamp.max()) + 1, dtype=np.int64)
    return Split(
        name="random",
        train=np.sort(np.concatenate(train)),
        valid=np.sort(np.concatenate(valid)),
        test=np.sort(np.concatenate(test)),
        valid_asof=end,
        test_asof=end.copy(),
    )


def _matrix(data: Interactions, idx: np.ndarray) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(idx.size, dtype=np.float32), (data.user[idx], data.item[idx])),
        shape=(data.n_users, data.n_items),
    )
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def make_task(
    data: Interactions,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    asof: np.ndarray,
    threshold: int = POSITIVE_THRESHOLD,
) -> Task:
    """Build the matrices for fitting on ``fit_idx`` and evaluating on ``eval_idx``.

    Evaluated users are those with at least one held-out positive and at least one training
    positive (users without any history are the cold-start case, reported separately).
    """
    fit_pos = fit_idx[data.rating[fit_idx] >= threshold]
    eval_pos = eval_idx[data.rating[eval_idx] >= threshold]
    X = _matrix(data, fit_pos)
    truth = _matrix(data, eval_pos)
    users = np.flatnonzero((np.diff(truth.indptr) > 0) & (np.diff(X.indptr) > 0))
    train = TrainData(
        X=X,
        user=data.user[fit_pos],
        item=data.item[fit_pos],
        timestamp=data.timestamp[fit_pos],
        asof=np.asarray(asof, dtype=np.int64),
    )
    return Task(train=train, seen=_matrix(data, fit_idx), truth=truth, users=users)


def tuning_task(data: Interactions, split: Split) -> Task:
    """Fit on train, evaluate on validation (used for hyperparameter search)."""
    return make_task(data, split.train, split.valid, split.valid_asof)


def final_task(data: Interactions, split: Split) -> Task:
    """Fit on train + validation, evaluate on test (used once, for the final report)."""
    fit = np.sort(np.concatenate([split.train, split.valid]))
    return make_task(data, fit, split.test, split.test_asof)
