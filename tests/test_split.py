import numpy as np

from recsys.split import (
    POSITIVE_THRESHOLD,
    final_task,
    global_time_split,
    per_user_temporal_split,
    random_split_like,
    tuning_task,
)


def _assert_partition(split, n):
    parts = np.concatenate([split.train, split.valid, split.test])
    assert parts.size == n
    assert np.array_equal(np.sort(parts), np.arange(n))


def test_toy_data_has_timestamp_ties(toy_data):
    pairs = np.stack([toy_data.user, toy_data.timestamp])
    assert np.unique(pairs, axis=1).shape[1] < len(toy_data)


def test_per_user_split_is_strictly_temporal(toy_data):
    s = per_user_temporal_split(toy_data, n_holdout=5)
    _assert_partition(s, len(toy_data))
    t, u = toy_data.timestamp, toy_data.user
    for user in range(toy_data.n_users):
        tr = t[s.train][u[s.train] == user]
        va = t[s.valid][u[s.valid] == user]
        te = t[s.test][u[s.test] == user]
        assert te.size >= 5 and va.size >= 5  # ties may only enlarge the held-out sets
        # Every training rating is strictly earlier than every validation rating, which is
        # strictly earlier than every test rating (no shared timestamps across the cut).
        assert tr.max() < va.min()
        assert va.max() < te.min()
        assert s.test_asof[user] == te.min()
        assert s.valid_asof[user] == va.min()


def test_global_split_respects_cutoffs(toy_data):
    s = global_time_split(toy_data, 0.7, 0.85)
    _assert_partition(s, len(toy_data))
    t = toy_data.timestamp
    t_valid, t_test = s.valid_asof[0], s.test_asof[0]
    assert t[s.train].max() < t_valid <= t[s.valid].min()
    assert t[s.valid].max() < t_test <= t[s.test].min()


def test_random_split_keeps_per_user_sizes(toy_data):
    ref = per_user_temporal_split(toy_data)
    s = random_split_like(toy_data, ref, seed=3)
    _assert_partition(s, len(toy_data))
    for part in ("valid", "test"):
        a = np.bincount(toy_data.user[getattr(ref, part)], minlength=toy_data.n_users)
        b = np.bincount(toy_data.user[getattr(s, part)], minlength=toy_data.n_users)
        assert np.array_equal(a, b)
    assert not np.array_equal(np.sort(s.test), np.sort(ref.test))


def test_tasks_use_positives_and_mask_all_seen(toy_data):
    s = per_user_temporal_split(toy_data)
    valid, test = tuning_task(toy_data, s), final_task(toy_data, s)
    pos = toy_data.rating >= POSITIVE_THRESHOLD
    # validation task fits on train only; final task on train + validation
    assert valid.train.X.nnz == np.unique(
        toy_data.user[s.train][pos[s.train]] * 1000 + toy_data.item[s.train][pos[s.train]]).size
    assert test.seen.nnz == s.train.size + s.valid.size
    assert valid.seen.nnz == s.train.size
    # truth holds held-out positives only, and no test positive is in the fitting data
    assert test.truth.nnz == pos[s.test].sum()
    assert test.truth.multiply(test.seen).nnz == 0
    # evaluated users have a held-out positive and a training positive
    assert np.all(np.diff(test.truth.indptr)[test.users] > 0)
    assert np.all(np.diff(test.train.X.indptr)[test.users] > 0)
    # time-aware models only see training events before each user's test time
    assert np.all(test.train.timestamp < s.test_asof[test.train.user])
