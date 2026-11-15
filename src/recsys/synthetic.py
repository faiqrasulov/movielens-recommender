"""Small synthetic interaction data with known structure (for tests and the smoke run)."""

from __future__ import annotations

import numpy as np

from recsys.data import Interactions


def synthetic_interactions(
    n_users: int = 60,
    n_items: int = 80,
    n_groups: int = 3,
    ratings_per_user: tuple[int, int] = (20, 40),
    days: int = 120,
    seed: int = 0,
) -> Interactions:
    """Users and items fall into ``n_groups`` taste groups; users mostly rate (and rate highly)
    items of their own group. Timestamps are spread over ``days`` days, with some ratings
    sharing a timestamp, as in MovieLens."""
    rng = np.random.default_rng(seed)
    user_group = np.arange(n_users) % n_groups
    item_group = np.arange(n_items) % n_groups
    users, items, ratings, times = [], [], [], []
    for u in range(n_users):
        n = int(rng.integers(ratings_per_user[0], ratings_per_user[1] + 1))
        own = item_group == user_group[u]
        p = np.where(own, 8.0, 1.0)
        chosen = rng.choice(n_items, size=n, replace=False, p=p / p.sum())
        liked = (own[chosen] & (rng.random(n) < 0.85)) | (~own[chosen] & (rng.random(n) < 0.2))
        start = rng.integers(0, days * 86_400 // 2)
        stamps = start + np.sort(rng.integers(0, days * 86_400 // 2, n))
        stamps[1::4] = stamps[0::4][: stamps[1::4].size]  # batched submissions -> ties
        users.append(np.full(n, u))
        items.append(chosen)
        ratings.append(np.where(liked, rng.integers(4, 6, n), rng.integers(1, 4, n)))
        times.append(np.sort(stamps) + 880_000_000)
    return Interactions(
        user=np.concatenate(users).astype(np.int64),
        item=np.concatenate(items).astype(np.int64),
        rating=np.concatenate(ratings).astype(np.int8),
        timestamp=np.concatenate(times).astype(np.int64),
        titles=np.array([f"Item {i} (group {item_group[i]})" for i in range(n_items)],
                        dtype=object),
        n_users=n_users,
        n_items=n_items,
    )
