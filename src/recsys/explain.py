""""Because you liked X" explanations for linear item-item models (EASE, item-kNN).

For these models the score is ``s_uj = sum_{i in history(u)} W_ij``, so each history item's
contribution ``W_ij`` is an exact additive part of the score, not a post-hoc approximation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recsys.models import EASE, ItemKNN


def contributions(model: EASE | ItemKNN, user: int, item: int) -> tuple[np.ndarray, np.ndarray]:
    """History items of ``user`` and their additive contributions to the score of ``item``."""
    history = model.X_[user].indices
    return history, model.W_[history, item]


def explain_recommendations(
    model: EASE | ItemKNN,
    user: int,
    recs: np.ndarray,
    titles: np.ndarray,
    top: int = 2,
) -> list[dict[str, Any]]:
    """For each recommended item, the ``top`` history items that contribute most to its score."""
    out = []
    for item in recs:
        history, contrib = contributions(model, user, int(item))
        order = np.argsort(-contrib, kind="stable")[:top]
        out.append({
            "item": int(item),
            "title": str(titles[item]),
            "score": float(contrib.sum()),
            "because": [
                {"item": int(history[o]), "title": str(titles[history[o]]),
                 "contribution": float(contrib[o])}
                for o in order
            ],
        })
    return out
