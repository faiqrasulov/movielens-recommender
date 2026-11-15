"""Grid search on the validation set (never on test)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from recsys.evaluate import evaluate
from recsys.models import BPR, EASE, ImplicitALS, ItemKNN, MostPopular, RecentPopular
from recsys.models.base import Recommender
from recsys.split import Task

MODELS: dict[str, type[Recommender]] = {
    "MostPopular": MostPopular,
    "RecentPopular": RecentPopular,
    "ItemKNN": ItemKNN,
    "iALS": ImplicitALS,
    "BPR": BPR,
    "EASE": EASE,
}

# Iterative models: the number of epochs is tuned from the validation curve of a single run.
EPOCH_PARAM = {"iALS": "iterations", "BPR": "epochs"}

SEARCH_SPACE: dict[str, dict[str, list[Any]]] = {
    "MostPopular": {},
    "RecentPopular": {"window_days": [7, 14, 30, 60, 120, None]},
    "ItemKNN": {"k": [25, 50, 100, 200, 400], "shrink": [0, 10, 50, 100]},
    "iALS": {"factors": [32, 64, 128], "reg": [3, 10, 30], "alpha": [0, 0.3, 1, 10],
             "iterations": [15]},
    "BPR": {"factors": [64, 128], "lr": [0.05], "reg": [0.003, 0.01], "epochs": [120]},
    "EASE": {"lam": [30, 100, 200, 300, 500, 1000, 3000]},
}

# Validation checkpoints for iterative models (every n-th epoch).
CHECK_EVERY = {"iALS": 1, "BPR": 5}


@dataclass
class Trial:
    """One configuration: its parameters, best validation NDCG@k and, for iterative models,
    the validation curve and the training loss per epoch."""

    params: dict[str, Any]
    ndcg: float
    curve: list[tuple[int, float]] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)


def _grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in
            itertools.product(*(space[k] for k in keys))]


def run_trial(name: str, params: dict[str, Any], task: Task, k: int = 10) -> Trial:
    """Fit one configuration on ``task.train`` and score it on ``task``'s held-out users.

    For iterative models, NDCG is recorded every ``CHECK_EVERY`` epochs and the best epoch
    is written back into the returned parameters.
    """
    model = MODELS[name](**params)
    if name not in EPOCH_PARAM:
        model.fit(task.train)
        return Trial(dict(params), evaluate(model, task, k).mean("ndcg"))
    curve: list[tuple[int, float]] = []
    every = CHECK_EVERY.get(name, 1)

    def checkpoint(epoch: int, m: Recommender) -> None:
        if epoch % every == 0:
            curve.append((epoch, evaluate(m, task, k).mean("ndcg")))

    model.fit(task.train, callback=checkpoint)  # type: ignore[call-arg]
    best_epoch, best = max(curve, key=lambda c: (c[1], -c[0]))
    tuned = dict(params)
    tuned[EPOCH_PARAM[name]] = best_epoch
    return Trial(tuned, best, curve, list(model.loss_history_))  # type: ignore[attr-defined]


def grid_search(
    name: str, task: Task, space: dict[str, list[Any]] | None = None, k: int = 10
) -> tuple[dict[str, Any], list[Trial]]:
    """Evaluate every configuration of ``space`` (default: ``SEARCH_SPACE[name]``) on the
    validation task; return the best parameters and all trials."""
    space = SEARCH_SPACE[name] if space is None else space
    trials = [run_trial(name, params, task, k) for params in _grid(space)]
    best = max(trials, key=lambda t: t.ndcg)
    return best.params, trials
