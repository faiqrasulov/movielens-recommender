"""The full experiment: tune on validation, report once on test, then stress-test the protocol.

Writes ``metrics.json`` (every number quoted in the README) and the figures.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from recsys.data import Interactions
from recsys.evaluate import (
    Evaluation,
    beyond_accuracy,
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
)
from recsys.explain import explain_recommendations
from recsys.metrics import RANKING_METRICS
from recsys.models.base import Recommender
from recsys.split import (
    POSITIVE_THRESHOLD,
    Split,
    Task,
    final_task,
    global_time_split,
    per_user_temporal_split,
    random_split_like,
    tuning_task,
)
from recsys.tuning import EPOCH_PARAM, MODELS, SEARCH_SPACE, grid_search

MODEL_ORDER = ["MostPopular", "RecentPopular", "ItemKNN", "iALS", "BPR", "EASE"]
# Users grouped by the number of positives in their fitting history.
ACTIVITY_BUCKETS: list[tuple[int, int | None]] = [(1, 10), (11, 30), (31, 100), (101, None)]


@dataclass(frozen=True)
class Config:
    """Experiment settings; ``search_space`` overrides the default grids (used in tests)."""

    k: int = 10
    n_holdout: int = 5
    n_boot: int = 2000
    seed: int = 0
    search_space: dict[str, dict[str, list[Any]]] | None = None
    figures: bool = True


def _log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def _clean(obj: Any) -> Any:
    """JSON-ready copy: numpy scalars to Python, floats rounded to 6 significant digits."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(f"{float(obj):.6g}")
    return obj


def _date(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def dataset_summary(data: Interactions) -> dict[str, Any]:
    """Basic counts of the interaction data."""
    pos = data.rating >= POSITIVE_THRESHOLD
    _, per_user_time = np.unique(np.stack([data.user, data.timestamp]), axis=1,
                                 return_counts=True)
    first = np.full(data.n_users, np.iinfo(np.int64).max)
    last = np.zeros(data.n_users, dtype=np.int64)
    np.minimum.at(first, data.user, data.timestamp)
    np.maximum.at(last, data.user, data.timestamp)
    return {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_ratings": len(data),
        "positive_share": float(pos.mean()),
        "n_positives": int(pos.sum()),
        "density": len(data) / (data.n_users * data.n_items),
        "first_date": _date(data.timestamp.min()),
        "last_date": _date(data.timestamp.max()),
        "share_users_with_all_ratings_within_one_day": float(np.mean(last - first < 86_400)),
        # MovieLens stored ratings in batches: share of ratings that share their user's
        # timestamp with at least one other rating.
        "share_ratings_in_same_second_batch": float(
            per_user_time[per_user_time > 1].sum() / len(data)),
    }


def _split_summary(data: Interactions, split: Split, valid: Task, test: Task) -> dict[str, Any]:
    held_out = np.bincount(data.user[split.test], minlength=data.n_users)
    out: dict[str, Any] = {
        "n_train": int(split.train.size),
        "n_valid": int(split.valid.size),
        "n_test": int(split.test.size),
        "valid_users": int(valid.users.size),
        "test_users": int(test.users.size),
        "test_positives_of_evaluated_users": int(test.truth[test.users].nnz),
        "test_ratings_per_user_min_median_max": [
            int(held_out.min()), float(np.median(held_out)), int(held_out.max())],
    }
    if split.name == "global_temporal":
        out["valid_cutoff"] = _date(split.valid_asof[0])
        out["test_cutoff"] = _date(split.test_asof[0])
    return out


def _fit(name: str, params: dict[str, Any], task: Task) -> Recommender:
    return MODELS[name](**params).fit(task.train)


def _summary(ev: Evaluation, task: Task, cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {m: ev.mean(m) for m in RANKING_METRICS}
    out["ndcg_ci"] = bootstrap_ci(ev.per_user["ndcg"], cfg.n_boot, seed=cfg.seed)
    out.update(beyond_accuracy(ev, task))
    return out


def _activity(evals: dict[str, Evaluation], task: Task, cfg: Config) -> list[dict[str, Any]]:
    history = np.diff(task.train.X.indptr)[task.users]
    rows = []
    for lo, hi in ACTIVITY_BUCKETS:
        mask = (history >= lo) & ((history <= hi) if hi is not None else True)
        row: dict[str, Any] = {"bucket": f"{lo}-{hi}" if hi is not None else f"{lo}+",
                               "n_users": int(mask.sum()), "ndcg": {}, "ndcg_ci": {}}
        for name, ev in evals.items():
            values = ev.per_user["ndcg"][mask]
            row["ndcg"][name] = float(values.mean()) if values.size else float("nan")
            if values.size > 1:
                row["ndcg_ci"][name] = bootstrap_ci(values, cfg.n_boot, seed=cfg.seed)
        rows.append(row)
    return rows


def _cold_start(data: Interactions, split: Split, best: dict[str, dict[str, Any]],
                cfg: Config) -> dict[str, Any]:
    """Under the global time split: how many active users are brand new, and how well can
    the popularity baselines (the only option without history) serve them?"""
    task = final_task(data, split)
    active = np.flatnonzero(np.diff(task.truth.indptr) > 0)
    no_history = active[np.diff(task.seen.indptr)[active] == 0]
    out: dict[str, Any] = {
        "active_test_users": int(active.size),
        "new_users_without_history": int(no_history.size),
        "warm_users_evaluated": int(task.users.size),
        "ndcg_new_users": {},
    }
    if no_history.size:
        cold = Task(train=task.train, seen=task.seen, truth=task.truth, users=no_history)
        for name in ("MostPopular", "RecentPopular"):
            ev = evaluate(_fit(name, best[name], cold), cold, cfg.k)
            out["ndcg_new_users"][name] = ev.mean("ndcg")
    return out


def _explanation(data: Interactions, model: Recommender, ev: Evaluation,
                 task: Task) -> dict[str, Any]:
    """'Because you liked' for the first evaluated user with 20-40 liked training movies."""
    history = np.diff(task.train.X.indptr)[ev.users]
    candidates = np.flatnonzero((history >= 20) & (history <= 40))
    row = int(candidates[0]) if candidates.size else 0
    user = int(ev.users[row])
    recs = ev.recs[row][:5]
    mine = task.train.user == user
    recent = task.train.item[mine][np.argsort(-task.train.timestamp[mine], kind="stable")][:5]
    explained = explain_recommendations(model, user, recs, data.titles)  # type: ignore[arg-type]
    for entry in explained:
        entry["hit"] = bool(task.truth[user, entry["item"]] > 0)
    held_out = task.truth[user].indices
    return {
        "rule": "first evaluated user (lowest id) with 20-40 liked movies in the fitting data",
        "user_id": user + 1,  # MovieLens ids are 1-based
        "n_liked_in_history": int(history[row]),
        "recently_liked": [str(data.titles[i]) for i in recent],
        "held_out_liked": [str(data.titles[i]) for i in held_out],
        "hits_in_top_k": int(np.isin(ev.recs[row], held_out).sum()),
        "top_k": [str(data.titles[i]) for i in ev.recs[row]],
        "recommendations": explained,
    }


def run(data: Interactions, out_dir: str | Path, cfg: Config = Config(),
        dataset: str = "MovieLens 100K") -> dict[str, Any]:
    """Run the whole study on ``data`` and write ``metrics.json`` (+ figures) to ``out_dir``."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    space = SEARCH_SPACE if cfg.search_space is None else cfg.search_space

    main = per_user_temporal_split(data, cfg.n_holdout)
    valid, test = tuning_task(data, main), final_task(data, main)
    _log(f"per-user temporal split: {valid.users.size} validation / {test.users.size} test "
         "users", t0)

    # 1. Hyperparameter search on validation only.
    best: dict[str, dict[str, Any]] = {}
    tuning: dict[str, Any] = {}
    for name in MODEL_ORDER:
        params, trials = grid_search(name, valid, space[name], cfg.k)
        best[name] = params
        top = max(trials, key=lambda t: t.ndcg)
        tuning[name] = {
            "best_params": params,
            "best_valid_ndcg": top.ndcg,
            "n_configs": len(trials),
            "trials": [{"params": t.params, "valid_ndcg": t.ndcg} for t in trials],
        }
        if name in EPOCH_PARAM:
            tuning[name]["best_curve"] = {"epoch": [c[0] for c in top.curve],
                                          "valid_ndcg": [c[1] for c in top.curve],
                                          "train_loss": top.loss}
        _log(f"tuned {name:13s} {len(trials):3d} configs, valid NDCG@{cfg.k} "
             f"{top.ndcg:.4f}  {params}", t0)

    # 2. Final evaluation on test (refit on train + validation).
    evals: dict[str, Evaluation] = {}
    fitted: dict[str, Recommender] = {}
    results: dict[str, Any] = {}
    for name in MODEL_ORDER:
        fitted[name] = _fit(name, best[name], test)
        evals[name] = evaluate(fitted[name], test, cfg.k)
        results[name] = _summary(evals[name], test, cfg)
        _log(f"test {name:13s} NDCG@{cfg.k} {results[name]['ndcg']:.4f}", t0)
    best_model = max(MODEL_ORDER, key=lambda n: results[n]["ndcg"])
    runner_up = max((n for n in MODEL_ORDER if n != best_model),
                    key=lambda n: results[n]["ndcg"])
    paired = {
        f"{name}_vs_MostPopular": paired_bootstrap(
            evals[name].per_user["ndcg"], evals["MostPopular"].per_user["ndcg"],
            cfg.n_boot, seed=cfg.seed)
        for name in MODEL_ORDER if name != "MostPopular"
    }
    paired[f"{best_model}_vs_{runner_up}"] = paired_bootstrap(
        evals[best_model].per_user["ndcg"], evals[runner_up].per_user["ndcg"],
        cfg.n_boot, seed=cfg.seed)

    # 3. The same tuned models under other evaluation protocols.
    splits = {"random": random_split_like(data, main, seed=cfg.seed),
              "per_user_temporal": main,
              "global_temporal": global_time_split(data)}
    protocols: dict[str, Any] = {}
    for pname, split in splits.items():
        task = test if split is main else final_task(data, split)
        entry: dict[str, Any] = {"summary": _split_summary(data, split, tuning_task(data, split),
                                                             task),
                                 "ndcg": {}, "ndcg_ci": {}, "recall": {}}
        for name in MODEL_ORDER:
            ev = evals[name] if split is main else evaluate(_fit(name, best[name], task),
                                                            task, cfg.k)
            entry["ndcg"][name] = ev.mean("ndcg")
            entry["recall"][name] = ev.mean("recall")
            entry["ndcg_ci"][name] = bootstrap_ci(ev.per_user["ndcg"], cfg.n_boot,
                                                  seed=cfg.seed)
        protocols[pname] = entry
        _log(f"protocol {pname:18s} {entry['summary']['test_users']} users", t0)

    metrics: dict[str, Any] = {
        "dataset": {"name": dataset, **dataset_summary(data)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "setup": {"k": cfg.k, "positive_threshold": POSITIVE_THRESHOLD,
                  "n_holdout_per_user": cfg.n_holdout, "n_bootstrap": cfg.n_boot,
                  "seed": cfg.seed, "main_protocol": "per_user_temporal"},
        "tuning": tuning,
        "test": results,
        "best_model": best_model,
        "paired_ndcg_differences": paired,
        "activity_buckets": _activity(evals, test, cfg),
        "protocols": protocols,
        "cold_start_global_split": _cold_start(data, splits["global_temporal"], best, cfg),
        "explanation_example": _explanation(data, fitted["EASE"], evals["EASE"], test),
    }
    metrics = _clean(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if cfg.figures:
        from recsys.figures import make_figures

        make_figures(metrics, out_dir / "figures")
    _log(f"wrote {out_dir / 'metrics.json'}", t0)
    return metrics
