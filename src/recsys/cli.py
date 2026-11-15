"""Command line: ``python -m recsys.cli run`` reproduces every number in the README;
``python -m recsys.cli recommend --user 42`` prints explained recommendations for one user."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recsys.data import load_ml100k
from recsys.evaluate import top_k
from recsys.explain import explain_recommendations
from recsys.models import EASE
from recsys.pipeline import Config, run
from recsys.split import make_task
from recsys.synthetic import synthetic_interactions


def _run(args: argparse.Namespace) -> None:
    if args.synthetic:
        data, name = synthetic_interactions(seed=args.seed), "synthetic (smoke test)"
    else:
        data, name = load_ml100k(args.data_dir), "MovieLens 100K"
    run(data, args.out, Config(seed=args.seed, figures=not args.no_figures), dataset=name)


def _recommend(args: argparse.Namespace) -> None:
    data = load_ml100k(args.data_dir)
    lam = args.lam
    metrics_path = Path(args.metrics)
    if lam is None and metrics_path.exists():
        lam = json.loads(metrics_path.read_text())["tuning"]["EASE"]["best_params"]["lam"]
    lam = 300.0 if lam is None else lam
    user = args.user - 1  # MovieLens ids are 1-based
    if not 0 <= user < data.n_users:
        raise SystemExit(f"--user must be between 1 and {data.n_users}")
    everything = np.arange(len(data))
    task = make_task(data, everything, everything[:0],
                     np.full(data.n_users, data.timestamp.max() + 1))
    model = EASE(lam).fit(task.train)
    recs = top_k(model.score(np.array([user])), args.k, task.seen[[user]])[0]
    print(f"User {args.user}: {task.train.X[user].nnz} liked movies, EASE (lambda = {lam:g})")
    for rank, entry in enumerate(explain_recommendations(model, user, recs, data.titles), 1):
        reasons = "; ".join(b["title"] for b in entry["because"])
        print(f"{rank:2d}. {entry['title']}\n    because you liked: {reasons}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recsys", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="full experiment -> metrics.json + figures")
    p_run.add_argument("--data-dir", default="data")
    p_run.add_argument("--out", default="reports")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--synthetic", action="store_true",
                       help="use small synthetic data instead of MovieLens (no download)")
    p_run.add_argument("--no-figures", action="store_true")
    p_run.set_defaults(func=_run)
    p_rec = sub.add_parser("recommend", help="explained top-k for one MovieLens user")
    p_rec.add_argument("--user", type=int, required=True, help="MovieLens user id (1-943)")
    p_rec.add_argument("--k", type=int, default=10)
    p_rec.add_argument("--lam", type=float, default=None,
                       help="EASE regularisation (default: tuned value in metrics.json)")
    p_rec.add_argument("--data-dir", default="data")
    p_rec.add_argument("--metrics", default="reports/metrics.json")
    p_rec.set_defaults(func=_recommend)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
