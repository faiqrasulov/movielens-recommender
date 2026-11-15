import json
import urllib.error

import numpy as np
import pytest

from recsys import data as data_module
from recsys.cli import main
from recsys.pipeline import Config, run

TINY_SPACE = {
    "MostPopular": {},
    "RecentPopular": {"window_days": [30, None]},
    "ItemKNN": {"k": [10, 40], "shrink": [0, 10]},
    "iALS": {"factors": [4], "reg": [1.0], "alpha": [0.0, 3.0], "iterations": [4]},
    "BPR": {"factors": [8], "lr": [0.05], "reg": [0.01], "epochs": [10]},
    "EASE": {"lam": [5.0, 50.0]},
}


def test_pipeline_smoke(toy_data, tmp_path):
    cfg = Config(n_boot=100, search_space=TINY_SPACE, figures=True)
    metrics = run(toy_data, tmp_path, cfg, dataset="synthetic")
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["best_model"] == metrics["best_model"]
    assert set(saved["test"]) == set(TINY_SPACE)
    for res in saved["test"].values():
        assert 0.0 <= res["ndcg_ci"][0] <= res["ndcg"] <= res["ndcg_ci"][1] <= 1.0
        assert 0.0 < res["coverage"] <= 1.0
    assert set(saved["protocols"]) == {"random", "per_user_temporal", "global_temporal"}
    assert saved["tuning"]["iALS"]["best_curve"]["epoch"] == [1, 2, 3, 4]
    assert len(saved["explanation_example"]["recommendations"]) == 5
    assert sum(b["n_users"] for b in saved["activity_buckets"]) == \
        saved["protocols"]["per_user_temporal"]["summary"]["test_users"]
    pngs = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert pngs == ["accuracy_vs_popularity.png", "model_comparison.png",
                    "ndcg_by_activity.png", "training_curves.png"]


def test_cli_run_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr("recsys.pipeline.SEARCH_SPACE", TINY_SPACE)
    main(["run", "--synthetic", "--no-figures", "--out", str(tmp_path)])
    assert json.loads((tmp_path / "metrics.json").read_text())["dataset"]["n_users"] == 60


def test_download_fails_clearly_when_offline(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(data_module.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="needs internet access"):
        data_module.download_ml100k(tmp_path)
    assert not list(tmp_path.glob("*.zip*"))


def test_load_from_cached_files(tmp_path):
    root = tmp_path / "ml-100k"
    root.mkdir()
    (root / "u.data").write_text("1\t2\t5\t881250949\n2\t1\t3\t891717742\n1\t1\t4\t881250950\n")
    (root / "u.item").write_bytes(
        "1|Toy Story (1995)|01-Jan-1995||\n2|Am\xe9lie (2001)|01-Jan-2001||\n".encode("latin-1"))
    d = data_module.load_ml100k(tmp_path)
    assert (d.n_users, d.n_items, len(d)) == (2, 2, 3)
    np.testing.assert_array_equal(d.user, [0, 1, 0])
    np.testing.assert_array_equal(d.item, [1, 0, 0])
    assert d.titles[1] == "Am\xe9lie (2001)"
