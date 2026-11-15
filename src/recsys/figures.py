"""Report figures (static PNGs for the README), drawn from metrics.json only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

FAMILY = {"MostPopular": "Popularity", "RecentPopular": "Popularity",
          "ItemKNN": "Item-item", "EASE": "Item-item",
          "iALS": "Latent factor", "BPR": "Latent factor"}
FAMILY_COLOR = {"Popularity": BLUE, "Item-item": ORANGE, "Latent factor": AQUA}
# Second model of each family: dashed line, square marker.
SECOND = {"RecentPopular", "EASE", "BPR"}

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _color(model: str) -> str:
    return FAMILY_COLOR[FAMILY[model]]


def _family_legend(ax: plt.Axes, loc: str = "lower right", **kw: Any) -> None:
    handles = [Line2D([], [], marker="o", ls="none", color=c, label=f, markersize=8)
               for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc=loc, **kw)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(m: dict[str, Any], path: Path) -> None:
    """Hero: test NDCG@10 with bootstrap CIs, and how much a random split inflates it."""
    test, k = m["test"], m["setup"]["k"]
    order = sorted(test, key=lambda n: test[n]["ndcg"])
    y = np.arange(len(order))
    n_users = m["protocols"]["per_user_temporal"]["summary"]["test_users"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True,
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    for yi, name in zip(y, order, strict=True):
        r = test[name]
        lo, hi = r["ndcg_ci"]
        ax.plot([lo, hi], [yi, yi], color=_color(name), lw=2.5, solid_capstyle="round")
        ax.plot(r["ndcg"], yi, "o", color=_color(name), markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(hi + 0.004, yi, f"{r['ndcg']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order)
    ax.set_xlim(0, max(test[n]["ndcg_ci"][1] for n in order) * 1.18)
    ax.set_xlabel(f"NDCG@{k} on the per-user temporal test set")
    ax.set_title(f"Test NDCG@{k} with 95% bootstrap CI ({n_users} users)")
    ax.grid(axis="y", visible=False)
    _family_legend(ax)

    rnd = m["protocols"]["random"]["ndcg"]
    for yi, name in zip(y, order, strict=True):
        t_val, r_val = test[name]["ndcg"], rnd[name]
        ax2.plot([t_val, r_val], [yi, yi], color=_color(name), lw=2, alpha=0.45)
        ax2.plot(t_val, yi, "o", color=_color(name), markersize=9,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax2.plot(r_val, yi, "o", markerfacecolor=SURFACE, color=_color(name), markersize=9,
                 markeredgewidth=2)
        ax2.text(r_val + 0.007, yi, f"{r_val / t_val:.1f}x", va="center", fontsize=9,
                 color=INK)
    ax2.set_xlim(0, max(rnd.values()) * 1.15)
    ax2.set_xlabel(f"NDCG@{k}")
    ratios = [rnd[n] / test[n]["ndcg"] for n in order if FAMILY[n] != "Popularity"]
    ax2.set_title(f"A random split inflates personalised models {min(ratios):.1f}-"
                  f"{max(ratios):.1f}x")
    ax2.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], marker="o", ls="none", color=INK2, markersize=8,
               label="Per-user temporal split (last ratings held out)"),
        Line2D([], [], marker="o", ls="none", color=INK2, markerfacecolor=SURFACE,
               markeredgewidth=2, markersize=8, label="Random split, same hold-out sizes"),
    ]
    ax2.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def fig_accuracy_vs_popularity(m: dict[str, Any], path: Path) -> None:
    """Accuracy against popularity bias and catalogue coverage."""
    test, k = m["test"], m["setup"]["k"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    personal = [n for n in test if FAMILY[n] != "Popularity"]
    ratios = [test[n]["coverage"] / test["MostPopular"]["coverage"] for n in personal]
    panels = (
        ("avg_popularity", "Average popularity of recommended movies\n"
         "(share of users who liked them in training)",
         "Personalised models: higher NDCG with less popular picks"),
        ("coverage", f"Catalogue coverage (share of movies in any top-{k} list)",
         f"Personalised models cover {min(ratios):.1f}-{max(ratios):.1f}x more of the "
         "catalogue"),
    )
    # Label offsets in points; negative x puts the label on the left of its marker.
    offsets = {"MostPopular": (7, 7), "RecentPopular": (7, -12), "ItemKNN": (8, -7),
               "EASE": (7, 7), "iALS": (-9, -6), "BPR": (8, -7)}
    for ax, (key, xlabel, title) in zip(axes, panels, strict=True):
        for name, r in test.items():
            ax.plot(r[key], r["ndcg"], "o", color=_color(name), markersize=10,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            dx, dy = offsets.get(name, (7, 0))
            ax.annotate(name, (r[key], r["ndcg"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9, color=INK, va="center", ha="right" if dx < 0 else "left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        lo = min(r[key] for r in test.values())
        hi = max(r[key] for r in test.values())
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
    axes[0].set_ylabel(f"NDCG@{k} (test)")
    axes[0].set_ylim(0, max(r["ndcg"] for r in test.values()) * 1.15)
    _family_legend(axes[1])
    fig.tight_layout()
    _save(fig, path)


def fig_activity(m: dict[str, Any], path: Path) -> None:
    """NDCG@10 by how many liked movies the user has in their history."""
    rows, k = m["activity_buckets"], m["setup"]["k"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for name in m["test"]:
        ax.plot(x, [r["ndcg"][name] for r in rows], color=_color(name),
                ls="--" if name in SECOND else "-", marker="s" if name in SECOND else "o",
                markeredgecolor=SURFACE, label=name)
    ax.set_xticks(x, [f"{r['bucket']} liked\n({r['n_users']} users)" for r in rows])
    ax.set_xlabel("Liked movies in the user's history (fit on train + validation)")
    ax.set_ylabel(f"NDCG@{k} (test)")
    ax.set_ylim(0, max(max(r["ndcg"].values()) for r in rows) * 1.15)
    personal = [n for n in m["test"] if FAMILY[n] != "Popularity"]
    wins = all(min(r["ndcg"][n] for n in personal) > max(r["ndcg"]["MostPopular"],
               r["ndcg"]["RecentPopular"]) for r in rows)
    ax.set_title("Every personalised model beats popularity in every bucket" if wins
                 else f"NDCG@{k} by user activity")
    ax.legend(loc="upper right", ncol=2, handlelength=3.2)
    fig.tight_layout()
    _save(fig, path)


def fig_training(m: dict[str, Any], path: Path) -> None:
    """Training objective and validation NDCG per epoch for iALS and BPR."""
    k = m["setup"]["k"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for col, (name, unit, loss_label) in enumerate((
        ("iALS", "ALS iteration", "Weighted squared loss (full objective)"),
        ("BPR", "Epoch", "Mean BPR loss  -ln sigmoid(x_ui - x_uj)"),
    )):
        tune = m["tuning"][name]
        curve = tune["best_curve"]
        params = tune["best_params"]
        loss = np.asarray(curve["train_loss"], dtype=float)
        top, bottom = axes[0, col], axes[1, col]
        if name == "iALS":  # loss_history_ starts with the objective at initialisation
            top.plot(np.arange(1, loss.size), loss[1:], "o-", color=_color(name),
                     markeredgecolor=SURFACE, markersize=5)
        else:
            top.plot(np.arange(1, loss.size + 1), loss, color=_color(name))
        top.set_ylabel(loss_label, fontsize=9)
        shown = ", ".join(f"{p}={v}" for p, v in params.items() if p not in
                          ("iterations", "epochs"))
        top.set_title(f"{name} training loss", fontsize=10.5)
        top.text(1.0, 1.02, shown, transform=top.transAxes, ha="right", va="bottom",
                 fontsize=8.5, color=MUTED)
        ep, val = curve["epoch"], curve["valid_ndcg"]
        bottom.plot(ep, val, "o-", color=_color(name), markeredgecolor=SURFACE, markersize=5)
        best_i = int(np.argmax(val))
        bottom.annotate(f"selected: {ep[best_i]}", (ep[best_i], val[best_i]),
                        xytext=(0, -26), textcoords="offset points", ha="center",
                        fontsize=9, color=INK,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        bottom.set_xlabel(unit)
        bottom.set_ylabel(f"Validation NDCG@{k}")
        bottom.set_title(f"{name} validation NDCG@{k}", fontsize=10.5)
        lo, hi = min(val), max(val)
        bottom.set_ylim(lo - 0.35 * (hi - lo) - 1e-3, hi + 0.15 * (hi - lo) + 1e-3)
    fig.tight_layout()
    _save(fig, path)


def make_figures(metrics: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write all report figures to ``out_dir``; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "model_comparison.png": fig_model_comparison,
        "accuracy_vs_popularity.png": fig_accuracy_vs_popularity,
        "ndcg_by_activity.png": fig_activity,
        "training_curves.png": fig_training,
    }
    paths = []
    with plt.rc_context(STYLE):
        for fname, fn in jobs.items():
            fn(metrics, out_dir / fname)
            paths.append(out_dir / fname)
    return paths
