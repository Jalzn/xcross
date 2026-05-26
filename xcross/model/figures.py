"""Render the metrics CSVs as polished tables/charts for visual inspection.

Reads artifacts/reports/metrics/*.csv (raw data) and writes artifacts/reports/figures/*.png.
Most figures are produced for both labels (success, shot); a few (importance, calibration,
lift) for each (feature_set, label). metrics/ holds raw numbers only; figures/ the visuals.
Each metric is documented in docs/metrics.md.

    uv run python -m xcross.model.figures
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.patches import Rectangle

from xcross.config import ROOT
from xcross.model.evaluate import lift_by_decile

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})

METRICS = ROOT / "artifacts" / "reports" / "metrics"
FIGURES = ROOT / "artifacts" / "reports" / "figures"
LABELS = ("success", "shot")
FEATURE_SETS = ("xcross", "xcrossot")
FEATURE_COLOR = {"xcross": "#1f77b4", "xcrossot": "#d62728"}
ESTIMATOR_MARKER = {
    "xgboost": "o", "adaboost": "s", "catboost": "^", "lightgbm": "D",
    "histgb": "v", "random_forest": "P", "logreg": "X", "tabpfn": "*",
}


def _fmt(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _save_table(headers: list[str], rows: list[list[str]], title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(min(2.2 + 1.6 * len(headers), 22), 0.5 * len(rows) + 1.4))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for c in range(len(headers)):
        table[0, c].set_facecolor("#34495e")
        table[0, c].set_text_props(color="white", fontweight="bold")
    ax.set_title(title, pad=14)
    fig.savefig(FIGURES / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- shared (cover both labels already) ---

def table_model_metrics() -> None:
    m = pl.read_csv(METRICS / "model_metrics.csv")
    models = m["model"].to_list()
    rows = [[c] + [_fmt(m.filter(pl.col("model") == mod)[c][0]) for mod in models]
            for c in m.columns if c != "model"]
    _save_table(["metric", *models], rows, "Final model metrics (all 4 targets)", "table_model_metrics.png")


def table_comparison() -> None:
    c = pl.read_csv(METRICS / "comparison.csv").sort(["label", "feature_set", "estimator", "calibration"])
    cols = [c2 for c2 in ["feature_set", "label", "estimator", "calibration", "auc", "auc_pr",
                          "brier_skill", "ece", "stability", "stability_temporal", "icc"] if c2 in c.columns]
    rows = [[_fmt(v) for v in r] for r in c.select(cols).iter_rows()]
    _save_table(cols, rows, "Comparison matrix (all runs)", "table_comparison.png")


# --- per label ---

def table_league_metrics(label: str) -> None:
    lg = pl.read_csv(METRICS / f"league_metrics_{label}.csv")
    cols = [c for c in ["league", "feature_set", "n", "auc", "brier_skill", "ece", "stability", "icc"]
            if c in lg.columns]
    rows = [[_fmt(v) for v in r] for r in lg.select(cols).iter_rows()]
    _save_table(cols, rows, f"Generalisation by league (xCross & xCrossOT) — {label}",
                f"table_league_metrics_{label}.png")


def chart_stability_vs_n(label: str) -> None:
    s = pl.read_csv(METRICS / f"stability_vs_n_{label}.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(range(len(s)), s["n_players"], color="#cfd8dc", label="number of players")
    ax.set_ylabel("number of qualified players")
    ax.set_xlabel("minimum crosses required per player")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s["min_crosses"])
    ax2 = ax.twinx()
    ax2.plot(range(len(s)), s["stability"], "o-", color="#d62728", lw=2, label="stability")
    ax2.set_ylabel("split-half stability (Spearman)")
    ax2.set_ylim(0, 1)
    ax.set_title(f"Ranking stability vs. min crosses — xCross/{label}")
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.16), fontsize=9)
    fig.savefig(FIGURES / f"chart_stability_vs_n_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_ranking_top(label: str, top: int = 20) -> None:
    r = pl.read_csv(METRICS / f"ranking_final_{label}.csv").head(top).reverse()
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * top + 1.2))
    err = 1.96 * r["xcross_se"] if "xcross_se" in r.columns else None
    ax.barh(r["nickname"], r["xcross"], xerr=err, color="#1f77b4", error_kw={"elinewidth": 0.8})
    ax.set_xlabel(f"mean xCross — quality of the cross situation created ({label}, 95% CI)")
    ax.set_title(f"Top {top} players — creation, xCross/{label}")
    ax.margins(y=0.01)
    fig.savefig(FIGURES / f"chart_ranking_top_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_quadrant(ax, ranking: pl.DataFrame, n_per_corner: int, fontsize: int, show_text: bool) -> None:
    """Scatter of players on xCross × xCrossOT, split into quadrants by the medians.
    Labels the standouts of each corner (best context, best outcome, best execution) with
    a white halo + leader line so the names stay readable over the dots and the shading."""
    ranking = ranking.with_columns(pl.col("nickname").fill_null(pl.col("player_id").cast(pl.Utf8)))
    x, yv = ranking["xcross"].to_numpy(), ranking["xcrossot"].to_numpy()
    mx, my = float(np.median(x)), float(np.median(yv))
    x0, x1 = x.min() - 0.02, x.max() + 0.02
    y0, y1 = yv.min() - 0.04, yv.max() + 0.04
    for (ox, oy), w, h, color, text in [
        ((mx, my), x1 - mx, y1 - my, "#fff3b0", "good creation\n+ high outcome"),
        ((x0, my), mx - x0, y1 - my, "#cdeac0", "weak creation\n+ high outcome"),
        ((x0, y0), mx - x0, my - y0, "#f4c7c3", "weak on both"),
        ((mx, y0), x1 - mx, my - y0, "#cfe2f3", "good creation\n+ low outcome"),
    ]:
        ax.add_patch(Rectangle((ox, oy), w, h, color=color, alpha=0.5, zorder=0))
        if show_text:
            ax.text(ox + w / 2, oy + h / 2, text, ha="center", va="center", fontsize=fontsize,
                    color="#666", fontweight="bold", alpha=0.6, zorder=1)
    ax.axvline(mx, color="gray", ls="--", lw=1, zorder=1)
    ax.axhline(my, color="gray", ls="--", lw=1, zorder=1)
    ax.scatter(x, yv, s=22, color="#555", alpha=0.45, linewidth=0, zorder=2)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    standouts = (ranking.sort("xcross", descending=True).head(n_per_corner)
                 .vstack(ranking.sort("xcrossot", descending=True).head(n_per_corner))
                 .vstack(ranking.sort("execution", descending=True).head(n_per_corner))
                 .unique("player_id"))
    ax.scatter(standouts["xcross"], standouts["xcrossot"], s=46, color="#1b2a4a",
               edgecolor="white", linewidth=0.8, zorder=4)
    rows = list(standouts.select("xcross", "xcrossot", "nickname").iter_rows(named=True))
    label_dx, min_gap = (x1 - x0) * 0.012, (y1 - y0) * 0.052
    for to_right in (True, False):  # declutter each side of the vertical median separately
        side = sorted((r for r in rows if (r["xcross"] >= mx) == to_right),
                      key=lambda r: r["xcrossot"], reverse=True)
        text_y = None
        for r in side:
            px, py = r["xcross"], r["xcrossot"]
            text_y = py if text_y is None else min(py, text_y - min_gap)
            tx, ha = (px + label_dx, "left") if to_right else (px - label_dx, "right")
            ax.annotate(r["nickname"], (px, py), xytext=(tx, text_y), textcoords="data",
                        fontsize=fontsize, ha=ha, va="center", zorder=5,
                        arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.6, "alpha": 0.7},
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])


def chart_ranking_quadrants(label: str) -> None:
    r = pl.read_csv(METRICS / f"ranking_final_{label}.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_quadrant(ax, r, n_per_corner=4, fontsize=9, show_text=True)
    ax.set_xlabel("mean xCross — quality of the situation created (context)")
    ax.set_ylabel("mean xCrossOT — outcome at the arrival point")
    ax.set_title(f"Players by quadrant: context × outcome — {label}")
    fig.savefig(FIGURES / f"chart_ranking_quadrants_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_ranking_quadrants_by_league(label: str) -> None:
    path = METRICS / f"ranking_by_league_{label}.csv"
    if not path.exists():
        return
    r = pl.read_csv(path)
    leagues = sorted(r["league"].unique().to_list())
    fig, axes = plt.subplots(1, len(leagues), figsize=(6.4 * len(leagues), 6.6), squeeze=False)
    for ax, lg in zip(axes[0], leagues, strict=True):
        sub = r.filter(pl.col("league") == lg)
        _plot_quadrant(ax, sub, n_per_corner=3, fontsize=8, show_text=False)
        ax.set_title(f"{lg} ({sub.height} players)")
        ax.set_xlabel("xCross (context)")
    axes[0][0].set_ylabel("xCrossOT (outcome)")
    fig.suptitle(f"Players by quadrant, per league — {label}", fontweight="bold", fontsize=14)
    fig.text(0.5, 0.93, "quadrants split by each league's own medians  ·  "
             "top-right = good context & outcome, bottom-left = weak on both",
             ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURES / f"chart_ranking_quadrants_by_league_{label}.png", dpi=120)
    plt.close(fig)


def chart_by_position(label: str, min_players: int = 4) -> None:
    r = pl.read_csv(METRICS / f"ranking_final_{label}.csv").filter(pl.col("position_group").is_not_null())
    counts = r.group_by("position_group").len().filter(pl.col("len") >= min_players)
    groups = (r.join(counts.select("position_group"), on="position_group")
              .group_by("position_group").agg(pl.col("xcross"), pl.col("xcross").median().alias("med")).sort("med"))
    if groups.height == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups["xcross"].to_list(), tick_labels=groups["position_group"].to_list())
    ax.set_xlabel("position group")
    ax.set_ylabel("mean xCross per player")
    ax.set_title(f"xCross by position — {label} (players with ≥20 crosses)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES / f"chart_by_position_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _grouped_bars(ax, categories: list[str], series: list[tuple[str, list[float]]]) -> None:
    """Draw one bar group per category, one bar per series. Shared by the comparison charts."""
    width = 0.8 / len(series)
    xs = range(len(categories))
    for i, (name, values) in enumerate(series):
        ax.bar([x + i * width for x in xs], values, width=width, label=name)
    ax.set_xticks([x + width * (len(series) - 1) / 2 for x in xs])
    ax.set_xticklabels(categories, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def chart_by_league(label: str) -> None:
    lg = pl.read_csv(METRICS / f"league_metrics_{label}.csv")
    feature_sets = lg["feature_set"].unique(maintain_order=True).to_list() if "feature_set" in lg.columns else [None]
    metric_cols = [c for c in ("auc", "brier_skill", "stability", "icc") if c in lg.columns]
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(6.0 * len(feature_sets), 5), squeeze=False, sharey=True)
    for ax, fs in zip(axes[0], feature_sets, strict=True):
        sub = lg.filter(pl.col("feature_set") == fs) if fs is not None else lg
        categories = [f"{sub['league'][j]}\n(n={sub['n'][j]})" for j in range(sub.height)]
        _grouped_bars(ax, categories, [(c, sub[c].to_list()) for c in metric_cols])
        ax.set_title(fs or "")
    axes[0][0].set_ylabel("metric value")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Generalisation by league — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_by_league_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_reliability(label: str) -> None:
    """Raw success rate vs the models on ranking reproducibility — the model's reason to exist:
    the raw rate is noise, the situation-based score recovers a stable ranking."""
    r = pl.read_csv(METRICS / f"reliability_{label}.csv")
    metrics_show = [("stability_random", "split-half (random)"),
                    ("stability_temporal", "temporal (early vs late)"),
                    ("icc", "ICC")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _grouped_bars(ax, r["ranking"].to_list(), [(lbl, r[col].to_list()) for col, lbl in metrics_show])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ranking reproducibility (higher = better)")
    ax.set_title(f"Does the player ranking measure a stable trait? — {label}")
    ax.legend(fontsize=9)
    fig.savefig(FIGURES / f"chart_reliability_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_ablation(label: str) -> None:
    """Impact of the spatial-entropy block: each feature set with vs without entropy."""
    a = pl.read_csv(METRICS / f"ablation_{label}.csv")
    metrics_show = [("auc", "AUC"), ("auc_pr", "AUC-PR"), ("brier_skill", "Brier skill"),
                    ("stability", "stability"), ("icc", "ICC")]
    feature_sets = a["feature_set"].unique(maintain_order=True).to_list()
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(6.2 * len(feature_sets), 5), squeeze=False, sharey=True)
    for ax, fs in zip(axes[0], feature_sets, strict=True):
        sub = a.filter(pl.col("feature_set") == fs)
        with_e = sub.filter(pl.col("variant") == "with_entropy")
        no_e = sub.filter(pl.col("variant") == "no_entropy")
        categories = [lbl for _, lbl in metrics_show]
        ax.bar([x - 0.2 for x in range(len(categories))], [with_e[c][0] for c, _ in metrics_show],
               width=0.4, label="with entropy", color="#2ca02c")
        ax.bar([x + 0.2 for x in range(len(categories))], [no_e[c][0] for c, _ in metrics_show],
               width=0.4, label="no entropy", color="#b0b0b0")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=8, rotation=15)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{fs}  ({with_e['n_features'][0]} → {no_e['n_features'][0]} features)")
    axes[0][0].set_ylabel("metric value")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Entropy ablation — impact of the spatial-entropy features ({label})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_ablation_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- per (feature_set, label) ---

def chart_importance(fs: str, label: str, top: int = 20) -> None:
    s = pl.read_csv(METRICS / f"importance_{fs}_{label}.csv").head(top).reverse()
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * top + 1.2))
    ax.barh(s["feature"], s["importance"], color="#2ca02c")
    ax.set_xlabel("feature importance (final model)")
    ax.set_title(f"Top {top} features — {fs}/{label} (final model)")
    fig.savefig(FIGURES / f"chart_importance_{fs}_{label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_lift(fs: str, label: str) -> None:
    preds = pl.read_csv(METRICS / "oof_predictions.csv")
    table = lift_by_decile(preds[label].to_numpy(), preds[f"prob_{fs}_{label}"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(table["decile"], table["actual_rate"], alpha=0.6, label="actual rate")
    ax.plot(table["decile"], table["mean_pred"], "o-", color="black", label="predicted prob.")
    ax.set_xlabel("predicted-probability decile (1 = worst, 10 = best)")
    ax.set_ylabel(f"{label} rate")
    ax.set_title(f"Lift / ordering — {fs}/{label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"lift_{fs}_{label}.png", dpi=110)
    plt.close(fig)


def _safe(fn, *args) -> bool:
    try:
        fn(*args)
        return True
    except Exception as exc:  # one broken chart must not drop the others
        logger.warning(f"skip {fn.__name__}{args}: {type(exc).__name__}: {exc}")
        return False


def run() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ok = total = 0
    for fn in (table_model_metrics, table_comparison):
        total += 1
        ok += _safe(fn)
    for label in LABELS:
        for fn in (table_league_metrics, chart_stability_vs_n, chart_ranking_top,
                   chart_ranking_quadrants, chart_ranking_quadrants_by_league,
                   chart_by_position, chart_by_league, chart_reliability, chart_ablation):
            total += 1
            ok += _safe(fn, label)
    for fs in FEATURE_SETS:
        for label in LABELS:
            for fn in (chart_importance, chart_lift):
                total += 1
                ok += _safe(fn, fs, label)
    logger.info(f"Figures written to {FIGURES} ({ok}/{total} ok)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
