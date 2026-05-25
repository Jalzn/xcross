"""Model-comparison figures for the paper's model-selection section: they justify the
headline choice with evidence, not a single number. Read the comparison/robustness CSVs and
the OOF matrix; write artifacts/reports/figures/*.png.

    uv run python -m xcross.model.comparison_figures
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve

from xcross.model.figures import FEATURE_SETS, FIGURES, LABELS, METRICS

CAL = "isotonic"
FAMILY = {
    "logreg": "linear", "random_forest": "bagging", "tabpfn": "foundation",
    "xgboost": "boosting", "adaboost": "boosting", "catboost": "boosting",
    "lightgbm": "boosting", "histgb": "boosting",
}
FAMILY_COLOR = {"linear": "#7f7f7f", "bagging": "#2ca02c", "boosting": "#1f77b4", "foundation": "#d62728"}
FS_TITLE = {"xcross": "xCross (creation)", "xcrossot": "xCrossOT (danger)"}


def _family_legend() -> list[Line2D]:
    return [Line2D([0], [0], marker="o", linestyle="", color=c, label=f) for f, c in FAMILY_COLOR.items()]


def chart_tradeoff_by_family(label: str = "success") -> None:
    """Discrimination (AUC) × reproducibility (stability_temporal), one panel per model,
    coloured by family — shows no model wins both axes, justifying a per-objective choice."""
    c = pl.read_csv(METRICS / "comparison.csv").filter(
        (pl.col("calibration") == CAL) & (pl.col("label") == label)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, fs in zip(axes, FEATURE_SETS, strict=True):
        sub = c.filter(pl.col("feature_set") == fs)
        for r in sub.iter_rows(named=True):
            ax.scatter(r["auc"], r["stability_temporal"], s=170, zorder=3,
                       color=FAMILY_COLOR[FAMILY.get(r["estimator"], "boosting")],
                       edgecolor="black", linewidth=0.8)
            ax.annotate(r["estimator"], (r["auc"], r["stability_temporal"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)
        ax.set_title(FS_TITLE[fs])
        ax.set_xlabel("AUC — discrimination")
        ax.set_ylabel("temporal split-half — ranking reproducibility")
        ax.grid(alpha=0.3)
    axes[0].legend(handles=_family_legend(), title="family", loc="best", fontsize=8)
    fig.suptitle(f"Model trade-off: discrimination × reproducibility — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_model_tradeoff_{label}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_robustness(label: str) -> None:
    """AUC and stability per model with bootstrap CIs — the evidence that a difference is real
    (non-overlapping intervals) rather than noise."""
    r = pl.read_csv(METRICS / "robustness.csv").filter(
        (pl.col("calibration") == CAL) & (pl.col("label") == label)
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, fs in enumerate(FEATURE_SETS):
        sub = r.filter(pl.col("feature_set") == fs)
        for row, (metric, lo, hi) in enumerate([("auc", "auc_lo", "auc_hi"),
                                                ("stability", "stability_lo", "stability_hi")]):
            ax = axes[row, col]
            s = sub.sort(metric)
            ys = np.arange(s.height)
            colors = [FAMILY_COLOR[FAMILY.get(e, "boosting")] for e in s["estimator"]]
            ax.errorbar(s[metric], ys, xerr=[s[metric] - s[lo], s[hi] - s[metric]],
                        fmt="none", ecolor="#999", zorder=1, capsize=3)
            ax.scatter(s[metric], ys, color=colors, s=90, zorder=2, edgecolor="black", linewidth=0.6)
            ax.set_yticks(ys)
            ax.set_yticklabels(s["estimator"].to_list(), fontsize=8)
            ax.set_xlabel(metric)
            ax.set_title(f"{FS_TITLE[fs]} — {'AUC' if metric == 'auc' else 'stability'} (95% CI)")
            ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Model robustness (bootstrap 95% CI) — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_model_robustness_{label}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_ranking_agreement(label: str) -> None:
    """Spearman between the player rankings of each pair of models — is the crosser ranking
    robust to the model choice, or an artefact of it?"""
    a = pl.read_csv(METRICS / "ranking_agreement.csv").filter(pl.col("label") == label)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, fs in zip(axes, FEATURE_SETS, strict=True):
        sub = a.filter(pl.col("feature_set") == fs)
        models = sorted(set(sub["model_a"].to_list()) | set(sub["model_b"].to_list()))
        idx = {m: i for i, m in enumerate(models)}
        grid = np.full((len(models), len(models)), np.nan)
        for r in sub.iter_rows(named=True):
            i, j = idx[r["model_a"]], idx[r["model_b"]]
            grid[i, j] = grid[j, i] = r["spearman"]
        im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(models)), models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(models)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[i, j] < 0.6 else "black")
        ax.set_title(FS_TITLE[fs])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Ranking agreement between models (Spearman) — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_ranking_agreement_{label}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_calibration_compare(label: str) -> None:
    """Reliability curves of every model overlaid (one panel per feature set) — shows which
    models stay calibrated, a dimension the AUC trade-off hides."""
    oof = pl.read_parquet(METRICS / "oof_matrix.parquet")
    y = oof[label].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, fs in zip(axes, FEATURE_SETS, strict=True):
        cols = {c.split("__")[0]: c for c in oof.columns if c.endswith(f"__{fs}__{label}__{CAL}")}
        for est in sorted(cols):
            frac, mean = calibration_curve(y, oof[cols[est]].to_numpy(), n_bins=10, strategy="quantile")
            ax.plot(mean, frac, marker="o", markersize=3, linewidth=1.3,
                    color=FAMILY_COLOR[FAMILY.get(est, "boosting")], alpha=0.8, label=est)
        ax.plot([0, 1], [0, 1], "--", color="#444", linewidth=1)
        ax.set_title(FS_TITLE[fs])
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Calibration by model — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"chart_calibration_compare_{label}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def run() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        chart_tradeoff_by_family(label)
        chart_robustness(label)
        chart_ranking_agreement(label)
        chart_calibration_compare(label)
    logger.info("Wrote model-comparison figures.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
