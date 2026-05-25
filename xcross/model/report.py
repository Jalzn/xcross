"""Final evaluation report.

For each label (success, shot) writes per-label raw CSVs (ranking, league, stability),
plus shared CSVs (comparison via compare.py, model_metrics, oof_predictions) and the
per-(feature_set, label) importance CSVs and PDP figures. figures.py then renders every
CSV. The headline estimator per target is read from comparison.csv (selection.py).

    uv run python -m xcross.model.report
"""

from __future__ import annotations

import glob
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from xcross.config import ROOT
from xcross.model.dataset import load_features, make_xy, match_dates
from xcross.model.estimators import ESTIMATORS
from xcross.model.evaluate import (
    combined_ranking,
    icc,
    metrics,
    player_discrimination,
    rank_agreement,
    split_half_stability,
    stability_curve,
    temporal_split_stability,
)
from xcross.model.figures import run as render_figures
from xcross.model.selection import load_comparison, select_best
from xcross.model.train import oof_predict

METRICS = ROOT / "artifacts" / "reports" / "metrics"
FIGURES = ROOT / "artifacts" / "reports" / "figures"
MIN_LEAGUE_CROSSES = 300
MIN_LEAGUE_RANKED_PLAYERS = 6  # below this a per-league quadrant chart is too sparse to read
LABELS = ("success", "shot")
LABEL_COLUMN = {"success": "success", "shot": "shot_in_window"}  # short label -> feature column / comparison.csv label
FEATURE_SETS = ("xcross", "xcrossot")
ABLATION_METRICS = ("auc", "auc_pr", "brier_skill", "ece", "stability", "icc", "player_discrimination")


def _roster() -> pl.DataFrame:
    files = glob.glob(str(ROOT / "data" / "processed" / "*" / "*" / "*" / "roster.parquet"))
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .unique("player_id")
        .select("player_id", "nickname", "position_group")
    )


def _feature_importance(model: object, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Native feature_importances_ for tree models; permutation importance on AUC otherwise
    (model-agnostic — covers logreg and TabPFN, which expose no native importance)."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=5,
        max_samples=min(2000, len(X)), random_state=0, n_jobs=-1,
    )
    return result.importances_mean


def _model_importance(estimator: str, X: np.ndarray, y: np.ndarray, names: list[str], tag: str) -> None:
    """Feature importance of the *selected* final model — what we actually use."""
    model = ESTIMATORS[estimator]().fit(X, y)
    pl.DataFrame({"feature": names, "importance": _feature_importance(model, X, y)}).sort(
        "importance", descending=True
    ).write_csv(METRICS / f"importance_{tag}.csv")


PDP_TOP = 9        # features shown (3x3 grid), the most important ones in the final model
PDP_SAMPLE = 1000  # rows averaged for the partial dependence (keeps it fast)
PDP_ABOVE = "#2e8b57"  # curve above the average prediction -> raises the probability
PDP_BELOW = "#c0392b"  # curve below the average prediction -> lowers it
PDP_LINE = "#1b2a4a"


def _pretty_feature(name: str) -> str:
    return name.replace("_", " ").title()


def _pdp_figure(estimator: str, X: np.ndarray, y: np.ndarray, names: list[str], tag: str) -> None:
    """Partial-dependence grid of the FINAL model: how the predicted probability moves as
    each top feature varies (direction and shape of the effect, complementing the magnitude
    in chart_importance). Each panel shades the curve above/below the model's average
    prediction, and a rug shows where the real crosses lie."""
    model = ESTIMATORS[estimator]().fit(X, y)
    top = np.argsort(_feature_importance(model, X, y))[::-1][:PDP_TOP]
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(PDP_SAMPLE, len(X)), replace=False)]
    baseline = float(model.predict_proba(sample)[:, 1].mean())

    display = PartialDependenceDisplay.from_estimator(
        model, sample, features=list(top), feature_names=names,
        kind="average", grid_resolution=40, n_cols=3,
        response_method="predict_proba", line_kw={"color": PDP_LINE, "linewidth": 2.2},
    )
    if display.deciles_vlines_ is not None:  # drop sklearn's default deciles; we draw our own rug
        for vline in np.ravel(display.deciles_vlines_):
            if vline is not None:
                vline.set_visible(False)

    for rank, (ax, line, feature) in enumerate(
        zip(display.axes_.ravel(), display.lines_.ravel(), top, strict=False), start=1
    ):
        grid_x, grid_y = line.get_xdata(), line.get_ydata()
        ax.fill_between(grid_x, grid_y, baseline, where=grid_y >= baseline,
                        interpolate=True, color=PDP_ABOVE, alpha=0.18)
        ax.fill_between(grid_x, grid_y, baseline, where=grid_y < baseline,
                        interpolate=True, color=PDP_BELOW, alpha=0.18)
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(sample[:, feature], np.full(len(sample), ax.get_ylim()[0]), "|", color="black", alpha=0.05, ms=8)
        ax.set_title(f"{rank}. {_pretty_feature(names[feature])}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in display.axes_[:, 0]:
        ax.set_ylabel("Predicted probability")

    fig = display.figure_
    fig.set_size_inches(15, 13)
    fig.suptitle(f"How each feature moves the prediction — {tag} ({estimator})", fontweight="bold", fontsize=15)
    fig.text(0.5, 0.945, f"Ordered by importance.  Green = above the model's average ({baseline:.0%}), "
             f"red = below.  Ticks at the bottom show where the real crosses are.",
             ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / f"pdp_{tag}.png", dpi=110)
    plt.close(fig)


def run() -> int:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    leagues = df["league"].to_numpy()
    order_key = np.array([match_dates().get(m, m) for m in df["match_id"].to_list()])
    X = {fs: make_xy(df, fs, "success") for fs in FEATURE_SETS}  # X identical across labels
    y = {label: df[LABEL_COLUMN[label]].cast(pl.Int8).to_numpy() for label in LABELS}
    groups = X["xcross"][2]

    comparison = load_comparison()
    eligible = set(ESTIMATORS)
    choice = {(fs, label): select_best(comparison, fs, LABEL_COLUMN[label], eligible=eligible)
              for fs in FEATURE_SETS for label in LABELS}

    logger.info("OOF for the 4 targets with the selected models ...")
    oof = {(fs, label): oof_predict(ESTIMATORS[choice[(fs, label)]["estimator"]],
                                    X[fs][0], y[label], groups, choice[(fs, label)]["calibration"])
           for fs in FEATURE_SETS for label in LABELS}
    for (fs, label), ch in choice.items():
        logger.info(f"{fs}/{label}: {ch['estimator']}/{ch['calibration']}")

    preds = pl.DataFrame({"cross_id": df["cross_id"], "success": y["success"], "shot": y["shot"]})
    for (fs, label), prob in oof.items():
        preds = preds.with_columns(pl.Series(f"prob_{fs}_{label}", prob))
    preds.write_csv(METRICS / "oof_predictions.csv")

    pl.DataFrame([
        {"model": f"{fs}/{label}", "estimator": choice[(fs, label)]["estimator"],
         "calibration": choice[(fs, label)]["calibration"],
         **metrics(y[label], oof[(fs, label)], player_ids, order_key)}
        for fs in FEATURE_SETS for label in LABELS
    ]).write_csv(METRICS / "model_metrics.csv")

    # Ablation: same selected model, entropy block removed -> isolates the entropy contribution.
    X_noent = {fs: make_xy(df, f"{fs}_noent", "success") for fs in FEATURE_SETS}
    logger.info("OOF for the no-entropy ablation ...")
    oof_noent = {(fs, label): oof_predict(ESTIMATORS[choice[(fs, label)]["estimator"]],
                                          X_noent[fs][0], y[label], groups, choice[(fs, label)]["calibration"])
                 for fs in FEATURE_SETS for label in LABELS}

    roster = _roster()
    for label in LABELS:
        prob_c, prob_d, yl = oof[("xcross", label)], oof[("xcrossot", label)], y[label]
        combined_ranking(player_ids, prob_c, prob_d, yl).join(roster, on="player_id", how="left") \
            .write_csv(METRICS / f"ranking_final_{label}.csv")

        per_league = []
        for lg in sorted(set(leagues)):
            mask = leagues == lg
            ranked = combined_ranking(player_ids[mask], prob_c[mask], prob_d[mask], yl[mask])
            if ranked.height >= MIN_LEAGUE_RANKED_PLAYERS:
                per_league.append(ranked.join(roster, on="player_id", how="left")
                                  .with_columns(pl.lit(lg).alias("league")))
        if per_league:
            pl.concat(per_league).write_csv(METRICS / f"ranking_by_league_{label}.csv")

        # Ranking reliability: raw rate (noise baseline) vs the models, random and chronological.
        sources = {"raw_rate": yl.astype(float), "xcross": prob_c, "xcrossot": prob_d}
        pl.DataFrame([
            {"ranking": name,
             "stability_random": split_half_stability(player_ids, v),
             "stability_temporal": temporal_split_stability(player_ids, v, order_key),
             "icc": icc(player_ids, v),
             "player_discrimination": player_discrimination(player_ids, v)}
            for name, v in sources.items()
        ]).write_csv(METRICS / f"reliability_{label}.csv")

        # Entropy ablation: each feature set with vs without the entropy block.
        ablation = []
        for fs in FEATURE_SETS:
            for variant, prob, x_set in (("with_entropy", oof[(fs, label)], X),
                                         ("no_entropy", oof_noent[(fs, label)], X_noent)):
                m = metrics(yl, prob, player_ids)
                ablation.append({"feature_set": fs, "variant": variant,
                                 "n_features": x_set[fs][0].shape[1],
                                 **{k: m[k] for k in ABLATION_METRICS}})
        pl.DataFrame(ablation).write_csv(METRICS / f"ablation_{label}.csv")

        league_rows = [
            {"league": lg, "feature_set": fs, "n": int((leagues == lg).sum()),
             **metrics(yl[leagues == lg], oof[(fs, label)][leagues == lg], player_ids[leagues == lg])}
            for lg in sorted(set(leagues)) if int((leagues == lg).sum()) >= MIN_LEAGUE_CROSSES
            for fs in FEATURE_SETS
        ]
        pl.DataFrame(league_rows).write_csv(METRICS / f"league_metrics_{label}.csv")
        stability_curve(player_ids, prob_c).write_csv(METRICS / f"stability_vs_n_{label}.csv")

    logger.info(f"xCross ranking: success vs shot Spearman = "
                f"{rank_agreement(player_ids, oof[('xcross', 'success')], oof[('xcross', 'shot')]):.2f}")

    for fs in FEATURE_SETS:
        for label in LABELS:
            _model_importance(choice[(fs, label)]["estimator"], X[fs][0], y[label], X[fs][3], f"{fs}_{label}")
            _pdp_figure(choice[(fs, label)]["estimator"], X[fs][0], y[label], X[fs][3], f"{fs}_{label}")

    render_figures()
    docs = ROOT / "docs" / "metrics.md"
    if docs.exists():
        shutil.copy(docs, FIGURES.parent / "README.md")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
