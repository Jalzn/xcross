"""Metrics for the ranking objective: discrimination, calibration, spread, stability."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

RANKING_MIN_CROSSES = 20


def brier_skill_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Brier improvement over always predicting the base rate. >0 means it adds value."""
    base = float(y_true.mean())
    reference = base * (1 - base)
    return float(1 - brier_score_loss(y_true, prob) / reference) if reference > 0 else float("nan")


def calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of y on logit(prob). slope≈1, intercept≈0 = well calibrated;
    slope<1 = over-confident, slope>1 = under-confident."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(solver="lbfgs").fit(logit, y_true)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean gap between predicted confidence and observed frequency, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bin_of == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(error)


def spread(prob: np.ndarray) -> dict[str, float]:
    pct = np.percentile(prob, [5, 25, 50, 75, 95])
    return {
        "std": float(prob.std()),
        "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
        "p75": float(pct[3]), "p95": float(pct[4]),
        "range_5_95": float(pct[4] - pct[0]),
    }


def lift_by_decile(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Actual success rate per predicted-probability decile (monotonic = good ordering)."""
    bins = np.array_split(np.argsort(prob), n_bins)
    return pl.DataFrame({
        "decile": list(range(1, n_bins + 1)),
        "mean_pred": [float(prob[b].mean()) for b in bins],
        "actual_rate": [float(y_true[b].mean()) for b in bins],
        "n": [int(len(b)) for b in bins],
    })


def player_discrimination(
    player_ids: np.ndarray, prob: np.ndarray, min_crosses: int = RANKING_MIN_CROSSES
) -> float:
    """Std of per-player mean probability — how much the ranking separates players
    (the spread that matters for ranking, vs the spread of raw per-cross probs)."""
    means = (
        pl.DataFrame({"player_id": player_ids, "prob": prob})
        .group_by("player_id")
        .agg(pl.len().alias("n"), pl.col("prob").mean().alias("m"))
        .filter(pl.col("n") >= min_crosses)
    )
    if means.height < 2:
        return float("nan")
    return float(means["m"].std())


def metrics(y_true: np.ndarray, prob: np.ndarray, player_ids: np.ndarray | None = None) -> dict[str, float]:
    slope, intercept = calibration_slope_intercept(y_true, prob)
    out = {
        "auc": float(roc_auc_score(y_true, prob)),
        "auc_pr": float(average_precision_score(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "brier_skill": brier_skill_score(y_true, prob),
        "ece": expected_calibration_error(y_true, prob),
        "cal_slope": slope,
        "cal_intercept": intercept,
        **spread(prob),
    }
    if player_ids is not None:
        out["player_discrimination"] = player_discrimination(player_ids, prob)
        out["stability"] = split_half_stability(player_ids, prob)
        out["icc"] = icc(player_ids, prob)
    return out


def player_ranking(
    player_ids: np.ndarray, prob: np.ndarray, min_crosses: int = RANKING_MIN_CROSSES
) -> pl.DataFrame:
    """Mean (and sum) predicted probability per player with at least `min_crosses`."""
    return (
        pl.DataFrame({"player_id": player_ids, "prob": prob})
        .group_by("player_id")
        .agg(pl.len().alias("n"), pl.col("prob").mean().alias("mean"), pl.col("prob").sum().alias("sum"))
        .filter(pl.col("n") >= min_crosses)
        .sort("mean", descending=True)
    )


def combined_ranking(
    player_ids: np.ndarray,
    prob_creation: np.ndarray,
    prob_danger: np.ndarray,
    y_true: np.ndarray,
    min_crosses: int = RANKING_MIN_CROSSES,
) -> pl.DataFrame:
    """Per-player ranking with: xCross (creation, stable), xCrossOT (danger), execution
    (= xCrossOT - xCross), actual success rate, over_expected (= actual - xCross), and a
    95% CI for xCross (mean ± 1.96·SE) so each player's position carries its uncertainty."""
    return (
        pl.DataFrame({
            "player_id": player_ids, "xcross": prob_creation,
            "xcrossot": prob_danger, "actual": y_true,
        })
        .group_by("player_id")
        .agg(
            pl.len().alias("n"),
            pl.col("xcross").mean().alias("xcross"),
            pl.col("xcross").std().alias("_sd"),
            pl.col("xcrossot").mean().alias("xcrossot"),
            pl.col("actual").mean().alias("actual"),
        )
        .filter(pl.col("n") >= min_crosses)
        .with_columns((pl.col("_sd") / pl.col("n").sqrt()).alias("xcross_se"))
        .with_columns(
            (pl.col("xcrossot") - pl.col("xcross")).alias("execution"),
            (pl.col("actual") - pl.col("xcross")).alias("over_expected"),
            (pl.col("xcross") - 1.96 * pl.col("xcross_se")).alias("xcross_ci_low"),
            (pl.col("xcross") + 1.96 * pl.col("xcross_se")).alias("xcross_ci_high"),
        )
        .drop("_sd")
        .sort("xcross", descending=True)
    )


def icc(player_ids: np.ndarray, prob: np.ndarray, min_crosses: int = RANKING_MIN_CROSSES) -> float:
    """Intraclass correlation: fraction of probability variance that is *between* players
    (true skill) vs *within* (per-cross noise). The canonical reliability of the ranking;
    0 = all noise, 1 = all stable player differences."""
    g = (
        pl.DataFrame({"pid": player_ids, "p": prob})
        .group_by("pid")
        .agg(pl.len().alias("n"), pl.col("p").mean().alias("m"), pl.col("p").var().alias("v"))
        .filter(pl.col("n") >= min_crosses)
    )
    if g.height < 3:
        return float("nan")
    n_avg = float(g["n"].mean())
    within = float((g["v"].fill_null(0.0) * (g["n"] - 1)).sum() / (g["n"] - 1).sum())
    between_observed = float(g["m"].var())
    between = max(0.0, between_observed - within / n_avg)
    return between / (between + within) if (between + within) > 0 else float("nan")


def rank_agreement(
    player_ids: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
    min_crosses: int = RANKING_MIN_CROSSES,
) -> float:
    """Spearman correlation between two per-player rankings (e.g. success vs shot)."""
    a = player_ranking(player_ids, prob_a, min_crosses).select("player_id", pl.col("mean").alias("a"))
    b = player_ranking(player_ids, prob_b, min_crosses).select("player_id", pl.col("mean").alias("b"))
    joined = a.join(b, on="player_id")
    if joined.height < 3:
        return float("nan")
    return float(spearmanr(joined["a"], joined["b"]).statistic)


def stability_curve(
    player_ids: np.ndarray,
    prob: np.ndarray,
    thresholds: tuple[int, ...] = (10, 15, 20, 30, 40, 50),
    seed: int = 0,
) -> pl.DataFrame:
    """Split-half stability and surviving player count at each min-crosses threshold."""
    counts = pl.DataFrame({"player_id": player_ids}).group_by("player_id").len()
    return pl.DataFrame({
        "min_crosses": list(thresholds),
        "n_players": [int((counts["len"] >= t).sum()) for t in thresholds],
        "stability": [split_half_stability(player_ids, prob, t, seed) for t in thresholds],
    })


def split_half_stability(
    player_ids: np.ndarray, prob: np.ndarray, min_crosses: int = RANKING_MIN_CROSSES, seed: int = 0
) -> float:
    """Spearman corr between per-player mean prob on two random halves of their crosses.

    High = the model measures a stable player trait rather than noise."""
    rng = np.random.default_rng(seed)
    frame = pl.DataFrame({"player_id": player_ids, "prob": prob})
    first, second = [], []
    for (_,), group in frame.group_by("player_id", maintain_order=True):
        values = group["prob"].to_numpy()
        if len(values) < min_crosses:
            continue
        order = rng.permutation(len(values))
        cut = len(values) // 2
        first.append(values[order[:cut]].mean())
        second.append(values[order[cut:]].mean())
    if len(first) < 3:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def temporal_split_stability(
    player_ids: np.ndarray, prob: np.ndarray, order_key: np.ndarray,
    min_crosses: int = RANKING_MIN_CROSSES,
) -> float:
    """Like split_half_stability but the halves are *chronological* (early vs late crosses,
    ordered by `order_key`, e.g. match date) instead of random. The stricter test of a stable
    trait: does a player's early-season crossing predict their late-season crossing?"""
    frame = pl.DataFrame({"player_id": player_ids, "prob": prob, "order_key": order_key}).sort("order_key")
    early, late = [], []
    for (_,), group in frame.group_by("player_id", maintain_order=True):
        values = group["prob"].to_numpy()
        if len(values) < min_crosses:
            continue
        cut = len(values) // 2
        early.append(values[:cut].mean())
        late.append(values[cut:].mean())
    if len(early) < 3:
        return float("nan")
    return float(spearmanr(early, late).statistic)
