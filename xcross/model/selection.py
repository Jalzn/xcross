"""Pick the headline model per target from the training results (comparison.csv).

The report uses these choices instead of hardcoding an estimator: for the player-ranking
objective the criterion is stability (reproducibility), tie-broken by ECE (calibration).
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from xcross.config import ROOT

COMPARISON = ROOT / "artifacts" / "reports" / "metrics" / "comparison.csv"
FALLBACK = {"estimator": "adaboost", "calibration": "isotonic"}


def load_comparison() -> pl.DataFrame | None:
    return pl.read_csv(COMPARISON) if COMPARISON.exists() else None


def select_best(
    comparison: pl.DataFrame | None, feature_set: str, label: str, by: str = "stability"
) -> dict[str, str]:
    """Best (estimator, calibration) for a target: max `by`, tie-broken by lowest ECE."""
    if comparison is None:
        logger.warning(f"comparison.csv missing; falling back to {FALLBACK} for {feature_set}/{label}")
        return dict(FALLBACK)
    sub = comparison.filter((pl.col("feature_set") == feature_set) & (pl.col("label") == label))
    if sub.height == 0:
        return dict(FALLBACK)
    best = sub.sort([by, "ece"], descending=[True, False]).row(0, named=True)
    return {"estimator": best["estimator"], "calibration": best["calibration"]}
