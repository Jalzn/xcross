"""Pick the headline model per target from the training results (comparison.csv).

The criterion matches each model's job: xCross is the *creation* score, judged on ranking
reproducibility (`stability_temporal` — does early-season crossing predict late-season?);
xCrossOT is the *danger* score, judged on discrimination (`auc`). Ties break by ECE.

The headline is restricted to estimators the report can retrain in-process (`eligible`):
TabPFN appears in the comparison for benchmarking but is not a production candidate, since
it only joins the registry on a GPU host (`XCROSS_TABPFN=1`).
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
from loguru import logger

from xcross.config import ROOT

COMPARISON = ROOT / "artifacts" / "reports" / "metrics" / "comparison.csv"
FALLBACK = {"estimator": "adaboost", "calibration": "isotonic"}
CRITERION = {"xcross": "stability_temporal", "xcrossot": "auc"}


def load_comparison() -> pl.DataFrame | None:
    return pl.read_csv(COMPARISON) if COMPARISON.exists() else None


def select_best(
    comparison: pl.DataFrame | None,
    feature_set: str,
    label: str,
    by: str | None = None,
    eligible: Iterable[str] | None = None,
) -> dict[str, str]:
    """Best (estimator, calibration) for a target: max `by`, tie-broken by lowest ECE.

    `by` defaults to the per-model criterion (CRITERION). `eligible` restricts the choice to
    estimators the caller can instantiate (the report passes its in-process registry, excluding
    TabPFN)."""
    if by is None:
        by = CRITERION.get(feature_set, "stability_temporal")
    if comparison is None:
        logger.warning(f"comparison.csv missing; falling back to {FALLBACK} for {feature_set}/{label}")
        return dict(FALLBACK)
    sub = comparison.filter((pl.col("feature_set") == feature_set) & (pl.col("label") == label))
    if eligible is not None:
        sub = sub.filter(pl.col("estimator").is_in(list(eligible)))
    if sub.height == 0:
        return dict(FALLBACK)
    if by not in sub.columns:
        logger.warning(f"'{by}' not in comparison.csv; selecting by 'stability' for {feature_set}/{label}")
        by = "stability"
    best = sub.sort([by, "ece"], descending=[True, False]).row(0, named=True)
    return {"estimator": best["estimator"], "calibration": best["calibration"]}
