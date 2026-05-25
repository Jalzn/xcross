"""CLI: bootstrap confidence intervals for the model-comparison metrics, read straight from
the saved OOF predictions (oof_matrix.parquet) — no re-training, so it is cheap.

Discrimination (AUC) gets a CI by resampling crosses; ranking reproducibility (split-half
stability) gets one by varying the split seed. These intervals are what justify the headline
choice with evidence rather than a single number.

    uv run python -m xcross.model.robustness
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import roc_auc_score

from xcross.config import ROOT
from xcross.model.evaluate import rank_agreement, split_half_stability

METRICS = ROOT / "artifacts" / "reports" / "metrics"
OOF_MATRIX = METRICS / "oof_matrix.parquet"
N_BOOT = 500
N_SEEDS = 200
PCTL = (2.5, 50.0, 97.5)


def _auc_ci(y: np.ndarray, prob: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    values = []
    for _ in range(N_BOOT):
        sample = rng.choice(idx, len(idx), replace=True)
        if 0 < y[sample].sum() < len(sample):
            values.append(roc_auc_score(y[sample], prob[sample]))
    lo, mid, hi = np.percentile(values, PCTL)
    return float(lo), float(mid), float(hi)


def _stability_ci(player_ids: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    values = [split_half_stability(player_ids, prob, seed=s) for s in range(N_SEEDS)]
    values = [v for v in values if not np.isnan(v)]
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    lo, mid, hi = np.percentile(values, PCTL)
    return float(lo), float(mid), float(hi)


def _agreement_rows(oof: pl.DataFrame, player_ids: np.ndarray) -> list[dict]:
    """Spearman between the player rankings of every pair of estimators (same target),
    using the isotonic predictions — does the crosser ranking survive the model choice?"""
    rows: list[dict] = []
    targets = {(c.split("__")[1], c.split("__")[2]) for c in oof.columns if c.endswith("__isotonic")}
    for feature_set, label in sorted(targets):
        cols = {c.split("__")[0]: c for c in oof.columns if c.endswith(f"__{feature_set}__{label}__isotonic")}
        names = sorted(cols)
        for i, a in enumerate(names):
            for b in names[i:]:
                spearman = rank_agreement(player_ids, oof[cols[a]].to_numpy(), oof[cols[b]].to_numpy())
                rows.append({"feature_set": feature_set, "label": label,
                             "model_a": a, "model_b": b, "spearman": spearman})
    return rows


def run() -> int:
    if not OOF_MATRIX.exists():
        logger.error("oof_matrix.parquet missing; run `python -m xcross.model.compare` first.")
        return 1
    oof = pl.read_parquet(OOF_MATRIX)
    player_ids = oof["crosser_player_id"].to_numpy()
    model_cols = [c for c in oof.columns if "__" in c]
    logger.info(f"Bootstrapping {len(model_cols)} model columns ...")

    rows: list[dict] = []
    for col in model_cols:
        estimator, feature_set, label, calibration = col.split("__")
        y = oof[label].to_numpy()
        prob = oof[col].to_numpy()
        auc_lo, auc, auc_hi = _auc_ci(y, prob)
        stab_lo, stab, stab_hi = _stability_ci(player_ids, prob)
        rows.append({
            "estimator": estimator, "feature_set": feature_set, "label": label, "calibration": calibration,
            "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "stability": stab, "stability_lo": stab_lo, "stability_hi": stab_hi,
        })
    pl.DataFrame(rows).write_csv(METRICS / "robustness.csv")
    logger.info(f"Wrote robustness.csv ({len(rows)} rows).")

    pl.DataFrame(_agreement_rows(oof, player_ids)).write_csv(METRICS / "ranking_agreement.csv")
    logger.info("Wrote ranking_agreement.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
