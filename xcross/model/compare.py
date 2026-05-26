"""CLI: train the matrix {xCross,xCrossOT} × {success,shot} × estimators × calibrations,
score each on OOF probabilities, and write the comparison table (raw data only).

TabPFN joins the registry when `XCROSS_TABPFN=1` (typically on a GPU host); locally on macOS it
stays out by default to avoid the OpenMP clash with xgboost/lightgbm. report.py reads this table
to pick the best model per target (see selection.py) — not hardcoded. Figures come from figures.py
/ comparison_figures.py.

    uv run python -m xcross.model.compare
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from loguru import logger

from xcross.config import ROOT
from xcross.model.dataset import FEATURE_SETS, LABEL_COLS, load_features, make_xy, match_dates
from xcross.model.estimators import ESTIMATORS
from xcross.model.evaluate import metrics
from xcross.model.train import oof_predict

CALIBRATIONS = ("isotonic", "sigmoid")
METRICS_DIR = ROOT / "artifacts" / "reports" / "metrics"


def run() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    dates = match_dates()
    order_key = np.array([dates.get(m, m) for m in df["match_id"].to_list()])
    logger.info(f"Loaded {df.height} crosses.")

    rows: list[dict] = []
    oof: dict[str, object] = {
        "cross_id": df["cross_id"], "crosser_player_id": df["crosser_player_id"],
        "success": df["success"].cast(pl.Int8), "shot": df["shot_in_window"].cast(pl.Int8),
    }
    for feature_set in FEATURE_SETS:
        for label in LABEL_COLS:
            label_name = "shot" if label == "shot_in_window" else label
            X, y, groups, _ = make_xy(df, feature_set, label)
            for name, factory in ESTIMATORS.items():
                for calibration in CALIBRATIONS:
                    prob = oof_predict(factory, X, y, groups, calibration)
                    oof[f"{name}__{feature_set}__{label_name}__{calibration}"] = prob
                    record = metrics(y, prob, player_ids, order_key)
                    rows.append({
                        "feature_set": feature_set, "label": label_name,
                        "estimator": name, "calibration": calibration, **record,
                    })
                    logger.info(
                        f"{feature_set}/{label}/{name}/{calibration}: "
                        f"auc={record['auc']:.3f} ece={record['ece']:.3f} "
                        f"stab={record['stability']:.2f} stab_t={record['stability_temporal']:.2f} "
                        f"icc={record['icc']:.2f}"
                    )

    pl.DataFrame(oof).write_parquet(METRICS_DIR / "oof_matrix.parquet")
    logger.info(f"Wrote oof_matrix.parquet ({len(oof) - 4} model columns).")

    pl.DataFrame(rows).write_csv(METRICS_DIR / "comparison.csv")
    logger.info(f"Wrote comparison.csv ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
