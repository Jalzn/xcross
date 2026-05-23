"""CLI: train the matrix {xCross,xCrossOT} × {success,shot} × estimators × calibrations,
score each on OOF probabilities, and write the comparison table (raw data only).

report.py reads this table to pick the best model per target (see selection.py) — it is
not hardcoded. Figures are produced by figures.py / report.py.

    uv run python -m xcross.model.compare
"""

from __future__ import annotations

import sys

import polars as pl
from loguru import logger

from xcross.config import ROOT
from xcross.model.dataset import FEATURE_SETS, LABEL_COLS, load_features, make_xy
from xcross.model.estimators import ESTIMATORS
from xcross.model.evaluate import metrics
from xcross.model.train import oof_predict

CALIBRATIONS = ("isotonic", "sigmoid")
METRICS_DIR = ROOT / "artifacts" / "reports" / "metrics"


def run() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    logger.info(f"Loaded {df.height} crosses.")

    rows: list[dict] = []
    for feature_set in FEATURE_SETS:
        for label in LABEL_COLS:
            X, y, groups, _ = make_xy(df, feature_set, label)
            for name, factory in ESTIMATORS.items():
                for calibration in CALIBRATIONS:
                    prob = oof_predict(factory, X, y, groups, calibration)
                    record = metrics(y, prob, player_ids)
                    rows.append({
                        "feature_set": feature_set, "label": label,
                        "estimator": name, "calibration": calibration, **record,
                    })
                    logger.info(
                        f"{feature_set}/{label}/{name}/{calibration}: "
                        f"auc={record['auc']:.3f} ece={record['ece']:.3f} "
                        f"stab={record['stability']:.2f} icc={record['icc']:.2f}"
                    )

    table = pl.DataFrame(rows)
    table.write_csv(METRICS_DIR / "comparison.csv")
    logger.info(f"Wrote comparison.csv ({table.height} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
