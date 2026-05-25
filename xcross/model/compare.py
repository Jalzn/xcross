"""CLI: train the matrix {xCross,xCrossOT} × {success,shot} × estimators × calibrations,
score each on OOF probabilities, and write the comparison table (raw data only).

TabPFN is scored in an isolated subprocess (see tabpfn_oof.py) and appended; pass
`--no-tabpfn` to skip it. report.py reads this table to pick the best model per target
(see selection.py) — it is not hardcoded. Figures are produced by figures.py / report.py.

    uv run python -m xcross.model.compare
"""

from __future__ import annotations

import subprocess
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
TABPFN_CSV = METRICS_DIR / "comparison_tabpfn.csv"


def _append_tabpfn(table: pl.DataFrame) -> pl.DataFrame:
    """Run TabPFN in its own process (OpenMP clash with xgboost/lightgbm) and append it."""
    logger.info("Running TabPFN in an isolated subprocess ...")
    result = subprocess.run([sys.executable, "-m", "xcross.model.tabpfn_oof"])
    if result.returncode != 0 or not TABPFN_CSV.exists():
        logger.warning("TabPFN subprocess failed; comparison will omit it.")
        return table
    return pl.concat([table, pl.read_csv(TABPFN_CSV)], how="diagonal_relaxed")


def run() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    dates = match_dates()
    order_key = np.array([dates.get(m, m) for m in df["match_id"].to_list()])
    logger.info(f"Loaded {df.height} crosses.")

    rows: list[dict] = []
    for feature_set in FEATURE_SETS:
        for label in LABEL_COLS:
            X, y, groups, _ = make_xy(df, feature_set, label)
            for name, factory in ESTIMATORS.items():
                for calibration in CALIBRATIONS:
                    prob = oof_predict(factory, X, y, groups, calibration)
                    record = metrics(y, prob, player_ids, order_key)
                    rows.append({
                        "feature_set": feature_set, "label": label,
                        "estimator": name, "calibration": calibration, **record,
                    })
                    logger.info(
                        f"{feature_set}/{label}/{name}/{calibration}: "
                        f"auc={record['auc']:.3f} ece={record['ece']:.3f} "
                        f"stab={record['stability']:.2f} stab_t={record['stability_temporal']:.2f} "
                        f"icc={record['icc']:.2f}"
                    )

    table = pl.DataFrame(rows)
    if "--no-tabpfn" not in sys.argv:
        table = _append_tabpfn(table)
    table.write_csv(METRICS_DIR / "comparison.csv")
    logger.info(f"Wrote comparison.csv ({table.height} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
