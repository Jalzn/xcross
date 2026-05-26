"""Permutation importance of the TabPFN on the two xCrossOT targets, so it can stand
alongside the other headlines' importance. Runs on the studio (needs GPU); calibration
is skipped (it is monotonic and does not change permutation importance over AUC) to keep
the cost manageable.

    XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda uv run python scripts/tabpfn_importance.py
"""

from __future__ import annotations

import sys

import polars as pl
from loguru import logger
from sklearn.inspection import permutation_importance

from xcross.config import ROOT
from xcross.model.dataset import load_features, make_xy
from xcross.model.estimators import ESTIMATORS

METRICS = ROOT / "artifacts" / "reports" / "metrics"
TARGETS = (("success", "success"), ("shot_in_window", "shot"))
N_REPEATS = 5
MAX_SAMPLES = 2000


def run() -> int:
    df = load_features()
    for label_col, label_clean in TARGETS:
        logger.info(f"[tabpfn-importance] xcrossot/{label_clean}: building model ...")
        X, y, _, names = make_xy(df, "xcrossot", label_col)
        model = ESTIMATORS["tabpfn"]().fit(X, y)
        logger.info(f"[tabpfn-importance] xcrossot/{label_clean}: running permutation ({N_REPEATS} reps × {len(names)} features × {MAX_SAMPLES} samples) ...")
        result = permutation_importance(
            model, X, y, scoring="roc_auc",
            n_repeats=N_REPEATS, max_samples=MAX_SAMPLES, random_state=0, n_jobs=1,
        )
        out = pl.DataFrame({
            "feature": names, "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort("importance", descending=True)
        out.write_csv(METRICS / f"importance_xcrossot_{label_clean}_tabpfn.csv")
        logger.info(f"[tabpfn-importance] wrote importance_xcrossot_{label_clean}_tabpfn.csv; top5: {out.head(5)['feature'].to_list()}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
