"""Out-of-fold metrics for TabPFN, in an isolated process.

TabPFN (PyTorch) and xgboost/lightgbm each ship an OpenMP runtime; on macOS, exercising
both in one process segfaults non-deterministically. So TabPFN scores its part of the
comparison here, in a process that never imports the gradient-boosting libraries, and
compare.py launches this as a subprocess and merges comparison_tabpfn.csv.

Weights are pulled once from the public HF repo (no Prior Labs token needed). TabPFN is
O(n²) in the context size, so CPU is very slow (~15 min/combo); on a GPU host set
`XCROSS_TABPFN_DEVICE=cuda` and it runs in minutes.

    uv run python -m xcross.model.tabpfn_oof
    XCROSS_TABPFN_DEVICE=cuda uv run python -m xcross.model.tabpfn_oof   # GPU host
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from loguru import logger
from sklearn.base import ClassifierMixin
from tabpfn import TabPFNClassifier

from xcross.config import ROOT
from xcross.model.dataset import FEATURE_SETS, LABEL_COLS, load_features, make_xy, match_dates
from xcross.model.evaluate import metrics
from xcross.model.train import oof_predict

SEED = 0
CALIBRATIONS = ("isotonic", "sigmoid")
TABPFN_REPO = "Prior-Labs/TabPFN-v2-clf"
TABPFN_CKPT = "tabpfn-v2-classifier.ckpt"
DEVICE = os.environ.get("XCROSS_TABPFN_DEVICE", "cpu")
OUTPUT = ROOT / "artifacts" / "reports" / "metrics" / "comparison_tabpfn.csv"

_weights: str | None = None


def _tabpfn() -> ClassifierMixin:
    global _weights
    if _weights is None:
        _weights = hf_hub_download(TABPFN_REPO, TABPFN_CKPT)
    return TabPFNClassifier(
        model_path=_weights, device=DEVICE, random_state=SEED, ignore_pretraining_limits=True,
    )


def run() -> int:
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    dates = match_dates()
    order_key = np.array([dates.get(m, m) for m in df["match_id"].to_list()])
    logger.info(f"[tabpfn] Loaded {df.height} crosses. device={DEVICE}.")

    rows: list[dict] = []
    for feature_set in FEATURE_SETS:
        for label in LABEL_COLS:
            X, y, groups, _ = make_xy(df, feature_set, label)
            for calibration in CALIBRATIONS:
                prob = oof_predict(_tabpfn, X, y, groups, calibration)
                record = metrics(y, prob, player_ids, order_key)
                rows.append({
                    "feature_set": feature_set, "label": label,
                    "estimator": "tabpfn", "calibration": calibration, **record,
                })
                logger.info(
                    f"[tabpfn] {feature_set}/{label}/{calibration}: "
                    f"auc={record['auc']:.3f} stab_t={record['stability_temporal']:.2f}"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUT)
    logger.info(f"[tabpfn] Wrote {OUTPUT.name} ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
