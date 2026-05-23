"""CLI: serialize the headline calibrated model per target for inference.

Reads the (estimator, calibration) the report published in model_metrics.csv, refits the
calibrated model on ALL crosses, and dumps it to artifacts/models/ so the serialized model
reproduces the published numbers. A metadata.json records the feature order and library
versions, so the models load without the licensed tracking data.

    uv run python -m xcross.model.export
"""

from __future__ import annotations

import json
import sys

import catboost
import joblib
import polars as pl
import sklearn
import xgboost
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV

from xcross.config import ROOT
from xcross.model.dataset import load_features, make_xy
from xcross.model.estimators import ESTIMATORS

MODELS_DIR = ROOT / "artifacts" / "models"
MODEL_METRICS = ROOT / "artifacts" / "reports" / "metrics" / "model_metrics.csv"
CALIBRATION_CV = 3  # the inner calibration CV the OOF evaluation used (train.py)
LABEL_COLUMN = {"success": "success", "shot": "shot_in_window"}


def published_choices() -> list[dict]:
    """(feature_set, label, estimator, calibration) exactly as model_metrics.csv reports."""
    metrics = pl.read_csv(MODEL_METRICS)
    choices = []
    for row in metrics.iter_rows(named=True):
        feature_set, label = row["model"].split("/")
        choices.append({
            "feature_set": feature_set,
            "label": label,
            "estimator": row["estimator"],
            "calibration": row["calibration"],
        })
    return choices


def run() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    logger.info(f"Loaded {df.height} crosses.")

    metadata: dict = {
        "n_crosses": df.height,
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
        },
        "models": {},
    }

    for choice in published_choices():
        feature_set, label = choice["feature_set"], choice["label"]
        X, y, _, feature_names = make_xy(df, feature_set, LABEL_COLUMN[label])
        model = CalibratedClassifierCV(
            ESTIMATORS[choice["estimator"]](), method=choice["calibration"], cv=CALIBRATION_CV
        )
        model.fit(X, y)

        name = f"{feature_set}_{label}"
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        metadata["models"][name] = {
            "estimator": choice["estimator"],
            "calibration": choice["calibration"],
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "positive_rate": float(y.mean()),
        }
        logger.info(
            f"Wrote {name}.joblib ({choice['estimator']}/{choice['calibration']}, "
            f"{len(feature_names)} features)."
        )

    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote metadata.json ({len(metadata['models'])} models).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
