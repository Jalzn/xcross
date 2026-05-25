"""Registry of tabular estimators. All are sklearn-compatible (fit/predict_proba),
so comparison is a loop, not an abstraction. Kept lightly regularised to preserve
probability spread (the ranking objective). Linear models that need scaled inputs carry
their own preprocessing in a pipeline, so the training loop stays uniform.

TabPFN joins the registry only when XCROSS_TABPFN is set (a GPU host, typically the cloud):
its PyTorch OpenMP runtime segfaults non-deterministically alongside xgboost/lightgbm on macOS,
so locally it stays out (torch is never imported). On Linux/GPU it is a first-class estimator."""

from __future__ import annotations

import os
from collections.abc import Callable

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 0


def _xgboost() -> ClassifierMixin:
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=-1, random_state=SEED,
    )


def _adaboost() -> ClassifierMixin:
    return AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED)


def _catboost() -> ClassifierMixin:
    return CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, verbose=False, random_seed=SEED,
    )


def _lightgbm() -> ClassifierMixin:
    return LGBMClassifier(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1,
    )


def _histgb() -> ClassifierMixin:
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05, random_state=SEED,
    )


def _random_forest() -> ClassifierMixin:
    """Pure bagging — a bet on the ranking-stability axis rather than discrimination."""
    return RandomForestClassifier(
        n_estimators=400, max_depth=8, n_jobs=-1, random_state=SEED,
    )


def _logreg() -> ClassifierMixin:
    """Linear floor; scaling is required, so it travels with the estimator."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))


TABPFN_REPO = "Prior-Labs/TabPFN-v2-clf"
TABPFN_CKPT = "tabpfn-v2-classifier.ckpt"
_tabpfn_weights: str | None = None


def _tabpfn() -> ClassifierMixin:
    """Pretrained tabular transformer; torch is imported lazily (only when this runs), weights
    pulled once from the public HF repo (no Prior Labs token). Device via XCROSS_TABPFN_DEVICE."""
    global _tabpfn_weights
    from huggingface_hub import hf_hub_download
    from tabpfn import TabPFNClassifier

    if _tabpfn_weights is None:
        _tabpfn_weights = hf_hub_download(TABPFN_REPO, TABPFN_CKPT)
    return TabPFNClassifier(
        model_path=_tabpfn_weights, device=os.environ.get("XCROSS_TABPFN_DEVICE", "cpu"),
        random_state=SEED, ignore_pretraining_limits=True,
    )


ESTIMATORS: dict[str, Callable[[], ClassifierMixin]] = {
    "xgboost": _xgboost,
    "adaboost": _adaboost,
    "catboost": _catboost,
    "lightgbm": _lightgbm,
    "histgb": _histgb,
    "random_forest": _random_forest,
    "logreg": _logreg,
}

if os.environ.get("XCROSS_TABPFN"):
    ESTIMATORS["tabpfn"] = _tabpfn
