"""Registry of tabular estimators. All are sklearn-compatible (fit/predict_proba),
so comparison is a loop, not an abstraction. Kept lightly regularised to preserve
probability spread (the ranking objective)."""

from __future__ import annotations

from collections.abc import Callable

from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
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


ESTIMATORS: dict[str, Callable[[], ClassifierMixin]] = {
    "xgboost": _xgboost,
    "adaboost": _adaboost,
    "catboost": _catboost,
}
