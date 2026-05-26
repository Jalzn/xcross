"""Out-of-fold calibrated probabilities.

Each cross gets a probability from a model that never saw it (StratifiedGroupKFold by
match), so the OOF probabilities are usable for ranking and unbiased evaluation. Inside
each training fold, CalibratedClassifierCV calibrates (isotonic/sigmoid) via its own CV.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEED = 0


def _free_gpu() -> None:
    """Release the CUDA cache between folds; TabPFN otherwise accumulates VRAM until OOM.
    No-op when torch is absent (the gradient-boosting registry never imports it)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oof_predict(
    estimator_factory: Callable[[], ClassifierMixin],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str,
) -> np.ndarray:
    """5-fold out-of-fold calibrated P(label=1) for every row."""
    oof = np.zeros(len(y), dtype=float)
    folds = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_idx, val_idx in folds.split(X, y, groups):
        model = CalibratedClassifierCV(estimator_factory(), method=method, cv=3)
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        del model
        _free_gpu()
    return oof
