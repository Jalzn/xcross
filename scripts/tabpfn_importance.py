"""Permutation importance of the TabPFN on the two xCrossOT targets, so it can stand
alongside the other headlines' importance. Runs on the studio (needs GPU); calibration
is skipped (it is monotonic and does not change permutation importance over AUC) to keep
the cost manageable.

    XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda uv run python scripts/tabpfn_importance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402

from xcross.config import ROOT  # noqa: E402
from xcross.model.dataset import load_features, make_xy  # noqa: E402

METRICS = ROOT / "artifacts" / "reports" / "metrics"
TARGETS = (("success", "success"), ("shot_in_window", "shot"))
N_REPEATS = 3
MAX_SAMPLES = 500
N_ESTIMATORS = 1
TABPFN_REPO = "Prior-Labs/TabPFN-v2-clf"
TABPFN_CKPT = "tabpfn-v2-classifier.ckpt"


def _build_tabpfn():
    """Light-weight TabPFN for permutation importance: n_estimators=1 (no ensemble) and
    memory_saving_mode (avoids OOM on the L4)."""
    weights = hf_hub_download(TABPFN_REPO, TABPFN_CKPT)
    return TabPFNClassifier(
        model_path=weights, device="cuda", random_state=0,
        ignore_pretraining_limits=True, memory_saving_mode=True, n_estimators=N_ESTIMATORS,
    )


def run() -> int:
    df = load_features()
    for label_col, label_clean in TARGETS:
        logger.info(f"[tabpfn-importance] xcrossot/{label_clean}: building model (n_estimators={N_ESTIMATORS}) ...")
        X, y, _, names = make_xy(df, "xcrossot", label_col)
        model = _build_tabpfn().fit(X, y)
        logger.info(
            f"[tabpfn-importance] xcrossot/{label_clean}: running permutation "
            f"({N_REPEATS} reps × {len(names)} features × {MAX_SAMPLES} samples) ..."
        )
        result = permutation_importance(
            model, X, y, scoring="roc_auc",
            n_repeats=N_REPEATS, max_samples=MAX_SAMPLES, random_state=0, n_jobs=1,
        )
        out = pl.DataFrame({
            "feature": names, "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort("importance", descending=True)
        out.write_csv(METRICS / f"importance_xcrossot_{label_clean}_tabpfn.csv")
        logger.info(
            f"[tabpfn-importance] wrote importance_xcrossot_{label_clean}_tabpfn.csv; "
            f"top5: {out.head(5)['feature'].to_list()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(run())
