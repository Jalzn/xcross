"""CLI: block ablation — what each new feature block adds over the previous version.

For each (feature_set, label) it scores the *selected* model (from comparison.csv) on three
column sets: the FULL set, the BASE set (previous version = every new block removed) and each
leave-one-block-out set. It writes per-variant metrics plus a delta table where the `ALL_NEW`
row is the total new-vs-previous gain and each block row is that block's marginal contribution
(full − drop-one). Only blocks actually present in the features parquet are ablated, so this
adapts to whichever phase has been built.

    uv run python -m xcross.model.ablation
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from loguru import logger

from xcross.config import ROOT
from xcross.model.dataset import NEW_BLOCK_PREFIXES, NEW_OT_BLOCKS, load_features, make_xy, match_dates
from xcross.model.estimators import ESTIMATORS
from xcross.model.evaluate import metrics, temporal_split_stability
from xcross.model.selection import load_comparison, select_best
from xcross.model.train import oof_predict

METRICS_DIR = ROOT / "artifacts" / "reports" / "metrics"
LABELS = ("success", "shot")
LABEL_COLUMN = {"success": "success", "shot": "shot_in_window"}
FEATURE_SETS = ("xcross", "xcrossot")
DELTA_METRICS = (
    "auc", "brier_skill", "ece", "stability_random", "stability_temporal", "icc", "player_discrimination",
)


def _present_blocks(df: pl.DataFrame) -> set[str]:
    """New blocks that actually have columns in the parquet (so we don't ablate empty ones)."""
    return {b for b, prefix in NEW_BLOCK_PREFIXES.items() if any(c.startswith(prefix) for c in df.columns)}


def _variant_metrics(y: np.ndarray, prob: np.ndarray, player_ids: np.ndarray, order_key: np.ndarray) -> dict:
    m = metrics(y, prob, player_ids)
    return {
        "auc": m["auc"], "auc_pr": m["auc_pr"], "brier_skill": m["brier_skill"], "ece": m["ece"],
        "stability_random": m["stability"], "stability_temporal": temporal_split_stability(player_ids, prob, order_key),
        "icc": m["icc"], "player_discrimination": m["player_discrimination"],
    }


def _oof_metrics(df, spec, label_col, est, cal, player_ids, order_key) -> tuple[dict, int]:
    X, y, groups, _ = make_xy(df, spec, label_col)
    prob = oof_predict(ESTIMATORS[est], X, y, groups, cal)
    return _variant_metrics(y, prob, player_ids, order_key), X.shape[1]


def _delta(full: dict, other: dict, feature_set: str, block: str) -> dict:
    return {"feature_set": feature_set, "block": block, **{f"d_{k}": full[k] - other[k] for k in DELTA_METRICS}}


def run() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    player_ids = df["crosser_player_id"].to_numpy()
    dates = match_dates()
    order_key = np.array([dates.get(m, m) for m in df["match_id"].to_list()])
    comparison = load_comparison()
    present = _present_blocks(df)
    logger.info(f"Loaded {df.height} crosses. New blocks present: {sorted(present)}")

    for label in LABELS:
        label_col = LABEL_COLUMN[label]
        rows: list[dict] = []
        deltas: list[dict] = []
        for fs in FEATURE_SETS:
            est, cal = select_best(comparison, fs, label_col, eligible=set(ESTIMATORS)).values()
            applicable = sorted(present if fs == "xcrossot" else present - NEW_OT_BLOCKS)

            full_m, full_n = _oof_metrics(df, fs, label_col, est, cal, player_ids, order_key)
            base_m, base_n = _oof_metrics(df, f"{fs}__base", label_col, est, cal, player_ids, order_key)
            rows.append({"feature_set": fs, "variant": "full", "block": "-", "n_features": full_n,
                         "estimator": est, "calibration": cal, **full_m})
            rows.append({"feature_set": fs, "variant": "base", "block": "-", "n_features": base_n,
                         "estimator": est, "calibration": cal, **base_m})
            deltas.append(_delta(full_m, base_m, fs, "ALL_NEW"))
            logger.info(f"{fs}/{label}: full-vs-base d_auc={full_m['auc'] - base_m['auc']:+.3f} "
                        f"d_stab={full_m['stability_random'] - base_m['stability_random']:+.3f}")

            for block in applicable:
                drop_m, drop_n = _oof_metrics(df, f"{fs}__drop-{block}", label_col, est, cal, player_ids, order_key)
                rows.append({"feature_set": fs, "variant": "drop-one", "block": block, "n_features": drop_n,
                             "estimator": est, "calibration": cal, **drop_m})
                deltas.append(_delta(full_m, drop_m, fs, block))

        pl.DataFrame(rows).write_csv(METRICS_DIR / f"ablation_blocks_{label}.csv")
        pl.DataFrame(deltas).write_csv(METRICS_DIR / f"ablation_blocks_delta_{label}.csv")
        logger.info(f"Wrote ablation_blocks_{label}.csv ({len(rows)} rows) + delta.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
