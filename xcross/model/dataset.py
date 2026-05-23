"""Load the feature parquets and build (X, y, groups) for a given model variant.

`xcross` uses start-frame features only; `xcrossot` adds the destination features.
`cross_region` (categorical) is one-hot encoded so every tabular estimator can use it.
"""

from __future__ import annotations

import glob

import numpy as np
import polars as pl

from xcross.config import FEATURES_ROOT

ID_COLS = ["cross_id", "match_id", "league", "season", "crosser_player_id", "crosser_team_id"]
LABEL_COLS = ["success", "shot_in_window"]

# Features that depend on the cross destination (end_frame) — xCrossOT only.
OT_ONLY = [
    "end_x", "end_y", "distance_from_end_line", "polar_angle_cross",
    "entropy_attack_in_zone", "entropy_defense_in_zone",
    "entropy_general_in_zone", "entropy_diff_in_zone", "pitch_control_in_zone",
    "attackers_near_action_line", "defenders_near_action_line",
]
FEATURE_SETS = ("xcross", "xcrossot")
ENTROPY_PREFIX = "entropy_"  # the spatial-entropy block; dropped by the "_noent" ablation sets


def load_features() -> pl.DataFrame:
    """Concatenate every non-empty per-match features parquet."""
    files = sorted(glob.glob(str(FEATURES_ROOT / "*" / "*" / "*" / "features.parquet")))
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat([f for f in frames if f.width > 0])


def _feature_columns(df: pl.DataFrame, feature_set: str) -> list[str]:
    """`xcross`/`xcrossot` as documented; the `_noent` suffix drops the entropy block
    (the ablation that isolates the contribution of the spatial-entropy features)."""
    drop_entropy = feature_set.endswith("_noent")
    base = feature_set.removesuffix("_noent")
    excluded = set(ID_COLS + LABEL_COLS)
    columns = [c for c in df.columns if c not in excluded]
    if base == "xcross":
        columns = [c for c in columns if c not in OT_ONLY]
    if drop_entropy:
        columns = [c for c in columns if not c.startswith(ENTROPY_PREFIX)]
    return columns


def make_xy(
    df: pl.DataFrame, feature_set: str, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, groups=match_id, feature_names) for one feature set and label."""
    columns = _feature_columns(df, feature_set)
    features = df.select(columns)
    if "cross_region" in columns:
        features = features.to_dummies(columns=["cross_region"])
    return (
        features.to_numpy(),
        df[label].cast(pl.Int8).to_numpy(),
        df["match_id"].to_numpy(),
        features.columns,
    )
