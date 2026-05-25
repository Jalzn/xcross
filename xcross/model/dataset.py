"""Load the feature parquets and build (X, y, groups) for a given model variant.

`xcross` uses start-frame features only; `xcrossot` adds the destination features.
`cross_region` (categorical) is one-hot encoded so every tabular estimator can use it.
"""

from __future__ import annotations

import glob

import numpy as np
import polars as pl

from xcross.config import FEATURES_ROOT, PROCESSED_ROOT

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

# Feature blocks added on top of the original feature set, each tagged by a column prefix.
# The block ablation (see _feature_columns) drops a block by removing every column with its
# prefix. NEW_OT_BLOCKS are destination-dependent, so they are excluded from `xcross` too.
NEW_BLOCK_PREFIXES = {
    "pressure": "pressure_", "marking": "marking_",
    "pocket": "pocket_", "coverage": "coverage_", "gk": "gk_", "shape": "shape_",
    "support": "support_", "flight": "flight_",
    "clearance": "clearance_", "swing": "swing_", "temporal": "temporal_",
}
NEW_OT_BLOCKS = {"support", "flight", "clearance", "swing", "temporal"}
BLOCK_SEP = "__"  # separates the base set from block-drop suffixes (avoids clashing with "_noent")


def load_features() -> pl.DataFrame:
    """Concatenate every non-empty per-match features parquet."""
    files = sorted(glob.glob(str(FEATURES_ROOT / "*" / "*" / "*" / "features.parquet")))
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat([f for f in frames if f.width > 0])


def match_dates() -> dict[str, str]:
    """match_id -> ISO date (sorts chronologically as a string) for the temporal split."""
    files = glob.glob(str(PROCESSED_ROOT / "*" / "*" / "*" / "meta.parquet"))
    meta = pl.concat([pl.read_parquet(f) for f in files]).unique("match_id")
    return dict(zip(meta["match_id"].to_list(), meta["date"].to_list(), strict=True))


def _blocks_to_drop(suffixes: list[str]) -> set[str]:
    """`__base` drops every new block (= the previous version); `__drop-<block>` drops one."""
    drop: set[str] = set()
    for suffix in suffixes:
        if suffix == "base":
            drop |= set(NEW_BLOCK_PREFIXES)
        elif suffix.startswith("drop-"):
            drop.add(suffix.removeprefix("drop-"))
    return drop


def _feature_columns(df: pl.DataFrame, feature_set: str) -> list[str]:
    """Select columns for a feature-set spec: `<base>[_noent][__base|__drop-<block> ...]`.

    `xcross` keeps start-frame features only; `xcrossot` adds the destination block. `_noent`
    drops the entropy block; `__base` drops every new block (the previous version) and
    `__drop-<block>` drops a single one — these drive the new-vs-previous block ablation.
    """
    head, *suffixes = feature_set.split(BLOCK_SEP)
    drop_entropy = head.endswith("_noent")
    base = head.removesuffix("_noent")
    drop_blocks = _blocks_to_drop(suffixes)
    drop_prefixes = [NEW_BLOCK_PREFIXES[b] for b in drop_blocks]

    excluded = set(ID_COLS + LABEL_COLS)
    columns = [c for c in df.columns if c not in excluded]
    if base == "xcross":
        ot_prefixes = tuple(NEW_BLOCK_PREFIXES[b] for b in NEW_OT_BLOCKS)
        columns = [c for c in columns if c not in OT_ONLY and not c.startswith(ot_prefixes)]
    if drop_entropy:
        columns = [c for c in columns if not c.startswith(ENTROPY_PREFIX)]
    if drop_prefixes:
        columns = [c for c in columns if not c.startswith(tuple(drop_prefixes))]
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
