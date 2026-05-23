"""Assemble all features for one cross into a single flat row.

Identity/label columns first, then xCross features (start_frame), then xCrossOT-only
features (destination + `*_in_zone` at end_frame). The model stage selects which subset
each model uses; the parquet carries everything.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from xcross.features.counts import count_features, near_action_line_counts
from xcross.features.events import destination_features, event_features
from xcross.features.frames import frame_state
from xcross.features.grid import (
    around_mask,
    center_box_mask,
    first_post_mask,
    grid_centers,
    second_post_mask,
)
from xcross.features.spatial import spatial_features

# Columns that identify the cross / carry labels — not model inputs.
ID_COLS = ("cross_id", "match_id", "league", "season", "crosser_player_id", "crosser_team_id")
LABEL_COLS = ("success", "shot_in_window")


def cross_features(
    row: dict,
    tracking_cross: pl.DataFrame,
    ball_cross: pl.DataFrame,
    league: str,
    season: str,
    half_length: float,
    half_width: float,
    fps: float,
) -> dict:
    team = row["crosser_team_id"]
    gx, gy = grid_centers(half_length, half_width)
    full = np.ones_like(gx, dtype=bool)

    start = frame_state(tracking_cross, ball_cross, row["start_frame"], team, fps)
    ball_start = (float(row["start_x"]), float(row["start_y"]))
    start_regions = {
        "around": around_mask(gx, gy, ball_start),
        "in_first_post": first_post_mask(gx, gy, half_length),
        "in_second_post": second_post_mask(gx, gy, half_length),
        "in_center_box": center_box_mask(gx, gy, half_length),
        "sum": full,
    }

    out: dict = {
        "cross_id": row["cross_id"],
        "match_id": row["match_id"],
        "league": league,
        "season": season,
        "crosser_player_id": row["crosser_player_id"],
        "crosser_team_id": team,
        "success": row["success"],
        "shot_in_window": row["shot_in_window"],
    }
    out |= event_features(row, half_length)
    out |= count_features(start, half_length)
    out |= spatial_features(start, half_length, half_width, start_regions, with_grad=True)

    # xCrossOT-only: destination geometry, entropy/pitch-control at the arrival point,
    # and players near the cross line. The cross window never emits end_frame itself
    # (it closes before emitting), so use the last emitted frame — which is where
    # end_x/end_y come from.
    last_frame = int(tracking_cross["frame_num"].max())
    end = frame_state(tracking_cross, ball_cross, last_frame, team, fps)
    ball_end = (float(row["end_x"]), float(row["end_y"]))
    out |= destination_features(row, half_length)
    out |= spatial_features(end, half_length, half_width, {"in_zone": around_mask(gx, gy, ball_end)}, with_grad=False)
    out |= near_action_line_counts(start, ball_start, ball_end)
    return out
