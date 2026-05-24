"""Assemble all features for one cross into a single flat row.

Identity/label columns first, then xCross features (start_frame), then xCrossOT-only
features (destination + `*_in_zone` at end_frame). The model stage selects which subset
each model uses; the parquet carries everything.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from xcross.features.clearance import vertical_clearance
from xcross.features.counts import count_features, near_action_line_counts
from xcross.features.coverage import coverage_mismatch
from xcross.features.events import destination_features, event_features
from xcross.features.flight import ball_trajectory, flight_features
from xcross.features.frames import frame_state
from xcross.features.goalkeeper import goalkeeper_features
from xcross.features.grid import (
    around_mask,
    center_box_mask,
    first_post_mask,
    grid_centers,
    second_post_mask,
)
from xcross.features.marking import marking_tightness
from xcross.features.pockets import largest_empty_pocket
from xcross.features.pressure import crosser_pressure
from xcross.features.shape import defensive_shape
from xcross.features.spatial import spatial_features
from xcross.features.support import second_ball_support
from xcross.features.swing import swing_features
from xcross.features.temporal import defensive_collapse, zone_dynamics_delta

# Columns that identify the cross / carry labels — not model inputs.
ID_COLS = ("cross_id", "match_id", "league", "season", "crosser_player_id", "crosser_team_id")
LABEL_COLS = ("success", "shot_in_window")


def _defending_gk_ids(gk_ids: dict[int, set[int]] | None, crosser_team_id: int) -> set[int] | None:
    """The goalkeeper id(s) of the team that is *not* crossing (subs make this a set)."""
    if not gk_ids:
        return None
    defending = [pids for team, pids in gk_ids.items() if team != crosser_team_id]
    return set().union(*defending) if defending else None


def cross_features(
    row: dict,
    tracking_cross: pl.DataFrame,
    ball_cross: pl.DataFrame,
    league: str,
    season: str,
    half_length: float,
    half_width: float,
    fps: float,
    gk_ids: dict[int, set[int]] | None = None,
) -> dict:
    team = row["crosser_team_id"]
    defending_gk_ids = _defending_gk_ids(gk_ids, team)
    gx, gy = grid_centers(half_length, half_width)
    full = np.ones_like(gx, dtype=bool)

    start = frame_state(
        tracking_cross, ball_cross, row["start_frame"], team, fps,
        crosser_player_id=row["crosser_player_id"], gk_player_ids=defending_gk_ids,
    )
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
    out |= crosser_pressure(start)
    out |= marking_tightness(start, half_length)
    out |= largest_empty_pocket(start, half_length, half_width)
    out |= coverage_mismatch(start, half_length, half_width)
    out |= goalkeeper_features(start.gk_pos, start.gk_vel, start.ball_xy, half_length)
    out |= defensive_shape(start, half_length)

    # xCrossOT-only: destination geometry, entropy/pitch-control at the arrival point,
    # and players near the cross line. The cross window never emits end_frame itself
    # (it closes before emitting), so use the last emitted frame — which is where
    # end_x/end_y come from.
    last_frame = int(tracking_cross["frame_num"].max())
    end = frame_state(
        tracking_cross, ball_cross, last_frame, team, fps, gk_player_ids=defending_gk_ids
    )
    ball_end = (float(row["end_x"]), float(row["end_y"]))
    zone_region = {"in_zone": around_mask(gx, gy, ball_end)}
    end_zone = spatial_features(end, half_length, half_width, zone_region, with_grad=False)
    out |= destination_features(row, half_length)
    out |= end_zone
    out |= near_action_line_counts(start, ball_start, ball_end)

    # xCrossOT temporal deltas (start -> arrival; cheap, two frames only).
    start_zone = spatial_features(start, half_length, half_width, zone_region, with_grad=False)
    out |= defensive_collapse(start, end, half_length)
    out |= zone_dynamics_delta(start_zone, end_zone)

    # xCrossOT ball-flight block (the ball z trajectory, built once) + drop-zone support.
    traj = ball_trajectory(ball_cross, int(row["start_frame"]), last_frame)
    out |= flight_features(traj, fps)
    out |= vertical_clearance(start.defense_pos, start.gk_pos, traj)
    out |= swing_features(row, traj, half_length)
    out |= second_ball_support(start, ball_end)
    return out
