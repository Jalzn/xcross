"""Bridge from the per-cross tracking/ball parquets to numpy positions + velocities.

A cross window has no frames before start_frame, so velocity is a forward difference
(falls back to backward at the end of the window). Velocity is returned in m/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from xcross.config import VELOCITY_DELTA_FRAMES


@dataclass(frozen=True, slots=True)
class FrameState:
    attack_pos: np.ndarray   # (Na, 2) crossing team
    attack_vel: np.ndarray   # (Na, 2) m/s
    defense_pos: np.ndarray  # (Nd, 2)
    defense_vel: np.ndarray  # (Nd, 2) m/s
    ball_xy: tuple[float, float]


def _split_by_team(frame: pl.DataFrame, crosser_team_id: int) -> tuple[np.ndarray, ...]:
    attack = frame.filter(pl.col("team_id") == crosser_team_id)
    defense = frame.filter(pl.col("team_id") != crosser_team_id)
    return (
        attack.select("x", "y").to_numpy(),
        attack.select("vx", "vy").to_numpy(),
        defense.select("x", "y").to_numpy(),
        defense.select("vx", "vy").to_numpy(),
    )


def frame_state(
    cross_tracking: pl.DataFrame,
    cross_ball: pl.DataFrame,
    frame_num: int,
    crosser_team_id: int,
    fps: float,
    k: int = VELOCITY_DELTA_FRAMES,
) -> FrameState:
    """Positions + velocities of both teams at `frame_num` for a single cross."""
    available = set(cross_tracking["frame_num"].unique().to_list())
    reference = frame_num + k if (frame_num + k) in available else frame_num - k
    dt_s = (reference - frame_num) / fps

    current = cross_tracking.filter(pl.col("frame_num") == frame_num)
    later = (
        cross_tracking.filter(pl.col("frame_num") == reference)
        .select("player_id", pl.col("x").alias("x_ref"), pl.col("y").alias("y_ref"))
    )
    frame = current.join(later, on="player_id", how="left").with_columns(
        ((pl.col("x_ref") - pl.col("x")) / dt_s).fill_null(0.0).alias("vx"),
        ((pl.col("y_ref") - pl.col("y")) / dt_s).fill_null(0.0).alias("vy"),
    )

    attack_pos, attack_vel, defense_pos, defense_vel = _split_by_team(frame, crosser_team_id)
    ball = cross_ball.filter(pl.col("frame_num") == frame_num)
    ball_xy = (float(ball["x"][0]), float(ball["y"][0])) if ball.height else (0.0, 0.0)
    return FrameState(attack_pos, attack_vel, defense_pos, defense_vel, ball_xy)
