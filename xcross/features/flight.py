"""Ball-flight features from the 3D trajectory (xCrossOT). The ball's z is otherwise unused.

A cross is a projectile: its arc (apex, launch/descent angle), how long it hangs, how lofted vs
driven it is, its 3D pace and whether it bounced all distinguish deliveries that look identical
in 2D. The cleaned trajectory built here is reused by the clearance/swing/contact blocks."""

from __future__ import annotations

import numpy as np
import polars as pl

from xcross.config import FLIGHT_GROUND_M

_FLIGHT_KEYS = (
    "flight_apex_height", "flight_apex_timing", "flight_launch_angle", "flight_descent_angle",
    "flight_hang_time", "flight_loftiness", "flight_pace_3d", "flight_bounce_count",
)


def ball_trajectory(ball_cross: pl.DataFrame, start_frame: int, last_frame: int) -> np.ndarray:
    """(M, 4) [frame_num, x, y, z>=0] within the cross window, sorted by frame."""
    traj = (
        ball_cross
        .filter((pl.col("frame_num") >= start_frame) & (pl.col("frame_num") <= last_frame))
        .sort("frame_num")
        .select("frame_num", "x", "y", "z")
        .to_numpy()
    )
    if len(traj):
        traj[:, 3] = np.clip(traj[:, 3], 0.0, None)  # z dips slightly negative from smoothing
    return traj


def _segment_angle(segment: np.ndarray) -> float:
    """Angle (rad) of the ball path over a short segment: +up, 0 flat, -down."""
    if len(segment) < 2:
        return 0.0
    rise = segment[-1, 3] - segment[0, 3]
    run = float(np.hypot(segment[-1, 1] - segment[0, 1], segment[-1, 2] - segment[0, 2]))
    return float(np.arctan2(rise, run + 1e-9))


def _bounce_count(z: np.ndarray) -> int:
    """Local minima of z near the ground — each is the ball bouncing back up."""
    if len(z) < 3:
        return 0
    interior = z[1:-1]
    is_dip = (interior < z[:-2]) & (interior < z[2:]) & (interior < FLIGHT_GROUND_M)
    return int(is_dip.sum())


def flight_features(traj: np.ndarray, fps: float) -> dict:
    if len(traj) < 2:
        return dict.fromkeys(_FLIGHT_KEYS, 0.0)
    z = traj[:, 3]
    step_horizontal = np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2]))
    speeds_3d = np.hypot(step_horizontal, np.abs(np.diff(z))) * fps
    return {
        "flight_apex_height": float(z.max()),
        "flight_apex_timing": float(np.argmax(z)) / (len(z) - 1),
        "flight_launch_angle": _segment_angle(traj[:4]),
        "flight_descent_angle": _segment_angle(traj[-4:]),
        "flight_hang_time": float((z > FLIGHT_GROUND_M).sum()) / fps,
        "flight_loftiness": float(z.max()) / (float(step_horizontal.sum()) + 1e-6),
        "flight_pace_3d": float(np.median(speeds_3d)),
        "flight_bounce_count": float(_bounce_count(z)),
    }
