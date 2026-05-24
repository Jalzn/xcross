"""Goalkeeper geometry at the cross (start-frame, xCross).

The keeper is the decisive defender for a cross, yet elsewhere he is just one point in the
defensive blob. Here he is singled out: how far off his line, how far from the ball, and how
much he is shifting sideways."""

from __future__ import annotations

import numpy as np

_NO_GK = 100.0  # finite sentinel (> pitch diagonal) when the defending keeper is off the frame


def goalkeeper_features(
    gk_pos: tuple[float, float] | None,
    gk_vel: tuple[float, float] | None,
    ball_xy: tuple[float, float],
    half_length: float,
) -> dict:
    if gk_pos is None:
        return {"gk_distance_off_line": 0.0, "gk_ball_distance": _NO_GK,
                "gk_lateral_speed": 0.0, "gk_present": 0}
    return {
        "gk_distance_off_line": float(half_length - gk_pos[0]),
        "gk_ball_distance": float(np.hypot(gk_pos[0] - ball_xy[0], gk_pos[1] - ball_xy[1])),
        "gk_lateral_speed": float(abs(gk_vel[1])) if gk_vel is not None else 0.0,
        "gk_present": 1,
    }
