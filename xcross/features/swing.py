"""Cross type from the ball path (xCrossOT): cutback, and in/out-swing curl.

A cutback (pulled back from near the byline) and an in- vs out-swinger convert very differently;
both are read off the ball geometry. Curl is the signed perpendicular deviation of the flight from
the straight start->end chord — its sign distinguishes in- from out-swing."""

from __future__ import annotations

import numpy as np

from xcross.config import BOX_DEPTH_M


def _max_perp_deviation(traj: np.ndarray, sx: float, sy: float, ex: float, ey: float) -> float:
    if len(traj) < 3:
        return 0.0
    dx, dy = ex - sx, ey - sy
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return 0.0
    signed = ((traj[:, 1] - sx) * dy - (traj[:, 2] - sy) * dx) / length
    return float(signed[np.argmax(np.abs(signed))])


def swing_features(row: dict, traj: np.ndarray, half_length: float) -> dict:
    sx, sy = float(row["start_x"]), float(row["start_y"])
    ex, ey = float(row["end_x"]), float(row["end_y"])
    curl = _max_perp_deviation(traj, sx, sy, ex, ey)
    return {
        "swing_cutback": int(ex < sx and (half_length - sx) <= BOX_DEPTH_M),
        "swing_inout": curl,
        "swing_curl_magnitude": abs(curl),
    }
