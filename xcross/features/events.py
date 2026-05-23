"""Event-geometry features from the cross row (ball positions; pure functions)."""

from __future__ import annotations

import numpy as np


def _cross_region(start_x: float, start_y: float, half_length: float) -> str:
    """Coarse categorical origin region (depth to goal line × flank width)."""
    depth = half_length - start_x
    if depth <= 11.0:
        band = "byline"
    elif depth <= 22.0:
        band = "deep"
    else:
        band = "mid"
    flank = "touchline" if abs(start_y) >= 27.0 else "halfspace"
    return f"{band}_{flank}"


def event_features(row: dict, half_length: float) -> dict:
    """xCross geometry: ball origin position, distance to goal, origin region."""
    sx, sy = float(row["start_x"]), float(row["start_y"])
    return {
        "start_x": sx,
        "start_y": sy,
        "distance_start_from_goal": float(np.hypot(half_length - sx, sy)),
        "cross_region": _cross_region(sx, sy, half_length),
    }


def destination_features(row: dict, half_length: float) -> dict:
    """xCrossOT geometry: ball destination, distance to end line, cross angle."""
    sx, sy = float(row["start_x"]), float(row["start_y"])
    ex, ey = float(row["end_x"]), float(row["end_y"])
    return {
        "end_x": ex,
        "end_y": ey,
        "distance_from_end_line": half_length - ex,
        "polar_angle_cross": float(np.arctan2(ey - sy, ex - sx)),
    }
