"""Grid over the attacking half and the region masks used to summarise the maps.

Coordinates are already standardised (crosser attacks +x, cross comes from +y), so
the attacking half is x in [0, half_length] and the goal sits at (half_length, 0).
Every mask returns a boolean array shaped like the grid (GRID_NX, GRID_NY).
"""

from __future__ import annotations

import numpy as np

from xcross.config import (
    AROUND_RADIUS_M,
    BOX_DEPTH_M,
    BOX_HALF_WIDTH_M,
    CENTER_BOX_HALF_WIDTH_M,
    GRID_NX,
    GRID_NY,
    ZONE_DEPTH_M,
)


def grid_centers(half_length: float, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of the attacking-half grid, shaped (GRID_NX, GRID_NY)."""
    xs = np.linspace(0.0, half_length, GRID_NX)
    ys = np.linspace(-half_width, half_width, GRID_NY)
    return np.meshgrid(xs, ys, indexing="ij")


def box_mask(gx: np.ndarray, gy: np.ndarray, half_length: float) -> np.ndarray:
    return (gx >= half_length - BOX_DEPTH_M) & (np.abs(gy) <= BOX_HALF_WIDTH_M)


def first_post_mask(gx: np.ndarray, gy: np.ndarray, half_length: float) -> np.ndarray:
    """Near-post half of the box (+y, the side the cross comes from)."""
    return box_mask(gx, gy, half_length) & (gy >= 0)


def second_post_mask(gx: np.ndarray, gy: np.ndarray, half_length: float) -> np.ndarray:
    """Far-post half of the box (-y)."""
    return box_mask(gx, gy, half_length) & (gy < 0)


def center_box_mask(gx: np.ndarray, gy: np.ndarray, half_length: float) -> np.ndarray:
    """Central strip of the box (goal-area width)."""
    return (gx >= half_length - BOX_DEPTH_M) & (np.abs(gy) <= CENTER_BOX_HALF_WIDTH_M)


def zone_mask(gx: np.ndarray, gy: np.ndarray, half_length: float) -> np.ndarray:
    """Tactical zone in front of the box (box edge + arc), box width."""
    return (gx >= half_length - ZONE_DEPTH_M) & (np.abs(gy) <= BOX_HALF_WIDTH_M)


def around_mask(gx: np.ndarray, gy: np.ndarray, ball_xy: tuple[float, float]) -> np.ndarray:
    """Circle of radius AROUND_RADIUS_M around the ball."""
    bx, by = ball_xy
    return (gx - bx) ** 2 + (gy - by) ** 2 <= AROUND_RADIUS_M**2
