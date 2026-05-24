"""Defensive shape at the cross (start-frame, xCross).

How high the line is pushed (x of the deepest outfield defender = offside line; larger = deeper,
closer to its own goal), how wide and how big the block is, and the gap between that last line
and the goalkeeper. The keeper is excluded from the outfield line."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from xcross.features.frames import FrameState


def _outfield(defense_pos: np.ndarray, gk_pos: tuple[float, float] | None) -> np.ndarray:
    if gk_pos is None or len(defense_pos) == 0:
        return defense_pos
    return defense_pos[~np.all(np.isclose(defense_pos, np.array(gk_pos)), axis=1)]


def _hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # in 2D, ConvexHull.volume is the polygon area
    except QhullError:  # collinear defenders
        return 0.0


def defensive_shape(state: FrameState, half_length: float) -> dict:
    outfield = _outfield(state.defense_pos, state.gk_pos)
    if len(outfield) == 0:
        return {"shape_line_height": 0.0, "shape_block_width": 0.0,
                "shape_block_area": 0.0, "shape_last_line_to_gk_gap": 0.0}
    line_height = float(outfield[:, 0].max())
    gap = float(state.gk_pos[0] - line_height) if state.gk_pos is not None else 0.0
    return {
        "shape_line_height": line_height,
        "shape_block_width": float(outfield[:, 1].max() - outfield[:, 1].min()),
        "shape_block_area": _hull_area(outfield),
        "shape_last_line_to_gk_gap": gap,
    }
