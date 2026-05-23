"""Player-count features from a frame's positions (per team)."""

from __future__ import annotations

import numpy as np

from xcross.config import BOX_DEPTH_M, BOX_HALF_WIDTH_M, NEAR_ACTION_LINE_M, ZONE_DEPTH_M
from xcross.features.frames import FrameState


def _in_box(pos: np.ndarray, half_length: float) -> np.ndarray:
    if len(pos) == 0:
        return np.zeros(0, dtype=bool)
    return (pos[:, 0] >= half_length - BOX_DEPTH_M) & (np.abs(pos[:, 1]) <= BOX_HALF_WIDTH_M)


def _in_zone(pos: np.ndarray, half_length: float) -> np.ndarray:
    if len(pos) == 0:
        return np.zeros(0, dtype=bool)
    return (pos[:, 0] >= half_length - ZONE_DEPTH_M) & (np.abs(pos[:, 1]) <= BOX_HALF_WIDTH_M)


def count_features(state: FrameState, half_length: float) -> dict:
    a_box = int(_in_box(state.attack_pos, half_length).sum())
    d_box = int(_in_box(state.defense_pos, half_length).sum())
    a_zone = int(_in_zone(state.attack_pos, half_length).sum())
    d_zone = int(_in_zone(state.defense_pos, half_length).sum())
    return {
        "attackers_in_box": a_box,
        "defenders_in_box": d_box,
        "attackers_in_zone": a_zone,
        "defenders_in_zone": d_zone,
        "box_ratio": a_box / max(1, d_box),
        "zone_ratio": a_zone / max(1, d_zone),
    }


def near_action_line_counts(
    state: FrameState, start_xy: tuple[float, float], end_xy: tuple[float, float]
) -> dict:
    """Players within NEAR_ACTION_LINE_M of the origin→destination line (OT)."""
    sx, sy = start_xy
    dx, dy = end_xy[0] - sx, end_xy[1] - sy
    length = float(np.hypot(dx, dy))

    def near(pos: np.ndarray) -> int:
        if len(pos) == 0 or length < 1e-6:
            return 0
        dist = np.abs((pos[:, 0] - sx) * dy - (pos[:, 1] - sy) * dx) / length
        return int((dist < NEAR_ACTION_LINE_M).sum())

    return {
        "attackers_near_action_line": near(state.attack_pos),
        "defenders_near_action_line": near(state.defense_pos),
    }
