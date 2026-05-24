"""Scalar pitch geometry shared across feature blocks (standardised coords, goal at +x)."""

from __future__ import annotations

import numpy as np

from xcross.config import GOAL_HALF_WIDTH_M


def goal_angle(x: float, y: float, half_length: float) -> float:
    """Angle (rad) the goal mouth subtends from (x, y) — wider = better shooting angle."""
    post_near = np.array([half_length - x, GOAL_HALF_WIDTH_M - y])
    post_far = np.array([half_length - x, -GOAL_HALF_WIDTH_M - y])
    norm = np.linalg.norm(post_near) * np.linalg.norm(post_far)
    if norm == 0:
        return 0.0
    cosine = float(np.dot(post_near, post_far) / norm)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))
