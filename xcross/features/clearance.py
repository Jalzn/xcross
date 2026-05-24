"""Vertical clearance of the ball over defenders and the keeper (xCrossOT, the ball z).

The 3D version of "did the cross beat the first defender / clear the keeper": for each player the
ball passes near horizontally, its height there minus that player's reach. Uses the start-frame
defender positions as a static approximation (cheap — no per-frame player scan)."""

from __future__ import annotations

import numpy as np

from xcross.config import CLEARANCE_NEAR_M, DEFENDER_REACH_M, GK_REACH_M

_NO_OBSTACLE = 10.0  # finite sentinel: clears everything by a lot (m) when no player is on the path


def _height_passing(traj: np.ndarray, point: tuple[float, float]) -> float | None:
    """Ball z where its xy is nearest `point`, or None if it never passes near it."""
    distance = np.hypot(traj[:, 1] - point[0], traj[:, 2] - point[1])
    nearest = int(np.argmin(distance))
    return float(traj[nearest, 3]) if distance[nearest] < CLEARANCE_NEAR_M else None


def vertical_clearance(
    defense_pos: np.ndarray, gk_pos: tuple[float, float] | None, traj: np.ndarray
) -> dict:
    if len(traj) == 0:
        return {"clearance_min_margin_over_defender": _NO_OBSTACLE, "clearance_over_keeper": _NO_OBSTACLE}
    margins = [
        height - DEFENDER_REACH_M
        for defender in defense_pos
        if (height := _height_passing(traj, defender)) is not None
    ]
    over_keeper = _NO_OBSTACLE
    if gk_pos is not None and (height := _height_passing(traj, gk_pos)) is not None:
        over_keeper = height - GK_REACH_M
    return {
        "clearance_min_margin_over_defender": float(min(margins)) if margins else _NO_OBSTACLE,
        "clearance_over_keeper": float(over_keeper),
    }
