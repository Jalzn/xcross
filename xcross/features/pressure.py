"""Pressure on the crosser at the moment of the cross (start-frame, xCross).

A crosser closed down by a defender delivers a worse ball; this is a reproducible trait of
the situation the player creates, independent of where the ball lands."""

from __future__ import annotations

import numpy as np

from xcross.config import PRESSURE_RADIUS_M
from xcross.features.frames import FrameState

_NO_DEFENDER_DIST = 100.0  # finite sentinel (> pitch diagonal) when the crosser/defenders are absent


def crosser_pressure(state: FrameState) -> dict:
    crosser = state.crosser_pos
    if crosser is None or len(state.defense_pos) == 0:
        return {"pressure_crosser_nearest_def": _NO_DEFENDER_DIST, "pressure_crosser_def_within_3m": 0}
    dist = np.hypot(state.defense_pos[:, 0] - crosser[0], state.defense_pos[:, 1] - crosser[1])
    return {
        "pressure_crosser_nearest_def": float(dist.min()),
        "pressure_crosser_def_within_3m": int((dist < PRESSURE_RADIUS_M).sum()),
    }
