"""Marking tightness in the box at the cross (start-frame, xCross).

For each box attacker, the distance to the nearest defender: an unmarked attacker is the
textbook driver of a successful cross, more directly than the smoothed pitch-control map."""

from __future__ import annotations

from scipy.spatial import cKDTree

from xcross.config import FREE_MARK_RADIUS_M
from xcross.features.counts import _in_box
from xcross.features.frames import FrameState

_OPEN = 50.0  # finite sentinel: "wide open" gap (m) when there are no defenders


def marking_tightness(state: FrameState, half_length: float) -> dict:
    attackers = state.attack_pos[_in_box(state.attack_pos, half_length)]
    if len(attackers) == 0:
        return {"marking_max_attacker_gap_in_box": 0.0, "marking_free_attackers_in_box": 0,
                "marking_mean_gap_in_box": 0.0}
    if len(state.defense_pos) == 0:
        return {"marking_max_attacker_gap_in_box": _OPEN, "marking_free_attackers_in_box": len(attackers),
                "marking_mean_gap_in_box": _OPEN}
    nearest, _ = cKDTree(state.defense_pos).query(attackers)
    return {
        "marking_max_attacker_gap_in_box": float(nearest.max()),
        "marking_free_attackers_in_box": int((nearest > FREE_MARK_RADIUS_M).sum()),
        "marking_mean_gap_in_box": float(nearest.mean()),
    }
