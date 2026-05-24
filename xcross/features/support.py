"""Second-ball support around the cross's drop zone (xCrossOT, label-aware for possession).

Success = possession kept, so it is won even when the first contact is contested — whoever is
positioned to win the second ball around where the cross lands matters. Depends on the landing
(end_xy), so this is an OT block, not part of the start-frame creation set."""

from __future__ import annotations

import numpy as np

from xcross.config import RING_INNER_M, RING_OUTER_M
from xcross.features.frames import FrameState


def _in_ring(positions: np.ndarray, centre: tuple[float, float]) -> int:
    if len(positions) == 0:
        return 0
    distance = np.hypot(positions[:, 0] - centre[0], positions[:, 1] - centre[1])
    return int(((distance >= RING_INNER_M) & (distance <= RING_OUTER_M)).sum())


def second_ball_support(state: FrameState, drop_xy: tuple[float, float]) -> dict:
    attackers = _in_ring(state.attack_pos, drop_xy)
    defenders = _in_ring(state.defense_pos, drop_xy)
    return {
        "support_attackers_ring": attackers,
        "support_defenders_ring": defenders,
        "support_ratio_ring": attackers / max(1, defenders),
    }
