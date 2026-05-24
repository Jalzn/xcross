"""Coverage mismatch: KL divergence between attacker and defender density (start-frame, xCross).

The per-team entropies say how spread each team is; this says whether the defenders are *where
the attackers are*. High KL = the defence fails to cover the attack's distribution = free space.
Reuses the same occupancy grids as the entropy block."""

from __future__ import annotations

import numpy as np

from xcross.features.entropy import occupancy_grid
from xcross.features.frames import FrameState
from xcross.features.grid import box_mask, grid_centers, zone_mask

_EPS = 1e-9


def _kl(p_grid: np.ndarray, q_grid: np.ndarray, mask: np.ndarray) -> float:
    p, q = p_grid[mask], q_grid[mask]
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p, q = p / p_sum, q / q_sum
    nonzero = p > 0
    return float(np.sum(p[nonzero] * np.log((p[nonzero] + _EPS) / (q[nonzero] + _EPS))))


def coverage_mismatch(state: FrameState, half_length: float, half_width: float) -> dict:
    gx, gy = grid_centers(half_length, half_width)
    attack = occupancy_grid(state.attack_pos, state.attack_vel, half_length, half_width)
    defense = occupancy_grid(state.defense_pos, state.defense_vel, half_length, half_width)
    return {
        "coverage_kl_attack_defense_in_box": _kl(attack, defense, box_mask(gx, gy, half_length)),
        "coverage_kl_attack_defense_in_zone": _kl(attack, defense, zone_mask(gx, gy, half_length)),
    }
