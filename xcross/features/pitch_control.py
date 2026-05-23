"""Pitch control: probability the attacking team controls each grid cell.

Simplified arrival-time model: each player's projected position reaches a cell in
time ~ distance / max_speed; a cell's control is a soft mix (exp(-t/tau)) of the
two teams' arrival influences. Reuses the same grid/velocities as the entropy maps.
This is the baseline; the paper does not specify its pitch-control formulation.
"""

from __future__ import annotations

import numpy as np

from xcross.config import (
    PITCH_CONTROL_TAU_S,
    PLAYER_MAX_SPEED_MS,
    VELOCITY_PROJECTION_S,
)
from xcross.features.grid import grid_centers


def _arrival_influence(
    positions: np.ndarray, velocities: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros_like(gx)
    projected = positions + velocities * VELOCITY_PROJECTION_S
    dx = gx[..., None] - projected[:, 0]
    dy = gy[..., None] - projected[:, 1]
    arrival_time = np.sqrt(dx**2 + dy**2) / PLAYER_MAX_SPEED_MS
    return np.exp(-arrival_time / PITCH_CONTROL_TAU_S).sum(axis=2)


def pitch_control_map(
    attack_pos: np.ndarray,
    attack_vel: np.ndarray,
    defense_pos: np.ndarray,
    defense_vel: np.ndarray,
    half_length: float,
    half_width: float,
) -> np.ndarray:
    """P(attacking team controls cell), shaped (GRID_NX, GRID_NY), values in [0, 1]."""
    gx, gy = grid_centers(half_length, half_width)
    attack = _arrival_influence(attack_pos, attack_vel, gx, gy)
    defense = _arrival_influence(defense_pos, defense_vel, gx, gy)
    return attack / (attack + defense + 1e-9)
