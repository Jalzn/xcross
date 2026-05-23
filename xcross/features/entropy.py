"""Positional entropy: player density on the grid and its Shannon contribution.

Paper Sec 3.2: project each player 1 s ahead by their velocity, smooth with a
gaussian KDE into an occupancy density, then H = -sum p_i log p_i over cells.
`occupancy_grid` is the spatial representation (also the future CNN/ViT input);
`entropy_map` is the per-cell Shannon contribution whose sum over a region is H.
"""

from __future__ import annotations

import numpy as np

from xcross.config import KDE_BANDWIDTH_M, VELOCITY_PROJECTION_S
from xcross.features.grid import grid_centers


def occupancy_grid(
    positions: np.ndarray, velocities: np.ndarray, half_length: float, half_width: float
) -> np.ndarray:
    """Normalised player-presence density (sums to 1) on the attacking-half grid.

    `positions`/`velocities` are (N, 2) arrays for one team; players are projected
    `VELOCITY_PROJECTION_S` ahead, then a gaussian kernel is summed over the grid.
    """
    gx, gy = grid_centers(half_length, half_width)
    if len(positions) == 0:
        return np.zeros_like(gx)
    projected = positions + velocities * VELOCITY_PROJECTION_S
    dx = gx[..., None] - projected[:, 0]
    dy = gy[..., None] - projected[:, 1]
    kernels = np.exp(-(dx**2 + dy**2) / (2 * KDE_BANDWIDTH_M**2))
    density = kernels.sum(axis=2)
    total = density.sum()
    return density / total if total > 0 else density


def entropy_map(density: np.ndarray) -> np.ndarray:
    """Per-cell Shannon contribution -p log p (0 where p=0). Sum over cells gives H."""
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = -density * np.log(density)
    return np.where(np.isfinite(contribution), contribution, 0.0)
