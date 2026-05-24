"""Largest defender-free pocket inside the box (start-frame, xCross).

Where is the space to aim at? The largest empty circle (centre on a box grid cell, radius =
distance to the nearest defender), plus the goal angle from that pocket."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from xcross.features.frames import FrameState
from xcross.features.geometry import goal_angle
from xcross.features.grid import box_mask, grid_centers


def largest_empty_pocket(state: FrameState, half_length: float, half_width: float) -> dict:
    gx, gy = grid_centers(half_length, half_width)
    mask = box_mask(gx, gy, half_length)
    cells = np.column_stack([gx[mask], gy[mask]])
    if len(state.defense_pos) == 0:
        centre = cells[np.argmax(cells[:, 0])]  # no defenders: aim deepest, pocket = whole box
        radius = float(half_width)
    else:
        distances, _ = cKDTree(state.defense_pos).query(cells)
        best = int(np.argmax(distances))
        centre, radius = cells[best], float(distances[best])
    return {
        "pocket_radius_in_box": radius,
        "pocket_goal_angle_in_box": goal_angle(float(centre[0]), float(centre[1]), half_length),
    }
