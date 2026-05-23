"""Scalar entropy / pitch-control features: sums of the maps over regions.

For each region (and the whole attacking half = `sum`) we sum the Shannon-contribution
map (per team: attack/defense/general, plus diff = attack-defense) and the pitch-control
map. `grad_towards_goal` contrasts the goal-side half against the far half.
"""

from __future__ import annotations

import numpy as np

from xcross.features.entropy import entropy_map, occupancy_grid
from xcross.features.frames import FrameState
from xcross.features.grid import grid_centers
from xcross.features.pitch_control import pitch_control_map


def _team_entropy_maps(
    state: FrameState, half_length: float, half_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ_a = occupancy_grid(state.attack_pos, state.attack_vel, half_length, half_width)
    occ_d = occupancy_grid(state.defense_pos, state.defense_vel, half_length, half_width)
    pos = np.vstack([state.attack_pos, state.defense_pos]) if len(state.defense_pos) else state.attack_pos
    vel = np.vstack([state.attack_vel, state.defense_vel]) if len(state.defense_vel) else state.attack_vel
    occ_g = occupancy_grid(pos, vel, half_length, half_width)
    return entropy_map(occ_a), entropy_map(occ_d), entropy_map(occ_g)


def spatial_features(
    state: FrameState,
    half_length: float,
    half_width: float,
    regions: dict[str, np.ndarray],
    with_grad: bool,
) -> dict[str, float]:
    """Entropy + pitch-control sums over each named region mask (boolean grid)."""
    em_a, em_d, em_g = _team_entropy_maps(state, half_length, half_width)
    pc = pitch_control_map(
        state.attack_pos, state.attack_vel, state.defense_pos, state.defense_vel,
        half_length, half_width,
    )
    feats: dict[str, float] = {}
    for ctx, mask in regions.items():
        attack, defense = float(em_a[mask].sum()), float(em_d[mask].sum())
        feats[f"entropy_attack_{ctx}"] = attack
        feats[f"entropy_defense_{ctx}"] = defense
        feats[f"entropy_general_{ctx}"] = float(em_g[mask].sum())
        feats[f"entropy_diff_{ctx}"] = attack - defense
        feats[f"pitch_control_{ctx}"] = float(pc[mask].sum())

    if with_grad:
        gx, _ = grid_centers(half_length, half_width)
        near = gx >= half_length / 2
        for name, em in (("attack", em_a), ("defense", em_d), ("general", em_g)):
            feats[f"entropy_{name}_grad_towards_goal"] = float(em[near].sum() - em[~near].sum())
        feats["entropy_diff_grad_towards_goal"] = (
            feats["entropy_attack_grad_towards_goal"] - feats["entropy_defense_grad_towards_goal"]
        )
        feats["pitch_control_grad_towards_goal"] = float(pc[near].sum() - pc[~near].sum())
    return feats
