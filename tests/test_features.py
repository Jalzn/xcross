import numpy as np

from xcross.features.counts import count_features, near_action_line_counts
from xcross.features.entropy import entropy_map, occupancy_grid
from xcross.features.events import _cross_region, destination_features, event_features
from xcross.features.frames import FrameState
from xcross.features.grid import box_mask, first_post_mask, grid_centers, second_post_mask
from xcross.features.pitch_control import pitch_control_map
from xcross.features.spatial import spatial_features

HL, HW = 52.5, 34.0


def _state(attack_pos, defense_pos):
    a = np.array(attack_pos, dtype=float).reshape(-1, 2)
    d = np.array(defense_pos, dtype=float).reshape(-1, 2)
    return FrameState(a, np.zeros_like(a), d, np.zeros_like(d), (0.0, 0.0))


# --- occupancy / entropy ---

def test_occupancy_grid_is_a_distribution():
    grid = occupancy_grid(np.array([[40.0, 5.0]]), np.zeros((1, 2)), HL, HW)
    assert grid.shape == (120, 80)
    assert abs(grid.sum() - 1.0) < 1e-9


def test_occupancy_grid_empty_team_is_zero():
    assert occupancy_grid(np.zeros((0, 2)), np.zeros((0, 2)), HL, HW).sum() == 0.0


def test_occupancy_projects_along_velocity():
    pos = np.array([[30.0, 0.0]])
    gx, _ = grid_centers(HL, HW)
    static = (occupancy_grid(pos, np.zeros((1, 2)), HL, HW) * gx).sum()
    moving = (occupancy_grid(pos, np.array([[5.0, 0.0]]), HL, HW) * gx).sum()
    assert moving > static  # centre of mass shifts toward +x


def test_entropy_map_is_nonnegative_and_zero_on_empty():
    grid = occupancy_grid(np.array([[40.0, 0.0]]), np.zeros((1, 2)), HL, HW)
    assert (entropy_map(grid) >= 0).all()
    assert entropy_map(np.zeros((120, 80))).sum() == 0.0


def test_spread_players_have_more_entropy_than_concentrated():
    one = occupancy_grid(np.array([[40.0, 0.0]]), np.zeros((1, 2)), HL, HW)
    spread = occupancy_grid(
        np.array([[10, 0], [25, -20], [45, 20], [5, 10]], float), np.zeros((4, 2)), HL, HW
    )
    assert entropy_map(spread).sum() > entropy_map(one).sum()


# --- pitch control ---

def test_pitch_control_in_unit_range():
    pc = pitch_control_map(
        np.array([[40.0, 0.0]]), np.zeros((1, 2)),
        np.array([[10.0, 0.0]]), np.zeros((1, 2)), HL, HW,
    )
    assert pc.min() >= 0.0 and pc.max() <= 1.0


def test_pitch_control_favours_the_closer_team():
    gx, _ = grid_centers(HL, HW)
    pc = pitch_control_map(
        np.array([[50.0, 0.0]]), np.zeros((1, 2)),
        np.array([[5.0, 0.0]]), np.zeros((1, 2)), HL, HW,
    )
    assert pc[gx >= 45].mean() > 0.5  # attacker owns the area near goal


# --- grid masks ---

def test_box_mask_inside_outside():
    gx = np.array([50.0, 30.0, 50.0])
    gy = np.array([0.0, 0.0, 25.0])
    assert box_mask(gx, gy, HL).tolist() == [True, False, False]


def test_post_masks_split_by_flank():
    gx = np.array([50.0, 50.0])
    gy = np.array([10.0, -10.0])
    assert first_post_mask(gx, gy, HL).tolist() == [True, False]
    assert second_post_mask(gx, gy, HL).tolist() == [False, True]


# --- event geometry ---

def test_cross_region_byline_touchline():
    assert _cross_region(48.0, 30.0, HL) == "byline_touchline"


def test_event_distance_and_cross_angle():
    row = {"start_x": 40.0, "start_y": 10.0, "end_x": 50.0, "end_y": 0.0}
    assert abs(event_features(row, HL)["distance_start_from_goal"] - np.hypot(12.5, 10.0)) < 1e-9
    dest = destination_features(row, HL)
    assert abs(dest["distance_from_end_line"] - 2.5) < 1e-9
    assert abs(dest["polar_angle_cross"] - np.arctan2(-10.0, 10.0)) < 1e-9


# --- counts ---

def test_counts_in_box_and_ratio():
    state = _state([[50, 0], [48, 10], [10, 0]], [[49, 0]])
    counts = count_features(state, HL)
    assert counts["attackers_in_box"] == 2
    assert counts["defenders_in_box"] == 1
    assert counts["box_ratio"] == 2.0


def test_near_action_line_counts_perpendicular_distance():
    state = _state([[30, 0], [30, 10]], [])
    counts = near_action_line_counts(state, (20.0, 0.0), (50.0, 0.0))
    assert counts["attackers_near_action_line"] == 1  # only the one on the line (<5 m)


# --- spatial aggregation ---

def test_spatial_diff_is_attack_minus_defense():
    state = _state([[45, 5], [40, 0]], [[44, 0]])
    gx, gy = grid_centers(HL, HW)
    feats = spatial_features(state, HL, HW, {"sum": np.ones_like(gx, dtype=bool)}, with_grad=True)
    assert abs(feats["entropy_diff_sum"] - (feats["entropy_attack_sum"] - feats["entropy_defense_sum"])) < 1e-9
    assert feats["entropy_attack_sum"] >= 0
    assert "entropy_general_grad_towards_goal" in feats
    assert "pitch_control_grad_towards_goal" in feats
