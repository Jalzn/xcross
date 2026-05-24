import numpy as np
import polars as pl

from xcross.features.clearance import vertical_clearance
from xcross.features.counts import count_features, near_action_line_counts
from xcross.features.coverage import coverage_mismatch
from xcross.features.entropy import entropy_map, occupancy_grid
from xcross.features.events import _cross_region, destination_features, event_features
from xcross.features.flight import ball_trajectory, flight_features
from xcross.features.frames import FrameState
from xcross.features.geometry import goal_angle
from xcross.features.goalkeeper import goalkeeper_features
from xcross.features.grid import box_mask, first_post_mask, grid_centers, second_post_mask
from xcross.features.marking import marking_tightness
from xcross.features.pitch_control import pitch_control_map
from xcross.features.pockets import largest_empty_pocket
from xcross.features.pressure import crosser_pressure
from xcross.features.shape import defensive_shape
from xcross.features.spatial import spatial_features
from xcross.features.support import second_ball_support
from xcross.features.swing import swing_features
from xcross.features.temporal import defensive_collapse, zone_dynamics_delta

HL, HW = 52.5, 34.0


def _state(attack_pos, defense_pos):
    a = np.array(attack_pos, dtype=float).reshape(-1, 2)
    d = np.array(defense_pos, dtype=float).reshape(-1, 2)
    return FrameState(a, np.zeros_like(a), d, np.zeros_like(d), (0.0, 0.0))


def _state_full(attack_pos, defense_pos, *, attack_vel=None, defense_vel=None,
                crosser_pos=None, gk_pos=None, gk_vel=None, ball_xy=(0.0, 0.0)):
    a = np.array(attack_pos, dtype=float).reshape(-1, 2)
    d = np.array(defense_pos, dtype=float).reshape(-1, 2)
    av = np.array(attack_vel, dtype=float).reshape(-1, 2) if attack_vel is not None else np.zeros_like(a)
    dv = np.array(defense_vel, dtype=float).reshape(-1, 2) if defense_vel is not None else np.zeros_like(d)
    return FrameState(a, av, d, dv, ball_xy, crosser_pos, gk_pos, gk_vel)


def _traj(rows):
    """Ball trajectory as (M, 4) [frame_num, x, y, z]."""
    return np.array(rows, dtype=float)


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


# --- crosser pressure ---

def test_crosser_pressure_nearest_defender():
    state = _state_full([[40, 30]], [[42, 30], [10, 0]], crosser_pos=(40.0, 30.0))
    p = crosser_pressure(state)
    assert abs(p["pressure_crosser_nearest_def"] - 2.0) < 1e-9
    assert p["pressure_crosser_def_within_3m"] == 1


def test_crosser_pressure_no_crosser_is_sentinel():
    p = crosser_pressure(_state_full([[40, 30]], [], crosser_pos=None))
    assert p["pressure_crosser_nearest_def"] == 100.0
    assert p["pressure_crosser_def_within_3m"] == 0


# --- marking tightness ---

def test_marking_free_attacker_and_max_gap():
    state = _state_full([[50, 0], [50, 15]], [[50, 1], [50, 5]])
    m = marking_tightness(state, HL)
    assert m["marking_free_attackers_in_box"] == 1          # the one at y=15 is unmarked
    assert abs(m["marking_max_attacker_gap_in_box"] - 10.0) < 1e-9


# --- largest empty pocket ---

def test_pocket_radius_larger_with_sparse_defense():
    sparse = largest_empty_pocket(_state_full([[40, 0]], [[48, 0]]), HL, HW)
    dense = largest_empty_pocket(
        _state_full([[40, 0]], [[48, 0], [48, 5], [48, -5], [45, 10], [45, -10], [50, 0]]), HL, HW
    )
    assert sparse["pocket_radius_in_box"] > dense["pocket_radius_in_box"]


def test_goal_angle_central_greater_than_wide():
    assert goal_angle(48.0, 0.0, HL) > goal_angle(48.0, 25.0, HL)


# --- coverage mismatch (KL) ---

def test_coverage_kl_zero_when_densities_match():
    c = coverage_mismatch(_state_full([[50, 0]], [[50, 0]]), HL, HW)
    assert c["coverage_kl_attack_defense_in_box"] < 1e-6


def test_coverage_kl_positive_when_mismatched():
    c = coverage_mismatch(_state_full([[50, 15]], [[50, -15]]), HL, HW)
    assert c["coverage_kl_attack_defense_in_box"] > 0.1


# --- goalkeeper ---

def test_gk_distance_off_line_and_lateral_speed():
    g = goalkeeper_features((50.0, 1.0), (0.0, 2.0), (40.0, 30.0), HL)
    assert abs(g["gk_distance_off_line"] - 2.5) < 1e-9
    assert abs(g["gk_lateral_speed"] - 2.0) < 1e-9
    assert g["gk_present"] == 1


def test_gk_absent_is_sentinel():
    g = goalkeeper_features(None, None, (40.0, 30.0), HL)
    assert g["gk_present"] == 0
    assert g["gk_ball_distance"] == 100.0


# --- defensive shape ---

def test_shape_line_height_excludes_gk():
    state = _state_full([], [[40, 0], [48, 5], [44, -5], [52, 0]], gk_pos=(52.0, 0.0))
    s = defensive_shape(state, HL)
    assert abs(s["shape_line_height"] - 48.0) < 1e-9
    assert abs(s["shape_last_line_to_gk_gap"] - 4.0) < 1e-9
    assert s["shape_block_area"] > 0.0


def test_shape_block_area_zero_with_two_defenders():
    s = defensive_shape(_state_full([], [[40, 0], [48, 5]], gk_pos=None), HL)
    assert s["shape_block_area"] == 0.0


# --- ball flight ---

def test_ball_trajectory_sorts_and_clips_negative_z():
    df = pl.DataFrame({"frame_num": [2, 1, 0], "x": [3.0, 2.0, 1.0], "y": [0.0, 0.0, 0.0], "z": [-0.3, 2.0, 0.0]})
    traj = ball_trajectory(df, 0, 2)
    assert traj[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert (traj[:, 3] >= 0).all()


def test_flight_apex_timing_and_angles():
    traj = _traj([[0, 20, 0, 0], [1, 25, 0, 3], [2, 30, 0, 5], [3, 35, 0, 3], [4, 40, 0, 0.5]])
    f = flight_features(traj, fps=25.0)
    assert abs(f["flight_apex_height"] - 5.0) < 1e-9
    assert abs(f["flight_apex_timing"] - 0.5) < 1e-9
    assert f["flight_launch_angle"] > 0 and f["flight_descent_angle"] < 0
    assert f["flight_hang_time"] > 0


def test_flight_bounce_count():
    traj = _traj([[0, 10, 0, 3], [1, 12, 0, 0.2], [2, 14, 0, 2], [3, 16, 0, 0.1], [4, 18, 0, 1.5]])
    assert flight_features(traj, 25.0)["flight_bounce_count"] == 2.0


# --- vertical clearance ---

def test_clearance_margins_over_defender_and_keeper():
    traj = _traj([[0, 30, 0, 2], [1, 40, 0, 3.0], [2, 50, 0, 4.0]])
    c = vertical_clearance(np.array([[40.0, 0.0]]), (50.0, 0.0), traj)
    assert abs(c["clearance_min_margin_over_defender"] - (3.0 - 2.2)) < 1e-9
    assert abs(c["clearance_over_keeper"] - (4.0 - 2.6)) < 1e-9


def test_clearance_sentinel_when_nobody_on_path():
    traj = _traj([[0, 30, 0, 2], [1, 40, 0, 3]])
    c = vertical_clearance(np.array([[5.0, 30.0]]), None, traj)
    assert c["clearance_min_margin_over_defender"] == 10.0
    assert c["clearance_over_keeper"] == 10.0


# --- swing ---

def test_swing_cutback_detected_near_byline():
    row = {"start_x": 50.0, "start_y": 30.0, "end_x": 46.0, "end_y": 5.0}
    traj = _traj([[0, 50, 30, 1], [1, 48, 18, 2], [2, 46, 5, 1]])
    assert swing_features(row, traj, HL)["swing_cutback"] == 1


def test_swing_curl_magnitude_is_abs_of_signed():
    row = {"start_x": 40.0, "start_y": 0.0, "end_x": 50.0, "end_y": 0.0}
    traj = _traj([[0, 40, 0, 1], [1, 45, 4, 3], [2, 50, 0, 1]])
    s = swing_features(row, traj, HL)
    assert s["swing_curl_magnitude"] > 0
    assert abs(s["swing_inout"]) == s["swing_curl_magnitude"]


# --- second-ball support ---

def test_support_counts_in_ring_excludes_inner():
    state = _state_full([[50, 10], [50, 2]], [[50, 12]])
    sup = second_ball_support(state, (50.0, 0.0))
    assert sup["support_attackers_ring"] == 1   # (50,10) in ring, (50,2) inside inner
    assert sup["support_defenders_ring"] == 1


# --- temporal deltas ---

def test_defensive_collapse_counts_delta():
    start = _state([[50, 0]], [[49, 0]])                       # 1 attacker, 1 defender in box
    arrival = _state([[50, 0], [48, 5]], [[49, 0], [50, -3], [47, 8]])  # 2 atk, 3 def in box
    d = defensive_collapse(start, arrival, HL)
    assert d["temporal_defenders_in_box_delta"] == 2
    assert d["temporal_attackers_in_box_delta"] == 1


def test_zone_dynamics_delta_subtracts():
    start_zone = {"entropy_diff_in_zone": 1.0, "pitch_control_in_zone": 4.0}
    end_zone = {"entropy_diff_in_zone": 1.5, "pitch_control_in_zone": 3.0}
    d = zone_dynamics_delta(start_zone, end_zone)
    assert abs(d["temporal_entropy_diff_zone_delta"] - 0.5) < 1e-9
    assert abs(d["temporal_pitch_control_zone_delta"] - (-1.0)) < 1e-9
