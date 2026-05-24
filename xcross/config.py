"""Paths and constants for the xCross pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
FEATURES_ROOT = ROOT / "data" / "features"

# Window (seconds) after a cross to count downstream shots toward shot_in_window.
SHOT_WINDOW_S = 5.0

# Hard cap (seconds) on a cross window. The window normally ends at the next
# possession event or at the ball going OUT; this only kicks in when neither
# happens (e.g. the ball rolls untouched), so the window never runs away.
CROSS_MAX_WINDOW_S = 5.0

# PFF coordinates are real metres on the real pitch, centred at (0, 0). The zone
# boundaries below are derived per match from the actual half-length/half-width
# (see _in_target_zone / _in_origin_zone); these values are only the fallback for
# matches whose metadata is missing the pitch dimensions.
PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0

# Penalty-box dimensions are fixed by the laws of the game, independent of pitch
# size: 16.5 m deep, 40.32 m wide (= +-20.16 m). Used for both the target zone
# (paper Sec. 2.2, Fig. 2) and the "out wide" origin threshold (Sec. 2.1, Fig. 1).
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
END_LINE_TOLERANCE_M = 2.0

# Corner exclusion: PFF tags corner deliveries as CR but this export has no
# setpiece_type to filter them. They form an isolated origin cluster in the pitch
# corner; these margins (empirical, from the start-position heatmap on 105x68
# pitches) carve that corner off the goal line and the touchline.
CORNER_X_MARGIN_M = 4.5
CORNER_Y_MARGIN_M = 3.0

# --- Spatial maps: grid, KDE, player velocity (paper Sec 3.2) ---
# Grid over the attacking half (x in [0, half_length], y in [-half_width, half_width]).
GRID_NX = 120  # cells along x (length)
GRID_NY = 80   # cells along y (width)
KDE_BANDWIDTH_M = 4.0          # gaussian KDE smoothing of player density
VELOCITY_DELTA_FRAMES = 3      # frames apart used to estimate player velocity
VELOCITY_PROJECTION_S = 1.0    # project players this far ahead by their velocity
PLAYER_MAX_SPEED_MS = 7.0      # pitch-control: speed used for arrival time
PITCH_CONTROL_TAU_S = 1.0      # pitch-control: softness of the arrival-time mix

# --- Region masks within the attacking half (standardised coords) ---
AROUND_RADIUS_M = 10.0          # circle around the ball
CENTER_BOX_HALF_WIDTH_M = 9.16  # goal-area half width (central strip of the box)
ZONE_DEPTH_M = 25.0             # tactical zone: depth from the goal line (incl. box edge)
NEAR_ACTION_LINE_M = 5.0        # perpendicular distance to the origin->destination line

# --- New feature blocks: start-frame (xCross) ---
GOAL_HALF_WIDTH_M = 3.66        # half of the 7.32 m goal mouth, for goal-angle features
PRESSURE_RADIUS_M = 3.0         # pressure: a defender within this of the crosser presses him
FREE_MARK_RADIUS_M = 2.5        # marking: a box attacker with no defender within this is "free"

# --- New feature blocks: ball flight / OT (xCrossOT, the ball z trajectory) ---
FLIGHT_GROUND_M = 1.0           # flight: below this the ball is "on the ground" (hang-time / bounce)
ARRIVAL_HEAD_MAX_M = 2.2        # arrival: ball above this in the box went over everyone's head
DEFENDER_REACH_M = 2.2          # clearance: outfield standing/jumping reach
GK_REACH_M = 2.6                # clearance: keeper jumping reach
CLEARANCE_NEAR_M = 2.0          # clearance: ball counts as "over" a player within this horizontally
RING_INNER_M = 6.0             # support: inner radius of the second-ball ring around the drop zone
RING_OUTER_M = 14.0           # support: outer radius of that ring
