"""Temporal deltas across the cross window (xCrossOT): how the box and the arrival zone change
from the cross being played to the ball arriving. Uses only the start and arrival frames — no
full per-frame scan, so it is cheap."""

from __future__ import annotations

from xcross.features.counts import count_features
from xcross.features.frames import FrameState


def defensive_collapse(start: FrameState, arrival: FrameState, half_length: float) -> dict:
    """Change in box occupancy over the window — defenders converging, attackers arriving."""
    at_start = count_features(start, half_length)
    at_arrival = count_features(arrival, half_length)
    return {
        "temporal_defenders_in_box_delta": at_arrival["defenders_in_box"] - at_start["defenders_in_box"],
        "temporal_attackers_in_box_delta": at_arrival["attackers_in_box"] - at_start["attackers_in_box"],
    }


def zone_dynamics_delta(start_zone: dict, end_zone: dict) -> dict:
    """Change in the arrival-zone entropy/pitch-control from the start frame to arrival — the
    disorder the cross creates where it lands. `start_zone`/`end_zone` are `spatial_features`
    dicts over the same `in_zone` mask at the two frames."""
    return {
        "temporal_entropy_diff_zone_delta": end_zone["entropy_diff_in_zone"] - start_zone["entropy_diff_in_zone"],
        "temporal_pitch_control_zone_delta": end_zone["pitch_control_in_zone"] - start_zone["pitch_control_in_zone"],
    }
