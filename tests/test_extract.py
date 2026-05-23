from pathlib import Path

from xcross.data import extract as extract_mod
from xcross.data.extract import (
    _in_origin_zone,
    _in_target_zone,
    _normalize_flank,
    extract_match,
)
from xcross.data.meta import MatchMeta

FPS = 30.0
CAP_FRAMES = 150  # CROSS_MAX_WINDOW_S (5s) * FPS


def _meta():
    return MatchMeta(
        match_id="m1", league="L", season="S", date="",
        home_team_id=1, home_team_name="H",
        away_team_id=2, away_team_name="A",
        home_team_start_left=True, fps=FPS,
        pitch_length_m=105.0, pitch_width_m=68.0,
    )


ORIGIN_BALL = (38.0, 25.0, 0.5)  # final third + wide: inside the origin zone
TARGET_BALL = (45.0, 8.0, 0.5)   # inside the opposing box: inside the target zone


def _frame(frame_num, ball=TARGET_BALL, ge=None, ge_id=None, pe=None, pe_id=None):
    """Minimal PFF frame. Ball defaults to TARGET_BALL (inside the box)."""
    frame = {
        "frameNum": frame_num,
        "period": 1,
        "homePlayersSmoothed": [],
        "awayPlayersSmoothed": [],
        "ballsSmoothed": {"x": ball[0], "y": ball[1], "z": ball[2]},
    }
    if ge is not None:
        frame["game_event"] = ge
        frame["game_event_id"] = ge_id
    if pe is not None:
        frame["possession_event"] = pe
        frame["possession_event_id"] = pe_id
    return frame


def _cross_frame(frame_num, ge_id=500, pe_id=900, ball=ORIGIN_BALL):
    return _frame(
        frame_num,
        ball=ball,
        ge={"team_id": 1, "player_id": 10, "game_event_type": "OTB", "setpiece_type": None},
        ge_id=ge_id,
        pe={"possession_event_type": "CR", "game_event_id": ge_id, "start_time": frame_num / FPS},
        pe_id=pe_id,
    )


def _possession_frame(frame_num, pe_type, team_id=1, ge_id=700):
    """End event with a team, so the outcome resolves (default team 1 = crosser)."""
    return _frame(
        frame_num,
        ge={"team_id": team_id, "player_id": 11, "game_event_type": "OTB"},
        ge_id=ge_id,
        pe={"possession_event_type": pe_type, "game_event_id": ge_id, "start_time": frame_num / FPS},
        pe_id=frame_num,
    )


def _out_frame(frame_num, ge_id=600):
    return _frame(frame_num, ge={"team_id": 2, "player_id": None, "game_event_type": "OUT"}, ge_id=ge_id)


def _run(frames, monkeypatch):
    monkeypatch.setattr(extract_mod, "load_metadata", lambda *a, **k: _meta())
    monkeypatch.setattr(extract_mod, "load_roster", lambda *a, **k: [])
    monkeypatch.setattr(extract_mod, "iter_frames", lambda *a, **k: iter(frames))
    return extract_match(Path("m1"), "L")


def _extract(frames, monkeypatch):
    return _run(frames, monkeypatch).crosses.to_dicts()


def test_window_ends_at_next_possession_event(monkeypatch):
    frames = [_cross_frame(100)]
    frames += [_frame(f) for f in range(101, 130)]
    frames.append(_possession_frame(130, "RE"))  # reception by the crossing team

    rows = _extract(frames, monkeypatch)

    assert len(rows) == 1
    assert rows[0]["end_frame"] == 130
    assert rows[0]["end_event_type"] == "RE"
    assert rows[0]["ball_out_in_window"] is False
    assert rows[0]["success"] is True


def test_window_ends_at_out_not_at_far_restart(monkeypatch):
    frames = [_cross_frame(100)]
    frames += [_frame(f) for f in range(101, 110)]
    frames.append(_out_frame(110))
    frames.append(_possession_frame(400, "PA"))  # restart 10s later must be ignored

    rows = _extract(frames, monkeypatch)

    assert rows[0]["end_frame"] == 110
    assert rows[0]["end_event_type"] == "OUT"
    assert rows[0]["ball_out_in_window"] is True
    assert rows[0]["success"] is False


def test_unresolved_cross_beyond_cap_is_dropped(monkeypatch):
    # A resolvable reception arrives well past the 5s cap: the window caps out first
    # (unresolved outcome), so the cross is dropped instead of grabbing a far event.
    late = 100 + CAP_FRAMES + 60
    frames = [_cross_frame(100)]
    frames += [_frame(f) for f in range(101, late)]
    frames.append(_possession_frame(late, "RE"))

    rows = _extract(frames, monkeypatch)

    assert rows == []


def test_repeated_cr_frame_makes_one_cross_not_zero_length_ones(monkeypatch):
    # PFF repeats the CR frame; duplicates must not open/close extra zero-length crosses.
    frames = [_cross_frame(100), _cross_frame(100), _cross_frame(100)]  # same frame & pe_id
    frames += [_frame(f) for f in range(101, 130)]
    frames.append(_possession_frame(130, "RE"))

    rows = _extract(frames, monkeypatch)

    assert len(rows) == 1
    assert rows[0]["start_frame"] == 100
    assert rows[0]["end_frame"] == 130


def test_stats_track_the_cross_funnel(monkeypatch):
    # CR1: flank origin, resolved -> kept. CR2: central origin -> dropped (out of zone).
    frames = [_cross_frame(100)]
    frames += [_frame(f) for f in range(101, 130)]
    frames.append(_possession_frame(130, "RE", ge_id=700))
    frames.append(_cross_frame(200, ge_id=510, pe_id=910, ball=(38.0, 5.0, 0.5)))
    frames += [_frame(f) for f in range(201, 230)]
    frames.append(_possession_frame(230, "RE", ge_id=710))

    stats = _run(frames, monkeypatch).stats

    assert stats == {"n_cr_detected": 2, "n_in_zone": 1, "n_kept": 1}


HALF_LEN = 52.5  # 105 m pitch
HALF_WID = 34.0  # 68 m pitch


def test_origin_zone_accepts_wide_final_third_cross():
    assert _in_origin_zone(40.0, 25.0, HALF_LEN, HALF_WID) is True
    assert _in_origin_zone(40.0, -25.0, HALF_LEN, HALF_WID) is True  # symmetric across x-axis


def test_origin_zone_rejects_central_ball():
    assert _in_origin_zone(40.0, 5.0, HALF_LEN, HALF_WID) is False  # not wide enough


def test_origin_zone_rejects_ball_behind_final_third():
    assert _in_origin_zone(10.0, 25.0, HALF_LEN, HALF_WID) is False  # too far from goal


def test_origin_zone_rejects_corner():
    assert _in_origin_zone(52.0, 33.0, HALF_LEN, HALF_WID) is False  # set-piece corner cluster


def test_origin_zone_accepts_byline_cross_outside_corner():
    assert _in_origin_zone(50.0, 25.0, HALF_LEN, HALF_WID) is True  # deep but not in the corner


def test_origin_zone_corner_threshold_scales_with_pitch_width():
    # On a wider pitch (72 m -> half 36) the corner sits further out, so a cross at
    # |y|=32 is a legitimate flank cross, while on a 68 m pitch it is a corner.
    assert _in_origin_zone(52.0, 32.0, HALF_LEN, 36.0) is True
    assert _in_origin_zone(52.0, 32.0, HALF_LEN, HALF_WID) is False


def test_target_zone_accepts_ball_in_box():
    assert _in_target_zone(45.0, 8.0, HALF_LEN) is True
    assert _in_target_zone(45.0, -8.0, HALF_LEN) is True


def test_target_zone_rejects_ball_short_of_box():
    assert _in_target_zone(30.0, 8.0, HALF_LEN) is False  # x before the box edge


def test_target_zone_rejects_ball_wider_than_box():
    assert _in_target_zone(45.0, 25.0, HALF_LEN) is False  # |y| beyond box width


def test_target_zone_allows_end_line_tolerance_and_scales_with_pitch():
    assert _in_target_zone(54.0, 0.0, HALF_LEN) is True   # within 2 m past goal line
    assert _in_target_zone(55.0, 0.0, HALF_LEN) is False  # too far past on a 105 m pitch
    assert _in_target_zone(55.0, 0.0, 55.0) is True       # but fine on a 110 m pitch


def test_central_origin_cross_is_dropped(monkeypatch):
    # Ball starts central (out of the origin zone) but still reaches the box.
    frames = [_cross_frame(100, ball=(38.0, 5.0, 0.5))]
    frames += [_frame(f) for f in range(101, 130)]
    frames.append(_possession_frame(130, "RE"))

    rows = _extract(frames, monkeypatch)

    assert rows == []


def test_flips_cross_originating_from_negative_flank():
    cross_rows = [{"cross_id": "c1", "start_y": -10.0, "end_y": -3.0}]
    track_rows = [{"cross_id": "c1", "y": -8.0}]
    ball_rows = [{"cross_id": "c1", "y": -9.0}]

    _normalize_flank(cross_rows, track_rows, ball_rows)

    assert cross_rows[0]["start_y"] == 10.0
    assert cross_rows[0]["end_y"] == 3.0
    assert track_rows[0]["y"] == 8.0
    assert ball_rows[0]["y"] == 9.0


def test_keeps_cross_already_on_positive_flank():
    cross_rows = [{"cross_id": "c1", "start_y": 10.0, "end_y": 3.0}]
    track_rows = [{"cross_id": "c1", "y": 8.0}]
    ball_rows = [{"cross_id": "c1", "y": 9.0}]

    _normalize_flank(cross_rows, track_rows, ball_rows)

    assert cross_rows[0]["start_y"] == 10.0
    assert track_rows[0]["y"] == 8.0
    assert ball_rows[0]["y"] == 9.0


def test_treats_zero_start_y_as_positive_flank():
    cross_rows = [{"cross_id": "c1", "start_y": 0.0, "end_y": 5.0}]
    track_rows = [{"cross_id": "c1", "y": -2.0}]
    ball_rows = []

    _normalize_flank(cross_rows, track_rows, ball_rows)

    assert cross_rows[0]["end_y"] == 5.0
    assert track_rows[0]["y"] == -2.0


def test_flips_each_cross_independently():
    cross_rows = [
        {"cross_id": "c1", "start_y": -10.0, "end_y": -3.0},
        {"cross_id": "c2", "start_y": 10.0, "end_y": 3.0},
    ]
    track_rows = [
        {"cross_id": "c1", "y": -8.0},
        {"cross_id": "c2", "y": 8.0},
    ]
    ball_rows = [
        {"cross_id": "c1", "y": -9.0},
        {"cross_id": "c2", "y": 9.0},
    ]

    _normalize_flank(cross_rows, track_rows, ball_rows)

    assert track_rows[0]["y"] == 8.0
    assert track_rows[1]["y"] == 8.0
    assert ball_rows[0]["y"] == 9.0
    assert ball_rows[1]["y"] == 9.0
