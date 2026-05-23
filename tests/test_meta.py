from xcross.data.meta import MatchMeta, attack_direction

HOME, AWAY = 1, 2


def _meta(home_team_start_left):
    return MatchMeta(
        match_id="m", league="L", season="S", date="",
        home_team_id=HOME, home_team_name="H",
        away_team_id=AWAY, away_team_name="A",
        home_team_start_left=home_team_start_left, fps=30.0,
        pitch_length_m=105.0, pitch_width_m=68.0,
    )


def test_home_attacks_positive_x_when_starting_left_first_half():
    assert attack_direction(_meta(True), period=1, team_id=HOME) == 1


def test_home_starting_right_attacks_negative_x():
    assert attack_direction(_meta(False), period=1, team_id=HOME) == -1


def test_teams_switch_sides_in_second_half():
    assert attack_direction(_meta(True), period=2, team_id=HOME) == -1


def test_away_attacks_opposite_of_home():
    assert attack_direction(_meta(True), period=1, team_id=AWAY) == -1
    assert attack_direction(_meta(True), period=2, team_id=AWAY) == 1


def test_extra_time_follows_half_parity():
    # Period 3 mirrors period 1 (odd), period 4 mirrors period 2 (even).
    assert attack_direction(_meta(True), period=3, team_id=HOME) == 1
    assert attack_direction(_meta(True), period=4, team_id=HOME) == -1
