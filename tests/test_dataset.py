import polars as pl

from xcross.model.dataset import _feature_columns

# A frame with id/label cols, an original start-frame feature, an entropy feature, an original
# OT feature, and one new-block feature per relevant block (start-frame, OT, temporal).
_DF = pl.DataFrame({
    "cross_id": [1], "match_id": [1], "league": ["pl"], "season": ["x"],
    "crosser_player_id": [1], "crosser_team_id": [1], "success": [1], "shot_in_window": [0],
    "start_x": [40.0],
    "entropy_attack_sum": [1.0],
    "end_x": [50.0],                       # original OT (OT_ONLY)
    "pressure_crosser_nearest_def": [3.0], # new start-frame block
    "marking_free_attackers_in_box": [1],  # new start-frame block
    "flight_apex_height": [4.0],           # new OT block
    "temporal_defenders_in_box_delta": [1],# new OT block (destination-dependent)
})


def test_xcross_excludes_original_and_new_ot_blocks():
    cols = _feature_columns(_DF, "xcross")
    assert "end_x" not in cols          # OT_ONLY
    assert "flight_apex_height" not in cols
    assert "temporal_defenders_in_box_delta" not in cols
    assert "pressure_crosser_nearest_def" in cols  # start-frame block kept


def test_xcrossot_includes_ot_blocks():
    cols = _feature_columns(_DF, "xcrossot")
    assert "end_x" in cols
    assert "flight_apex_height" in cols
    assert "temporal_defenders_in_box_delta" in cols


def test_base_drops_all_new_blocks():
    cols = _feature_columns(_DF, "xcrossot__base")
    assert "pressure_crosser_nearest_def" not in cols
    assert "marking_free_attackers_in_box" not in cols
    assert "flight_apex_height" not in cols
    assert "start_x" in cols            # original features survive
    assert "end_x" in cols
    assert "entropy_attack_sum" in cols


def test_drop_one_block_only():
    cols = _feature_columns(_DF, "xcross__drop-pressure")
    assert "pressure_crosser_nearest_def" not in cols
    assert "marking_free_attackers_in_box" in cols  # other blocks untouched


def test_noent_still_drops_entropy():
    cols = _feature_columns(_DF, "xcross_noent")
    assert "entropy_attack_sum" not in cols
    assert "start_x" in cols


def test_noent_base_composes():
    cols = _feature_columns(_DF, "xcrossot_noent__base")
    assert "entropy_attack_sum" not in cols
    assert "flight_apex_height" not in cols
    assert "start_x" in cols
