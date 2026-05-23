import numpy as np
import polars as pl

from xcross.model.evaluate import (
    brier_skill_score,
    calibration_slope_intercept,
    combined_ranking,
    expected_calibration_error,
    icc,
    lift_by_decile,
    player_discrimination,
    player_ranking,
    rank_agreement,
    split_half_stability,
    spread,
    temporal_split_stability,
)


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    prob = np.full(2000, 0.3)
    y = (rng.random(2000) < 0.3).astype(int)
    assert expected_calibration_error(y, prob) < 0.05


def test_ece_high_when_miscalibrated():
    prob = np.full(1000, 0.9)  # claims 0.9...
    y = np.zeros(1000, dtype=int)  # ...but never happens
    assert expected_calibration_error(y, prob) > 0.8


def test_spread_percentiles_ordered():
    s = spread(np.linspace(0.0, 1.0, 100))
    assert s["p5"] < s["p50"] < s["p95"]
    assert abs(s["range_5_95"] - (s["p95"] - s["p5"])) < 1e-9


def test_player_ranking_filters_by_count_and_sorts_by_mean():
    pid = np.array([1] * 20 + [2] * 20 + [3] * 5)
    prob = np.concatenate([np.full(20, 0.8), np.full(20, 0.2), np.full(5, 0.9)])
    ranking = player_ranking(pid, prob, min_crosses=20)
    assert ranking["player_id"].to_list() == [1, 2]  # player 3 dropped (n<20), 1 above 2
    assert ranking["n"].to_list() == [20, 20]


def test_split_half_stability_high_when_player_trait_is_stable():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    levels = rng.random(30)
    prob = np.repeat(levels, 20) + rng.normal(0, 0.01, 600)  # each player ~ own level
    assert split_half_stability(pid, prob, min_crosses=20) > 0.8


def test_split_half_stability_low_when_pure_noise():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    prob = rng.random(600)  # no per-player signal
    assert abs(split_half_stability(pid, prob, min_crosses=20)) < 0.4


def test_temporal_stability_high_when_trait_persists_over_time():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    levels = rng.random(30)
    prob = np.repeat(levels, 20) + rng.normal(0, 0.01, 600)  # stable across the whole timeline
    order_key = np.tile(np.arange(20), 30)  # each player's 20 crosses span the same time range
    assert temporal_split_stability(pid, prob, order_key, min_crosses=20) > 0.8


def test_temporal_stability_low_when_trait_drifts():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    order_key = np.tile(np.arange(20), 30)
    early_mask = np.tile(np.arange(20) < 10, 30)
    early = np.repeat(rng.random(30), 20)  # each player's early level ...
    late = np.repeat(rng.random(30), 20)   # ... unrelated to their late level
    prob = np.where(early_mask, early, late)
    assert abs(temporal_split_stability(pid, prob, order_key, min_crosses=20)) < 0.4


def test_brier_skill_positive_when_better_than_baserate():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    prob = np.where(y == 1, 0.7, 0.15)  # informative
    assert brier_skill_score(y, prob) > 0.3


def test_calibration_slope_near_one_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # perfectly calibrated by construction
    slope, intercept = calibration_slope_intercept(y, p)
    assert 0.8 < slope < 1.2
    assert abs(intercept) < 0.2


def test_player_discrimination_positive_with_distinct_levels():
    pid = np.repeat(np.arange(10), 20)
    prob = np.repeat(np.linspace(0.1, 0.9, 10), 20)
    assert player_discrimination(pid, prob, min_crosses=20) > 0.1


def test_lift_increases_for_informative_model():
    rng = np.random.default_rng(0)
    prob = rng.random(1000)
    y = (rng.random(1000) < prob).astype(int)
    rates = lift_by_decile(y, prob, n_bins=5)["actual_rate"].to_list()
    assert rates[0] < rates[-1]


def test_combined_ranking_execution_and_over_expected():
    pid = np.repeat([1, 2], 20)
    xc = np.repeat([0.3, 0.4], 20)
    xo = np.repeat([0.5, 0.45], 20)
    y = np.repeat([0, 1], 20).astype(int)
    r = combined_ranking(pid, xc, xo, y, min_crosses=20)
    assert {"xcross", "xcrossot", "execution", "actual", "over_expected", "xcross_ci_low"} <= set(r.columns)
    row = r.filter(pl.col("player_id") == 1).row(0, named=True)
    assert abs(row["execution"] - (0.5 - 0.3)) < 1e-9
    assert abs(row["over_expected"] - (0.0 - 0.3)) < 1e-9


def test_icc_high_with_player_signal():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    prob = np.repeat(rng.random(30), 20) + rng.normal(0, 0.02, 600)
    assert icc(pid, prob, min_crosses=20) > 0.7


def test_icc_low_with_pure_noise():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 20)
    assert icc(pid, rng.random(600), min_crosses=20) < 0.3


def test_rank_agreement_high_when_rankings_match():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(20), 20)
    levels = np.repeat(np.linspace(0.1, 0.9, 20), 20)
    assert rank_agreement(pid, levels, levels + rng.normal(0, 0.01, 400), min_crosses=20) > 0.8
