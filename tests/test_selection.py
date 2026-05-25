import polars as pl

from xcross.model.selection import select_best


def _comparison() -> pl.DataFrame:
    return pl.DataFrame({
        "feature_set": ["xcross", "xcross"],
        "label": ["success", "success"],
        "estimator": ["tabpfn", "adaboost"],
        "calibration": ["isotonic", "isotonic"],
        "stability_temporal": [0.90, 0.70],
        "stability": [0.50, 0.50],
        "ece": [0.01, 0.01],
    })


def test_xcross_picks_highest_temporal_stability():
    choice = select_best(_comparison(), "xcross", "success")
    assert choice["estimator"] == "tabpfn"


def test_xcrossot_picks_highest_auc_not_stability():
    comparison = pl.DataFrame({
        "feature_set": ["xcrossot", "xcrossot"],
        "label": ["success", "success"],
        "estimator": ["random_forest", "xgboost"],
        "calibration": ["isotonic", "isotonic"],
        "stability_temporal": [0.30, 0.18],
        "auc": [0.820, 0.844],
        "ece": [0.02, 0.01],
    })
    choice = select_best(comparison, "xcrossot", "success")
    assert choice["estimator"] == "xgboost"


def test_select_best_excludes_ineligible_estimators():
    choice = select_best(_comparison(), "xcross", "success", eligible={"adaboost", "xgboost"})
    assert choice["estimator"] == "adaboost"


def test_select_best_falls_back_to_stability_when_temporal_missing():
    comparison = _comparison().drop("stability_temporal")
    choice = select_best(comparison, "xcross", "success")
    assert choice["estimator"] in {"tabpfn", "adaboost"}


def test_select_best_falls_back_when_target_absent():
    choice = select_best(_comparison(), "xcrossot", "shot")
    assert choice == {"estimator": "adaboost", "calibration": "isotonic"}
