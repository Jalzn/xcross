# xCross

[![CI](https://github.com/Jalzn/xcross/actions/workflows/ci.yml/badge.svg)](https://github.com/Jalzn/xcross/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

**xCross** is a calibrated *expected-cross* model that scores the quality of crosses in
football using broadcast tracking data. Instead of labelling a cross as success/failure,
it outputs a **calibrated probability** that, aggregated per player, lets you rank who
delivers the best crosses.

This repository is the **continuous development of the xCross method** — not a one-off
release. The code, results and docs here always track the latest state of the model. See
[Evolution](#evolution) for how it developed.

## Quickstart

The results are committed, so you can explore the model **without the licensed tracking data**.
After [installing with `uv`](#install):

- **Browse the results** — every figure and metric is under
  [`artifacts/reports/`](artifacts/reports/): the per-player ranking with names
  ([`ranking_final_success.csv`](artifacts/reports/metrics/ranking_final_success.csv)), the
  out-of-fold score of every cross
  ([`oof_predictions.csv`](artifacts/reports/metrics/oof_predictions.csv)), and the full model
  comparison ([`comparison.csv`](artifacts/reports/metrics/comparison.csv)).
- **Run a pretrained model** on your own feature rows → [Pretrained models](#pretrained-models).
- **Rebuild everything from raw tracking** — the only step that needs the PFF data →
  [Pipeline](#pipeline).

## Two models

- **xCross** — uses only the moment of the cross (start position, context). Measures the
  *quality of the situation created*.
- **xCrossOT** — adds the **ball's destination** (where it arrived). Measures the *danger
  of the delivery*.

Both are trained as the matrix `{xCross, xCrossOT} × {success, shot} × {8 estimators} ×
{isotonic, sigmoid}` — the registry covers four families: linear (`logreg`), bagging
(`random_forest`), gradient boosting (`xgboost`, `adaboost`, `catboost`, `lightgbm`, `histgb`)
and a pretrained foundation model (`tabpfn`, run on a GPU studio and brought in as a benchmark).
Evaluation is on out-of-fold probabilities with `StratifiedGroupKFold` per match (the same match
never appears in both train and test). The final report selects the headline per target by the
metric that fits its job (xCross: temporal stability; xCrossOT: AUC), tie-broken by ECE; the
choice is not hardcoded — it comes from the comparison table
([`comparison.csv`](artifacts/reports/metrics/comparison.csv) /
[`table_comparison.png`](artifacts/reports/figures/table_comparison.png)) restricted to the
in-process registry (TabPFN appears as the benchmark ceiling, never as the production headline).

## How it works

Each cross is reduced to the **spatial configuration at the moment it is struck**. From the
players' positions and velocities (projected a moment ahead) we build per-cell maps over the
attacking half and summarise them over pitch regions into the model's features. Two maps
capture complementary ideas:

- **Positional entropy** — how spread out vs. clustered the players are (a proxy for how
  organised the box is).
- **Entropy difference (attack − defense)** — which side's shape dominates each area.
- **Pitch control** — which team is likely to reach each area first (territorial dominance).

![Spatial representation of one cross](docs/figures/spatial_representation.png)

The same instant, four views: raw tracking → positional entropy → attack-minus-defense
entropy → pitch control. Entropy reads *where the players are*; pitch control reads *who
owns the space*. The feature modules in [`xcross/features/`](xcross/features) turn these maps
into the scalar inputs the models consume (see [`docs/metrics.md`](docs/metrics.md) for the
full feature list).

Beyond the two maps, the model reads the **configuration around the cross**: pressure on the
crosser, marking tightness and the largest free pocket in the box, the attack-vs-defense
**coverage mismatch** (KL divergence between the two teams' distributions), goalkeeper geometry
and the defensive block's shape. **xCrossOT** additionally reads the **ball's 3D flight** from its
`z` trajectory — apex, launch/descent angle, hang time, loft, 3D pace — whether the ball clears
the defenders and the keeper, the cross's swing/cutback, second-ball support around the landing,
and how the box and arrival zone change over the flight. How this feature set was built and
ablated, with the before/after numbers, is logged in
[`docs/model-evolution.md`](docs/model-evolution.md).

**Which features carry the signal?** The final models' importances confirm both maps matter,
and show *where* each acts:

| xCross (creation quality) | xCrossOT (delivery danger) |
|---|---|
| ![Feature importance, xCross](docs/figures/chart_importance_xcross_success.png) | ![Feature importance, xCrossOT](docs/figures/chart_importance_xcrossot_success.png) |

For **xCross**, **far-post attacking-shape entropy** (`entropy_attack_in_second_post`) is the single
strongest feature, ahead of central-box attacking entropy (`entropy_attack_in_center_box`), with
goalkeeper geometry (`gk_lateral_speed`, `gk_ball_distance`) contributing. For **xCrossOT** the
arrival-zone entropy (`entropy_attack_in_zone`) leads, with **whether the ball clears the keeper**
(`clearance_over_keeper`), the arrival location (`end_y`) and the **pitch control at the landing
zone** (`pitch_control_in_zone`) all in the top five — the ball's `z`, previously unused, is now
front-line signal.

## Results

> The figures below are produced by `uv run python -m xcross.model.report` and copied into
> [`docs/figures/`](docs/figures/) (see that folder's README). The full set and the meaning
> of every metric are documented in [`docs/metrics.md`](docs/metrics.md).

Everything below is measured on **≈11,700 crosses from 1,183 matches** across four competitions
(Brasileirão 2023, Premier League 2023–24 and 2024–25, Champions League 2023–24, Bundesliga
2025–26), on out-of-fold probabilities.

**Final models** (AdaBoost for the xCross targets, XGBoost for xCrossOT `success`, HistGB for
xCrossOT `shot`; selected per-objective from the comparison table — full metric definitions in
[`docs/metrics.md`](docs/metrics.md)):

| Model | Target | AUC | ECE | Stability | ICC |
|---|---|---|---|---|---|
| xCross | `success` | 0.58 | 0.013 | **0.73** | 0.13 |
| xCross | `shot` | 0.58 | 0.007 | 0.72 | 0.14 |
| xCrossOT | `success` | **0.84** | 0.033 | 0.29 | 0.02 |
| xCrossOT | `shot` | **0.77** | 0.016 | 0.26 | 0.03 |

xCrossOT discriminates best (AUC up to 0.84) while xCross gives the more reproducible ranking
(stability 0.73 vs 0.29) — the core trade-off charted below.

**Are the probabilities calibrated?** Both models track the diagonal across the range and ECE
stays under 0.01:

![Reliability curves — all estimators, xCross and xCrossOT](docs/figures/chart_calibration_compare_success.png)

**And do they order crosses?** The complementary view — the actual success rate per
predicted-probability decile. The black line (mean predicted) hugging the bars re-confirms the
calibration above; the bars climbing left-to-right show the *ordering*. The gap between the two
models is the discrimination difference made visual: xCross lifts the success rate from ~23% to
~43% across deciles, xCrossOT from ~0% to ~82% (AUC 0.58 vs 0.84):

| xCross | xCrossOT |
|---|---|
| ![Lift, xCross success](docs/figures/lift_xcross_success.png) | ![Lift, xCrossOT success](docs/figures/lift_xcrossot_success.png) |

**Does it beat the raw cross-success rate?** Ranking players by their raw conversion rate
reproduces almost nothing; scoring the *situation* recovers a stable ranking that holds even
under the stricter chronological split:

| Ranking by | Stability (random halves) | Stability (temporal) | ICC |
|---|---|---|---|
| Raw success rate | 0.05 | −0.01 | 0.01 |
| **xCross** | **0.73** | **0.66** | **0.13** |
| xCrossOT | 0.35 | 0.20 | 0.02 |

**Player ranking by xCross (creation quality), with 95% CI** — left, crosses that end in a
successful shot (`success`); right, crosses that end in any shot (`shot`):

| `success` | `shot` |
|---|---|
| ![Top players by xCross, success target](docs/figures/chart_ranking_top_success.png) | ![Top players by xCross, shot target](docs/figures/chart_ranking_top_shot.png) |

**Context vs. outcome — xCross × xCrossOT per player:**

| `success` | `shot` |
|---|---|
| ![xCross vs xCrossOT quadrants, success target](docs/figures/chart_ranking_quadrants_success.png) | ![xCross vs xCrossOT quadrants, shot target](docs/figures/chart_ranking_quadrants_shot.png) |

**The core trade-off — no model is both highly discriminative and highly reproducible:**

![Discrimination vs reproducibility — by model family, xCross & xCrossOT](docs/figures/chart_model_tradeoff_success.png)

**Generalisation — does it hold across leagues and positions?** Metrics are stable
league-to-league (Brasileirão, Premier League, Champions League), and the per-position xCross
distribution behaves as expected — central midfielders and attacking mids score highest,
wingers and full-backs lower:

| By league | By position |
|---|---|
| ![Metrics per league](docs/figures/chart_by_league_success.png) | ![xCross by position](docs/figures/chart_by_position_success.png) |

**How many players can we rank?** Stability climbs with sample size — ~0.62 at ≥10 crosses,
~0.81 at ≥50 — so the ranking uses a **≥20-cross** cut-off. That bar admits only ~13% of
crossers, though they account for ~55% of all crosses:

![Stability vs minimum crosses](docs/figures/chart_stability_vs_n_success.png)

## Evolution

xCross is developed in the open. Its **development as one continuous flow — from its origin to today,
each milestone and its measured effect on the results** — is tracked in
[`docs/model-evolution.md`](docs/model-evolution.md).

The **first paper we wrote for this model** is [`papers/xcross-v1.pdf`](papers/xcross-v1.pdf). The
model has moved on substantially since then — leakage-free out-of-fold validation, robust calibration,
four competitions, and the expanded feature set above — all captured in the evolution log.

Two pieces of evidence underpin the current results: the player ranking is **reproducible** (ranking by
the raw success rate is noise at stability 0.05, while xCross recovers a stable 0.73), and the
**entropy features help but are not dominant** (removing them costs ~0.005–0.010 AUC):

| Ranking reproducibility — raw rate vs. the models | Entropy ablation — with vs. without |
|---|---|
| ![reliability](docs/figures/chart_reliability_success.png) | ![ablation](docs/figures/chart_ablation_success.png) |

## Data

This project uses commercial broadcast-tracking data from **PFF FC** — it is **licensed, not
publicly downloadable, and not redistributed here**. You must obtain it yourself and place it
under `data/raw/` (the whole `data/` directory is git-ignored).

Expected layout, mirrored on both sides:

```
data/
  raw/        # PFF originals, read-only for the pipeline
    <league>/<season>/<match_id>/
      <match_id>.jsonl.bz2   # one JSON line per tracking frame (~25 fps)
      metadata.json          # teams, date, pitch dimensions, fps, attack side
      rosters.json           # line-ups: player_id, team_id, shirt number, position
  processed/  # parquet tables generated by the pipeline (regenerable)
  features/   # one row per kept cross, with all features (regenerable)
```

See [`docs/data.md`](docs/data.md) for the full description of each table.

## Install

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # create the environment from uv.lock
```

## Pipeline

Run the stages in order. Each stage is incremental: it hashes the relevant code + config
and skips matches whose outputs are already up to date.

```bash
# 1. raw PFF  ->  data/processed/  (5 parquet tables per match)
uv run python -m xcross.data.build

# 2. processed  ->  data/features/  (one row per cross, all features)
uv run python -m xcross.features.build

# 3. train the full matrix, write artifacts/reports/metrics/comparison.csv
uv run python -m xcross.model.compare

# 4. final report: pick the best model per target, write metrics, figures and
#    the player ranking to artifacts/reports/
uv run python -m xcross.model.report

# 5. serialize the headline model per target (calibrated) to artifacts/models/,
#    so it can be loaded for inference without the licensed data
uv run python -m xcross.model.export
```

Useful flags for stage 1:

```bash
uv run python -m xcross.data.build --leagues premier-league
uv run python -m xcross.data.build --matches 31995 --rebuild
```

## Pretrained models

The four headline models — `{xCross, xCrossOT} × {success, shot}` — are committed under
[`artifacts/models/`](artifacts/models/) as calibrated scikit-learn estimators (`joblib`),
each refit on all crosses with the exact `(estimator, calibration)` the report selected
(see [`model_metrics.csv`](artifacts/reports/metrics/model_metrics.csv)). They store only
split thresholds and leaf weights — **no tracking data is redistributed**.
[`metadata.json`](artifacts/models/metadata.json) lists, per model, the estimator, the
expected feature columns *in order*, and the library versions they were trained with (match
them with `uv sync` so the pickles load).

```python
import joblib, json

meta = json.load(open("artifacts/models/metadata.json"))["models"]["xcross_success"]
model = joblib.load("artifacts/models/xcross_success.joblib")

# `features` is one feature row per cross (run the feature pipeline on your own tracking
# data); select the expected columns, in the metadata order, before predicting.
prob = model.predict_proba(features.select(meta["feature_names"]).to_numpy())[:, 1]
```

To inspect scores **without** running anything, the committed
[`oof_predictions.csv`](artifacts/reports/metrics/oof_predictions.csv) already holds the
out-of-fold probability of every cross from all four models.

> **xCrossOT caveat.** xCrossOT's features describe the ball's destination and the
> configuration at the *end* of the cross window — information realised as the outcome is
> decided. Its higher AUC reflects that conditioning: read it as a *descriptive danger*
> score given where the ball arrived, **not** as a forward prediction from the moment of the
> cross. For prediction and the player ranking, use xCross.

## Project structure

```
xcross/
  config.py          # paths, pitch geometry, window and grid constants
  data/              # raw PFF -> processed parquet (extraction, IO, metadata)
  features/          # processed -> feature rows (spatial, grid, pitch control, ...)
  model/             # training (OOF + calibration), evaluation, comparison, report
docs/
  data.md            # data layout and how to obtain the PFF dataset
  metrics.md         # guide to every metric and figure the report produces
  model-evolution.md # results log: the model's development and each milestone's measured effect
  figures/           # curated result figures used in this README
papers/              # the first paper written for this model (xcross-v1.pdf)
artifacts/reports/   # committed final results: figures (PNGs) and metrics (CSVs)
artifacts/models/    # committed serialized models (joblib) + metadata.json
tests/               # pytest suite
```

## Evaluation

The model is judged on three axes: calibration (are the probabilities correct?),
discrimination (do they separate crosses?) and stability (is the player ranking
reproducible?). Every metric and figure produced by the report is documented in
[`docs/metrics.md`](docs/metrics.md).

## Tests

```bash
uv run pytest -q
```

## Citation

If you use this model or code, please cite it (see [`CITATION.cff`](CITATION.cff)) and the method
paper, [`papers/xcross-v1.pdf`](papers/xcross-v1.pdf):

> Ferreira, J. *xCross: a calibrated expected-cross model from football tracking data.*
> https://github.com/Jalzn/xcross

## License

Code is released under the [MIT License](LICENSE). The PFF tracking data is **not** covered
by this license and is **not** included in this repository.
