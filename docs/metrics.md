# Model evaluation — guide to the metrics and figures

**What the model is for:** the goal is not to classify a cross as 0/1, but to produce a
**calibrated probability** that, aggregated per player, lets us **evaluate the quality of
who crosses**. So we judge three things: is the probability **correct** (calibration)?
does it **separate** crosses (discrimination)? and is the player ranking **reproducible**
(stability)?

## Models

- **xCross** — uses only the moment of the cross (start position, context). Measures the
  *quality of the situation created*.
- **xCrossOT** — adds the ball's **destination** (where it arrived). Measures the *danger
  of the delivery*.
- Tabular estimators compared: **XGBoost**, **AdaBoost**, **CatBoost**.
- Calibration: **isotonic** or **sigmoid** (chosen by lowest ECE).
- Validation: **5-fold out-of-fold**, `StratifiedGroupKFold` by match (the same match never
  appears in both train and test). Every metric is measured on the out-of-fold
  probabilities (no leakage).

## Per-cross probability metrics

| metric | what it is | ideal |
|---|---|---|
| `auc` | ROC-AUC: probability of ranking a success above a failure | 1.0 (0.5 = random) |
| `auc_pr` | area under the precision-recall curve (sensitive to imbalance, ~35% positives) | 1.0 (baseline ≈ base rate) |
| `log_loss` | logarithmic error of the probabilities | lower |
| `brier` | mean squared error of the probability | lower |
| `brier_skill` | **Brier gain over always predicting the mean**. ≤0 = adds nothing | higher; >0 = useful |
| `ece` | Expected Calibration Error: mean gap between predicted probability and observed frequency | ~0 |
| `cal_slope` | slope of the logistic recalibration. <1 = over-confident, >1 = under-confident | **1.0** |
| `cal_intercept` | global calibration bias | **0.0** |

## Spread and ranking metrics (per player)

| metric | what it is | reading |
|---|---|---|
| `std`, `p5..p95`, `range_5_95` | spread of the **raw** per-cross probabilities | higher = separates the deliveries more |
| `player_discrimination` | standard deviation of the **per-player means** | higher = separates the players more |
| `stability` | **split-half**: Spearman correlation between the rankings of two random halves of each player's crosses | higher = reproducible ranking |
| `icc` | **Intraclass Correlation**: fraction of variance that is *between* players (skill) vs *within* (per-cross noise). The canonical version of stability | 0 = pure noise, 1 = pure skill |

> **Why a high `std` is not enough:** xCrossOT has a large `std` but low `stability`/`icc`
> — much of the spread comes from the **destination** (almost luck of the delivery), not
> from the player. To rank quality, `stability`/`icc` matter more than `std`.

## Ranking reliability & entropy ablation (final-results audit)

Two analyses that justify the model, written to `reliability_{label}.csv` and `ablation_{label}.csv`.

**`reliability_{label}.csv`** — compares three ways to rank players: the **raw success rate**, **xCross**
and **xCrossOT**. Columns:

| column | what it is |
|---|---|
| `stability_random` | split-half stability on two **random** halves of each player's crosses |
| `stability_temporal` | split-half but **chronological** (early vs late crosses by match date) — the stricter test: does early crossing predict later crossing? |
| `icc`, `player_discrimination` | as above |

> **The model's reason to exist:** the raw per-player success rate is essentially **noise**
> (`stability ≈ 0.06`) — ranking players by their cross conversion % reproduces nothing. By scoring the
> **situation** instead of the outcome, **xCross** recovers a **stable, reproducible** ranking
> (`stability ≈ 0.76`). This is the whole value of the expected-cross model over raw stats.

**`ablation_{label}.csv`** — each feature set run **with vs without the spatial-entropy block**
(`entropy_*`, ~half the features), same selected estimator/calibration. The metric gap is the
**contribution of the entropy features** (the proposed novelty). Columns: `feature_set`, `variant`
(`with_entropy`/`no_entropy`), `n_features`, and the key metrics (`auc`, `auc_pr`, `brier_skill`,
`stability`, `icc`, `player_discrimination`).

**Block ablation** (`ablation_blocks_{label}.csv` and `ablation_blocks_delta_{label}.csv`, from
`uv run python -m xcross.model.ablation`) — the generalisation of the entropy ablation to **every
feature block**: it scores the full feature set, the previous set (all new blocks removed) and each
leave-one-block-out, giving each block's marginal effect on AUC, random/temporal stability, ICC and
calibration. It is the evidence behind which feature blocks were kept or dropped — see
[`model-evolution.md`](model-evolution.md).

## Columns of `ranking_final_{success,shot}.csv` (one ranking per label)

| column | meaning |
|---|---|
| `n` | number of crosses by the player (minimum 20) |
| `xcross` | mean xCross — **creation quality** (stable) |
| `xcross_se`, `xcross_ci_low/high` | standard error and **95% CI** of the mean (uncertainty given `n`) |
| `xcrossot` | mean xCrossOT — outcome of the delivery |
| `execution` | `xcrossot − xcross` — how much the destination adds to the initial situation |
| `actual` | real rate of the label (success / shot) |
| `over_expected` | `actual − xcross` — **delivery above what the situation expected** (observed execution) |

`ranking_by_league_{label}.csv` has the same columns **plus `league`**: the ranking recomputed within
each league (player counts and `n` are per league), used by the per-league quadrant figure.

## Figures

Most figures are produced **for both labels** (`success`, `shot`) with a `_{label}` suffix, and a
few per **(feature_set, label)** with a `_{xcross,xcrossot}_{label}` suffix. The exceptions cover
both labels already and have no suffix: `table_model_metrics`, `table_comparison`, `chart_tradeoff`.

**Tables** (raw values rendered):
- `table_model_metrics.png` — the 4 final models (feature set × label) with the selected estimator.
- `table_comparison.png` — the matrix of the 24 runs (feature set × label × estimator × calibration).
- `table_league_metrics_{label}.png` — metrics per league (generalisation), for **both** xCross and
  xCrossOT.

**Charts (per label):**
- `chart_tradeoff.png` — AUC × stability of every run: none combines high discrimination **and** high
  reproducibility (colour = feature set, marker = estimator, light = sigmoid).
- `chart_stability_vs_n_{label}.png` — stability vs. minimum crosses (and how many players remain).
- `chart_ranking_top_{label}.png` — top 20 by xCross, with 95% CI.
- `chart_ranking_quadrants_{label}.png` — **xCross × xCrossOT per player**, split into quadrants by
  the medians (style of the paper's Fig. 15). The standout of each corner (best context, best
  outcome, best execution) is labelled with a leader line.
- `chart_ranking_quadrants_by_league_{label}.png` — the same quadrant view **per league** (one panel
  each), with quadrants split by **that league's own medians**. Only leagues with at least 6 qualified
  players (≥20 crosses) are shown.
- `chart_by_position_{label}.png` — distribution of xCross by position group (validates the ranking).
- `chart_by_league_{label}.png` — metrics per league, one panel per feature set (xCross | xCrossOT).
- `chart_reliability_{label}.png` — **raw rate vs xCross vs xCrossOT** on random/temporal stability and
  ICC. Shows the raw success rate is noise and xCross recovers a stable ranking.
- `chart_ablation_{label}.png` — each feature set **with vs without entropy**, one panel each; the bar
  gap is the entropy features' contribution.

**Per (feature_set, label):**
- `chart_importance_{xcross,xcrossot}_{label}.png` — feature importance of the **final model** (the
  one we actually use, chosen per target).
- `calibration_{xcross,xcrossot}_{label}.png` — left = calibration curve (predicted × observed,
  closer to the diagonal is better), right = histogram of predicted probabilities (the spread).
- `lift_{xcross,xcrossot}_{label}.png` — actual label rate per predicted-probability decile (ordering).
- `pdp_{xcross,xcrossot}_{label}.png` — **partial-dependence** grid of the **final model**: for the
  9 most important features, how the predicted probability moves as the feature varies (the
  **direction and shape** of the effect — monotonic, threshold, U-shape), with a rug showing where
  the real data lies. Complements the magnitude in `chart_importance_*`. Model-agnostic, so it works
  with AdaBoost (the fast tree SHAP explainer does not).

## Agreement between rankings
The `report.py` log reports the **Spearman correlation between the ranking by `success` and
by `shot`** (do the two readings of "good cross" point to the same players?).
