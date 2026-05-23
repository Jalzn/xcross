# xCross — evolution log

The version-to-version record of the xCross method: what changed between releases and what each
change uncovered (every version's PDF is in [`../papers/`](../papers/)). It currently covers **v1 → v2**.

Framed as a comparison with v1 ([`../papers/xcross-v1.pdf`](../papers/xcross-v1.pdf)): **v2 reproduces
v1's structure, fixes its methodology, and adds the validations v1 never ran — which both *confirm* the
xCross idea and *temper* v1's claims about "execution" and entropy.**

All v2 numbers come from `artifacts/reports/metrics/` (run `uv run python -m xcross.model.report`).
Every metric is defined in [`metrics.md`](metrics.md).

---

## 1. Setup

| | Paper | v2 |
|---|---|---|
| Crosses | 2,448 (562 successful) | 7,510 |
| Leagues / seasons | Premier League 2024 only | Premier League + Brasileirão + Bundesliga |
| Success rate (base) | 23% | 35% |
| Validation | single 80/20 stratified split | **5-fold out-of-fold, `StratifiedGroupKFold` by match** |
| Calibration | one `CalibratedClassifierCV` | nested calibration inside every training fold |
| Estimator | XGBoost | AdaBoost (selected for per-player stability) |

The two methodological changes matter most: **(a)** grouping folds by `match_id` means no cross from a
match used in training is ever scored in testing — the paper's single split mixes crosses from the same
match across train/test, leaking context; **(b)** out-of-fold prediction gives every cross a held-out
score, so the per-player ranking is built entirely on unseen predictions.

---

## 2. Discrimination (AUC)

| | Paper | v2 (success) |
|---|---|---|
| xCross | 0.62 | **0.57** |
| xCrossOT | 0.74 | **0.78** |

- **xCrossOT improved** (0.74 → 0.78) with 3× the data.
- **xCross dropped** (0.62 → 0.57) — but this is almost certainly *more honest*, not worse. xCross
  describes the situation at the moment of the cross, which is heavily match-determined; it is exactly
  the metric most inflated by the paper's match leakage. Removing that leakage lowers the number.

## 3. Calibration

The paper's xCross calibration curve (its Fig. 8) **collapses above ~0.7**. v2 is well calibrated
across the whole range (ECE < 0.01) because calibration happens *inside* each fold rather than on a
single split.

## 4. Features — strong agreement, especially for xCrossOT

- **xCrossOT** lines up closely. Paper (SHAP): `end_y`, `end_x`, `entropy_general_in_zone`,
  `polar_angle_cross`, `pitch_control_in_zone`. v2 (partial dependence): `end_y` (dominant in both),
  `pitch_control_in_zone`, `polar_angle_cross`, `entropy_attack_in_zone`. Same story.
- **xCross** agrees on the *kind* of feature (entropy + pitch control) but differs on the specific
  ones (paper: `entropy_diff_around`, `entropy_defense_around`; v2: `entropy_attack_in_second_post`,
  `pitch_control_in_center_box`, `distance_start_from_goal`) — expected with different data/season.
- v2 also replaced SHAP with **partial-dependence plots**: AdaBoost is not supported by SHAP's fast
  tree explainer, and PDP shows the *direction and shape* of each effect, not just its magnitude.

## 5. Player rankings

Direct name-by-name comparison is confounded — the paper is the 2024 PL season, v2's PL is 2024-2025,
so the player pool differs. Still, there is clear **role persistence**: **Pervis Estupiñán**,
**Dwight McNeil** and **Gabriel Martinelli** appear near the top in both. Kieran Trippier (the paper's
xCross #1) is absent from v2 — likely not in our season.

Absolute values are higher in v2 (xCross median 0.354 vs the paper's 0.267; top 0.415 vs 0.392)
because the **base rate is higher** (35% vs 23%) — xCross tracks the base rate.

---

## 6. The scientific advance: validations the paper never ran

This is the core of the comparison. The paper presents the ranking tables and the quadrant chart
(its Fig. 15) as conclusions, **without ever testing whether the ranking is reproducible**. v2 does.

### 6.1 The ranking *is* reproducible — and the raw stat is not

`reliability_success.csv`:

| ranking | split-half (random) | temporal (early vs late) | ICC |
|---|---|---|---|
| raw success rate | 0.06 | 0.07 | 0.01 |
| **xCross** | **0.76** | **0.70** | 0.13 |
| xCrossOT | 0.30 | 0.21 | 0.03 |

![Reliability](figures/chart_reliability_success.png)

Ranking players by their **raw cross-success rate is essentially noise** (0.06): it does not
reproduce across halves of their crosses. By scoring the *situation* instead of the *outcome*,
**xCross recovers a stable ranking (0.76)**. This is the whole justification for an expected-cross
model over raw stats — and it was missing from the paper.

### 6.2 Predictive validity (the gold-standard test, like xG → goals)

Using only each player's **earlier** crosses to predict the success rate of their **later** crosses:

| past predictor | → future actual success rate (Spearman) |
|---|---|
| past raw success rate | +0.065 |
| **past xCross** | **+0.210** |

xCross predicts future success **~3× better** than the player's own historical success rate does.
The model is not just self-consistent — it forecasts real future outcomes.

### 6.3 Entropy ablation — directly testing the paper's central claim

The paper's thesis is that **spatial entropy** is the key driver, supported only by SHAP rankings.
But SHAP importance is not causal contribution. v2 actually removes the entire entropy block
(`*_noent` feature sets, ~half the features) and measures the drop (`ablation_*.csv`):

| target | AUC (with → without entropy) | stability (with → without) |
|---|---|---|
| xCrossOT / success | 0.777 → 0.762 | **0.30 → 0.20** |
| xCrossOT / shot | 0.705 → 0.688 | **0.43 → 0.33** |
| xCross / success | 0.567 → 0.557 | 0.76 → 0.70 |
| xCross / shot | 0.568 → 0.559 | 0.68 → 0.68 |

![Entropy ablation](figures/chart_ablation_success.png)

Entropy **helps — confirming the paper's direction — but the effect is modest, not dominant**.
Dropping all 24-28 entropy features costs only ~0.01-0.015 AUC. The gain concentrates in
**xCrossOT** (arrival-zone entropy), where it lifts stability by ~0.10; for xCross it is small
(success) or roughly neutral (shot). Much of the entropy signal is redundant with the position and
pitch-control features. This tempers the paper's framing of entropy as *the* key factor.

### 6.4 What this means for the paper's narrative

- It **validates xCross**: the creation ranking is stable and predictive, with evidence the paper
  lacked.
- It **weakens the "execution" story**. The paper's Fig. 15 reads the gap `xCrossOT − xCross` as a
  player's *technical precision*. That gap is the `execution` / `over_expected` metric, which v2 shows
  is built on the noisy side (xCrossOT stability 0.30, and the raw-outcome component is ~0.06 noise).
  The execution quadrant rests on a signal the paper never verified.

---

## 7. Honest limitations (stated, not hidden)

- **Coverage.** Only **115 of 891 crossers (13%)** reach the ≥20-cross threshold needed to be ranked.
  They do account for **50% of all crosses**, but half the crosses come from occasional crossers we
  cannot evaluate. "Player evaluation" here means *regular* crossers.
- **Shallow gradient.** xCross spans only **0.320–0.415** across the 115 players (p10–p90 = 0.052)
  against a typical 95% CI half-width of 0.016. The extremes are distinguishable; the middle is a tight
  pack within noise. The order is reproducible, but the differences are a few percentage points.
- **Skill vs. role/team.** xCross uses teammate and defender positioning, so it measures the *quality
  of the crossing situations a player is involved in* (skill + role + team), not isolated individual
  skill. Its high stability partly reflects that players occupy consistent zones/roles.
- **`over_expected` / `execution` are noisy.** Because they depend on the raw outcome (~0.06 stability),
  they should not be read as fine-grained player rankings.

---

## 8. Bottom line

v2 reproduces what the paper got right (xCrossOT > xCross on AUC, the same destination features, the
same player archetypes), fixes the methodology (no leakage, robust calibration, larger and
multi-league data), and **adds the reproducibility, predictive-validity and ablation evidence the
paper did not have** — which simultaneously *support* the xCross idea and *expose* that the "execution"
reading and the weight placed on entropy were more fragile than the paper suggested.
