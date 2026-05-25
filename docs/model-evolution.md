# xCross — model evolution log

One continuous record of how the model developed, results-first. Read top (origin) to bottom
(latest); new iterations are appended at the end. It tracks what changed at the model/feature level
and how the numbers moved — not code changes.

- Every metric is defined in [`metrics.md`](metrics.md). Current numbers come from
  `artifacts/reports/metrics/` (`uv run python -m xcross.model.report`).

## Timeline at a glance

| Milestone | What changed | xCross AUC | xCross stability | xCrossOT AUC | Calibration |
|---|---|---|---|---|---|
| **Spatial-situation scoring** | score the cross's spatial configuration — positional entropy + pitch control — instead of its raw outcome; two views, xCross (creation) and xCrossOT (danger) | 0.62\* | not measured | 0.74\* | breaks above ~0.7 |
| **Leakage-free validation & scale** | out-of-fold grouped by match, nested calibration, three leagues, and the first proof the ranking reproduces | 0.57 | 0.76 | 0.78 | ECE < 0.01 |
| **Feature expansion** *(current, 2026-05)* | add spatial-configuration, ball-flight (`z`) and temporal features | 0.57 | **0.78** | **0.81** | ECE < 0.01 |

\* The earliest AUCs were measured on a smaller, single-league dataset with a single train/test split
(and match leakage), so they are **not** directly comparable to the later numbers — the xCross drop
(0.62 → 0.57) is the leakage being removed, i.e. a *more honest* number, not the model getting worse.

**The throughline.** xCross has always scored the *spatial situation* of a cross rather than its raw
outcome, split into a stable **creation** view (xCross) and a discriminating **danger** view (xCrossOT).
Everything since has strengthened that same idea on its own terms: first making it trustworthy and
bigger (no leakage, robust calibration, three leagues) and proving the ranking is reproducible, then
widening what the model sees (the configuration around the cross, the ball's 3D flight, and how the
scene changes over the flight). Each step keeps creation stable and danger discriminating.

---

## 1 · Spatial-situation scoring (origin)

The founding idea: reduce a cross to its **spatial configuration** — positional entropy (how organised
the box is) and pitch control (who owns the space) — and score that instead of labelling
success/failure. It established the two-model view (**xCrossOT** danger > **xCross** creation on AUC),
the destination features that dominate xCrossOT, and the player archetypes. Open questions it left:
the score was validated on a single split (context leaked between train and test), calibration broke
down above ~0.7, it covered one league, and the player ranking was never tested for reproducibility.

## 2 · Leakage-free validation & scale

The idea was unchanged; **how it was trained, validated and evidenced** changed. The essentials:

- **No more leakage.** Validation moved from a single 80/20 split to **5-fold out-of-fold prediction
  grouped by `match_id`**, so every cross is scored by a model that never saw its match. This is why
  xCross AUC settled at 0.57 (the inflated part was leakage) — more honest, not worse.
- **Calibration fixed.** Nesting calibration inside each fold makes the probabilities well-calibrated
  across the whole range (ECE < 0.01), curing the collapse above ~0.7.
- **Scale.** 2,448 crosses (one league) → **7,510 across three leagues** (Premier League, Brasileirão,
  Bundesliga); base success rate 23% → 35%.
- **Reproducibility proven for the first time.** The **raw success rate is noise** (split-half 0.06),
  while **xCross recovers a stable ranking (0.76)** that also predicts future success ~3× better than a
  player's own past rate — the core justification for the model. An entropy ablation showed entropy
  *helps but is not dominant*, concentrating in xCrossOT.
- **xCrossOT AUC reached 0.78** with the extra data; danger > creation held.

## 3 · Feature expansion (current)

Until now the model scored each cross from two map families only — **positional entropy** and **pitch
control** — plus the ball's start/end geometry. This step adds three families of features that the
tracking data supported but the model never read, keeping each block only if it improved the metric it
was meant to (stability for the *creation* model, AUC for the *danger* model) **without hurting
calibration**. Blocks that did not pass that bar were dropped.

### What changed (feature level)

**Added to xCross (the moment of the cross — "creation"):**
- **Crosser pressure** — distance from the crosser to the nearest defender when he strikes the ball.
- **Marking tightness** — how free the box attackers are (nearest-defender distance, count of unmarked attackers).
- **Largest free pocket** — radius and goal-angle of the biggest defender-free space in the box.
- **Coverage mismatch (KL divergence)** — how poorly the defenders' distribution covers the attackers' (a *relational* term the per-team entropies miss).
- **Goalkeeper geometry** — keeper distance off his line, distance to the ball, lateral shift.
- **Defensive shape** — line height, block width/area, gap between the last line and the keeper.

**Added to xCrossOT (adds the delivery — "danger"), built from the previously unused ball `z`:**
- **Ball flight** — apex height, launch/descent angle, hang time, loftiness, 3D pace, bounces.
- **Vertical clearance** — whether the ball clears the defenders' and the keeper's reach as it passes them.
- **Swing / cutback** — in- vs out-swinger curl and cutback detection.
- **Second-ball support** — attackers/defenders positioned around where the ball lands.
- **Temporal deltas** — how the box occupancy and the arrival-zone entropy/pitch-control change over the flight.

**Tried and dropped (with the reason — the informative negatives):**
- **Run dynamics** (instantaneous player velocity at the cross): raised AUC but *destroyed* per-player
  stability (xCross/success split-half −0.10). It is a property of the individual play, not a
  reproducible trait of the crosser — dropping it confirms the creation-vs-noise idea.
- **Free-pocket location (x, y)**: positional noise, zero contribution.
- **Arrival-height bins and contact-point threat**: zero marginal contribution — fully redundant with
  the ball-flight features and the destination geometry already present.

### Results — before vs after this step

Out-of-fold, ≈7,510 crosses / 776 matches, headline model per target (AdaBoost).

| Model | Target | AUC | Stability (split-half) | ICC | ECE |
|---|---|---|---|---|---|
| xCross | `success` | 0.567 → **0.571** | 0.763 → **0.776** | 0.133 → 0.136 | 0.009 → 0.010 |
| xCross | `shot` | 0.568 → 0.567 | 0.681 → 0.659 | 0.121 → 0.118 | 0.007 → 0.009 |
| xCrossOT | `success` | 0.777 → **0.810** | 0.303 → 0.280 | 0.033 → 0.029 | 0.009 → **0.006** |
| xCrossOT | `shot` | 0.705 → **0.734** | 0.425 → 0.357 | 0.039 → 0.033 | 0.009 → 0.009 |

### What the numbers say

- **xCross (the player-ranking metric) improved on its own axis.** Split-half stability rose
  0.763 → 0.776 and AUC 0.567 → 0.571 — each block kept was selected for reproducibility, and the new
  geometry (pressure, keeper, shape) reinforces it without widening the (deliberately compressed)
  probability spread.
- **xCrossOT gained real discrimination from the ball's `z`.** AUC rose 0.777 → 0.810 (`success`) and
  0.705 → 0.734 (`shot`), with calibration unchanged or better. The flight features sharpen the
  probability distribution (more confident, accurate near-0 predictions for clearly failed deliveries).
  Its lower stability is expected and acceptable: xCrossOT is the *execution/danger* score, where
  discrimination — not reproducibility — is the goal.
- **Each model got better at its own job**, with no calibration regression anywhere.
- **One soft spot:** xCross/`shot` stability dipped slightly (0.681 → 0.659). The kept blocks were
  selected for the primary `success` target; on the secondary `shot` target a few are mildly negative,
  and the feature set is shared across labels.

### Which new features the models actually use

- **xCross/success:** `pressure_crosser_nearest_def` is the **single most important feature** — pressure
  on the crosser is the strongest creation signal — ahead of far-post attacking-shape entropy and
  central-box pitch control, with `gk_ball_distance` and `shape_last_line_to_gk_gap` contributing.
- **xCrossOT/success:** behind arrival-zone entropy, `clearance_over_keeper` is the **2nd most
  important feature** and `flight_pace_3d` the 5th — the ball `z`, previously idle, is now front-line
  signal — followed by `temporal_entropy_diff_zone_delta`.

### How it was evaluated

Each feature block carries a name prefix, and a block-ablation (`uv run python -m xcross.model.ablation`,
output in `ablation_blocks_*.csv` / `ablation_blocks_delta_*.csv`) measures, per block, the marginal
effect of removing it (full vs. leave-one-block-out) and the total effect of the whole step (full vs.
the previous feature set), on AUC, random and temporal stability, ICC and calibration. Blocks were kept
or pruned on that evidence.

## 4 · Influence-field probe (no change adopted)

A representation probe, not a feature step. Since the start, the entropy map turns each player into a
density by **projecting them 1 s ahead along their velocity** (`pos + v·t`) and smoothing with an
isotropic gaussian KDE. The question: is that deterministic projection injecting per-play velocity
noise — the same kind that made the `dynamics` block raise AUC but wreck stability — into the map?

The occupancy field was made pluggable (so only the density model changes; pitch-control held fixed)
and three alternatives were scored against the same régua, out-of-fold over the full set:

- **static** — drop the projection, smooth the current position;
- **anisotropic** — keep the position but turn velocity into *uncertainty*: a gaussian stretched along
  the direction of motion (spread grows with speed) instead of a displaced centre;
- **voronoi** — a different paradigm: hard winner-take-all assignment, each cell taking only its
  *nearest* player's kernel instead of the additive sum (no KDE pile-up in crowded areas).

| Model | Target | random split-half | **temporal split-half** | ICC |
|---|---|---|---|---|
| projected (current) | xCross `success` | 0.776 | **0.710** | 0.136 |
| static | xCross `success` | 0.806 (+0.030) | 0.655 (−0.055) | 0.140 |
| anisotropic | xCross `success` | 0.714 (−0.061) | 0.636 (−0.074) | 0.128 |
| voronoi | xCross `success` | 0.756 (−0.020) | 0.690 (−0.020) | 0.141 (+0.005) |

**Result: negative — the projection carries signal, not noise.** `static` inflated the *random*
split-half (+0.030) but the *temporal* split — the stricter "does early-season crossing predict
late-season?" test — **regressed** (−0.055), with ICC flat; the apparent gain does not persist across
the season. `anisotropic` was worse on every reliability axis and even flipped sign between a
single-league pilot (+0.031) and the full set (−0.061). `voronoi` was the most balanced — the best
calibration and a marginally higher ICC (+0.005) — but its temporal split still slipped (−0.020), so
it too fell short of replacing the baseline; its one clean gain was on the secondary `shot` target
(temporal +0.007). xCrossOT AUC barely moved for any of them (±0.003). So the velocity projection is
a genuine, season-persistent part of the signal, and the model is left unchanged. The probe also
re-confirmed the project's core lesson — **a higher random split-half is not a better ranking**; only
the temporal split and ICC decide (cf. §2).

**A second batch (fast, uncalibrated screen — relative deltas only).** To be sure the bottleneck was
not just *these three* shapes, four more fields and one additive block were screened: `arrival` (a
pitch-control-style influence as density), `soft_voronoi` (a 50/50 blend of the summed-KDE and the
winner-take-all), `free_space` (entropy of the *gaps* instead of the players), `threat_weighted`
(presence reweighted by goal proximity), and a structural block (convex-hull + Delaunay shape
descriptors). On xCross `success`, against this screen's own baseline, only one moved the deciding
(deterministic) temporal split the right way:

| Field / block | temporal split-half (Δ vs screen baseline) |
|---|---|
| **soft_voronoi** | **+0.012** (also +ICC, +random, +AUC) |
| arrival | −0.052 |
| threat_weighted | −0.027 |
| structural (additive) | −0.021 |
| free_space | −0.10 |

`soft_voronoi` was the only positive across all eight variants tried — but the gain is small and
unconfirmed (uncalibrated), so it was not adopted either.

**Bottom line.** Across three rounds (eight density models), none beats the production field on the
temporal split for the player-ranking target. **Re-modelling the entropy map is not a lever.** This is
distinct from "entropy is useless": the entropy *feature* still carries signal (the no-entropy
ablation, concentrated in xCrossOT — §2/§3); the point is that its current representation is already
near-optimal and cannot be squeezed further by reshaping the field. The levers lie elsewhere —
discrimination features (e.g. ball height at the strike), label variance, and ranking shrinkage.

The experiment harness (the pluggable density fields and the out-of-fold comparison runner) is kept on
the `features/entropy-influence-field` branch for reproducibility; it is deliberately **not merged**,
since the probe is negative and production keeps the original field.
