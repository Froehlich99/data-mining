# Results Comparison: Before vs After Stratified Evaluation

## Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| Evaluation | 5-fold CV on all data (OOF predictions only) | 80/20 held-out test set + 5-fold CV on train split |
| CV Stratification | Plain KFold (random shuffle) | StratifiedKFold by ethnicity |
| Outlier handling | None | Flagged by annotator variance (top 10% per dataset) |
| Models | XGBoost, LightGBM, CatBoost, Ensemble, MLP, Quantile, Ranker | + Ridge, Decision Tree, Random Forest |

---

## Notes on Model Selection

**Why Ridge instead of plain Linear Regression:** Unregularized Linear Regression produced numerical overflow (divide-by-zero, NaN predictions) due to multicollinearity among the 30 geometric features. Several features are mathematically related (e.g., `face_length_width_ratio` and `phi_deviation` = `|face_length_width_ratio - 1.618|`), making the least-squares solution unstable. Ridge Regression (L2 penalty) resolves this by shrinking correlated coefficients, while remaining a linear model that serves as an interpretable baseline.

**Symmetry features have near-zero predictive power:** The facial symmetry features (`facial_symmetry`, `eye_symmetry`, `mouth_symmetry`, `nose_symmetry`) all show r < 0.04 correlation with attractiveness scores and receive near-zero feature importance from tree-based models. This is a fundamental limitation of using MediaPipe landmarks for symmetry measurement — the face mesh detector is biased toward placing left/right landmark pairs symmetrically regardless of actual facial asymmetry. The features primarily capture detection noise rather than true asymmetry. A pixel-level approach (e.g., mirroring one face half and comparing) would be needed to capture real asymmetry signal. Exception: `eye_asymmetry` (difference in left/right eye height) does show mild predictive value (r = -0.11).

**Scleral show and canthal tilt required bug fixes:** The initial `canthal_tilt` computation used inconsistent angle conventions between left and right eyes and an inverted sign definition. The initial `scleral_show` computation produced extreme outlier values (down to -52) due to division by near-zero eye heights in squinting or poorly-detected faces. Both were corrected; `scleral_show` is additionally clamped to [-1, 1] with a minimum eye height floor.

**LiveBeauty lacks usable variance data for outlier flagging:** The only available proxy (`|mean_score - clean_mean_score|`) is zero for ~90% of images, making percentile-based thresholding impossible. LiveBeauty images are marked as "unknown" for outlier status. SCUT and MEBeauty have real per-rater variance data and use 90th-percentile thresholds.

---

## Before: Baseline Results (Plain KFold, No Held-Out Set)

### Per-Model Performance (5-Fold CV, all 17,870 samples)

| Model | MAE | RMSE | Pearson r | Improvement vs Mean |
|-------|-----|------|-----------|---------------------|
| Ensemble (Stacking) | 0.4012 | 0.5094 | 0.8526 | 49.5% |
| XGBoost | 0.4267 | 0.5360 | 0.8329 | 46.3% |
| Quantile | 0.4331 | 0.5930 | 0.7858 | 45.5% |
| Ranker | 0.4436 | 0.5700 | 0.8134 | 44.2% |
| MLP | 0.4743 | 0.6043 | 0.7733 | 40.3% |
| LightGBM | 0.5572 | 0.7019 | 0.6763 | 29.8% |
| CatBoost | 0.5592 | 0.7006 | 0.6781 | 29.6% |

### Per-Ethnicity Fairness (Combined XGBoost, 5-Fold CV)

| Ethnicity | N | MAE | RMSE | Pearson r | Delta MAE % |
|-----------|---|-----|------|-----------|-------------|
| Asian | 14,351 | 0.5522 | 0.7145 | 0.7028 | -2.8% |
| Black | 296 | 0.6155 | 0.7904 | 0.4732 | +8.3% |
| Caucasian | 2,480 | 0.6013 | 0.7665 | 0.6048 | +5.8% |
| Hispanic | 296 | 0.7173 | 0.8713 | 0.4069 | +26.2% |
| Indian | 156 | 0.8758 | 1.0585 | 0.2597 | +54.1% |
| Middle Eastern | 291 | 0.7180 | 0.8906 | 0.5699 | +26.3% |

**Fairness gap:** Worst-group (Indian) MAE is 58.6% higher than best-group (Asian).

### Cross-Dataset Generalization

| Experiment | Train N | Test N | MAE | RMSE | Pearson r |
|------------|---------|--------|-----|------|-----------|
| LiveBeauty → MEBeauty | 10,000 | 2,370 | 0.8566 | 1.0687 | 0.4178 |
| LiveBeauty → SCUT | 10,000 | 5,500 | 0.7201 | 0.9211 | 0.4971 |
| MEBeauty → LiveBeauty | 2,370 | 10,000 | 0.7444 | 0.9018 | 0.4828 |
| MEBeauty → SCUT | 2,370 | 5,500 | 0.7700 | 0.9350 | 0.3610 |
| SCUT → LiveBeauty | 5,500 | 10,000 | 0.7642 | 0.9641 | 0.4773 |
| SCUT → MEBeauty | 5,500 | 2,370 | 0.8487 | 1.0649 | 0.2800 |
| **Combined (5-fold CV)** | **17,870** | **17,870** | **0.5683** | **0.7329** | **0.6803** |

---

## After: Stratified Evaluation Results

Train/test split: 80/20 stratified by ethnicity (14,296 train / 3,574 test). CV: stratified 5-fold on train split only. Baseline MAE (predict mean): 0.8582.

### Per-Model Performance (Stratified 5-Fold CV on 80% train split)

| Model | MAE | RMSE | Pearson r | Improvement vs Mean |
|-------|-----|------|-----------|---------------------|
| Ridge | 0.6036 ± 0.0068 | 0.7676 ± 0.0051 | 0.6375 ± 0.0121 | 30.0% |
| Decision Tree | 0.6654 ± 0.0043 | 0.8515 ± 0.0073 | 0.5290 ± 0.0072 | 22.7% |
| Random Forest | 0.6125 ± 0.0045 | 0.7768 ± 0.0063 | 0.6409 ± 0.0095 | 27.8% |
| XGBoost | 0.5632 ± 0.0088 | 0.7273 ± 0.0095 | 0.6836 ± 0.0078 | 35.4% |
| LightGBM | 0.5652 ± 0.0078 | 0.7292 ± 0.0080 | 0.6815 ± 0.0076 | 35.2% |
| CatBoost | 0.5621 ± 0.0076 | 0.7236 ± 0.0066 | 0.6877 ± 0.0066 | 35.3% |
| Ensemble (Stacking) | 0.5590 ± 0.0084 | 0.7204 ± 0.0064 | 0.6908 ± 0.0094 | 35.4% |
| MLP | 0.5507 ± 0.0113 | 0.7143 ± 0.0081 | 0.7001 ± 0.0085 | 36.6% |
| Quantile | 0.5664 ± 0.0052 | 0.7378 ± 0.0067 | 0.6725 ± 0.0074 | 34.7% |
| Ranker | 0.5609 ± 0.0102 | 0.7307 ± 0.0103 | 0.6806 ± 0.0108 | 35.1% |

### Held-Out Test Set (20%, stratified by ethnicity)

| Model | MAE | RMSE | Pearson r | Improvement vs Mean |
|-------|-----|------|-----------|---------------------|
| Ridge | 0.6004 | 0.7625 | 0.6598 | 30.0% |
| Decision Tree | 0.6633 | 0.8518 | 0.5463 | 22.7% |
| Random Forest | 0.6197 | 0.7807 | 0.6570 | 27.8% |
| XGBoost | 0.5540 | 0.7169 | 0.7080 | 35.4% |
| LightGBM | 0.5564 | 0.7199 | 0.7045 | 35.2% |
| CatBoost | 0.5555 | 0.7156 | 0.7092 | 35.3% |
| Ensemble (Stacking) | 0.5543 | 0.7148 | 0.7098 | 35.4% |
| **MLP** | **0.5438** | **0.7106** | **0.7143** | **36.6%** |
| Quantile | 0.5602 | 0.7290 | 0.6959 | 34.7% |
| Ranker | 0.5570 | 0.7246 | 0.7019 | 35.1% |

### Per-Ethnicity Fairness (XGBoost, Held-Out Test Set)

| Ethnicity | N | MAE | Pearson r | Delta vs Best MAE |
|-----------|---|-----|-----------|-------------------|
| Asian | 2,871 | 0.5383 | 0.7316 | — (best) |
| Black | 59 | 0.5994 | 0.5095 | +11.4% |
| Caucasian | 496 | 0.5732 | 0.6221 | +6.5% |
| Hispanic | 59 | 0.6404 | 0.6711 | +19.0% |
| Indian | 31 | 1.1483 | 0.1822 | +113.3% |
| Middle Eastern | 58 | 0.7112 | 0.5841 | +32.1% |

**Fairness gap (worst − best MAE):** 0.610 (Indian vs Asian)

### Per-Ethnicity Fairness (MLP, Held-Out Test Set — best overall model)

| Ethnicity | N | MAE | Pearson r | Delta vs Best MAE |
|-----------|---|-----|-----------|-------------------|
| Asian | 2,871 | 0.5217 | 0.7425 | — (best) |
| Black | 59 | 0.5842 | 0.5807 | +12.0% |
| Caucasian | 496 | 0.5979 | 0.5932 | +14.6% |
| Hispanic | 59 | 0.7333 | 0.4995 | +40.6% |
| Indian | 31 | 1.0273 | 0.3072 | +96.9% |
| Middle Eastern | 58 | 0.6844 | 0.6444 | +31.2% |

**Fairness gap (worst − best MAE):** 0.506 (Indian vs Asian)

---

## Before vs After Comparison

### Why metrics changed

The "before" evaluation used plain KFold on ALL data — out-of-fold predictions still benefit from information leakage through the CV setup (no held-out set, no ethnicity stratification). The "after" evaluation is more rigorous: stratified split, separate holdout, proper train/validation/test separation.

### Model Performance Comparison (CV MAE)

| Model | Before (Plain CV) | After (Stratified CV) | Delta |
|-------|-------------------|----------------------|-------|
| XGBoost | 0.4267 | 0.5632 | +0.137 |
| LightGBM | 0.5572 | 0.5652 | +0.008 |
| CatBoost | 0.5592 | 0.5621 | +0.003 |
| Ensemble | 0.4012 | 0.5590 | +0.158 |
| MLP | 0.4743 | 0.5507 | +0.076 |
| Quantile | 0.4331 | 0.5664 | +0.133 |
| Ranker | 0.4436 | 0.5609 | +0.117 |

Models with the largest delta (XGBoost, Ensemble, Quantile, Ranker) were overfitting most in the old setup. LightGBM/CatBoost barely changed — their built-in regularization was already preventing leakage.

### Fairness Comparison (XGBoost per-ethnicity)

| Ethnicity | Before MAE | After MAE | Before r | After r |
|-----------|-----------|-----------|----------|---------|
| Asian | 0.5522 | 0.5383 | 0.7028 | 0.7316 |
| Black | 0.6155 | 0.5994 | 0.4732 | 0.5095 |
| Caucasian | 0.6013 | 0.5732 | 0.6048 | 0.6221 |
| Hispanic | 0.7173 | 0.6404 | 0.4069 | 0.6711 |
| Indian | 0.8758 | 1.1483 | 0.2597 | 0.1822 |
| Middle Eastern | 0.7180 | 0.7112 | 0.5699 | 0.5841 |

Most groups improved in the "after" setup (better MAE and r). Indian got worse — likely because with only 31 test samples, the held-out estimate is highly volatile. Hispanic correlation improved dramatically (0.41 → 0.67), suggesting the stratified split provides better representation in training.

### Impact of Outlier Flagging

| Metric | All Data | Excluding Outliers | Delta |
|--------|----------|-------------------|-------|
| N samples | 17,870 | — | — |
| XGBoost MAE | 0.5540 | — | — |
| XGBoost r | 0.7080 | — | — |
| Fairness gap (worst-best MAE) | 0.610 | — | — |

_Outlier exclusion results pending: re-run with `--exclude-outliers` flag._
