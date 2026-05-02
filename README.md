# Facial Attractiveness Prediction from Geometric Beauty Markers

University data mining project: can facial geometry alone predict how attractive a face is rated?

We extract 30 geometric beauty markers from facial landmarks (MediaPipe Face Mesh), then use XGBoost and other models to predict attractiveness ratings. SHAP analysis reveals which markers actually influence perceived attractiveness. Cross-dataset and fairness analyses test whether predictions generalize across cultures and demographic groups.

## Datasets

| Dataset | N | Scale | Ethnicities | Raters |
|---------|---|-------|-------------|--------|
| [MEBeauty](https://github.com/fbplab/MEBeauty-database) | 2,370 | 1-10 | asian, caucasian, black, hispanic, indian, mideastern | ~300 |
| [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) | 5,500 | 1-5 | asian, caucasian | 60 |
| [LiveBeauty](https://github.com/FredaXYu/LiveBeauty) | 10,000 | 1-5 | asian | ~20 per face (200K labels) |

Scores are z-normalized per dataset so they can be combined (17,870 total).

## Setup & Usage

```bash
cd code
uv sync

# 0. Download datasets into code/datasets/ (pinned to specific commits)
uv run scripts/prepare.py
# LiveBeauty: download manually into code/datasets/LiveBeauty_public/

# 1. Extract features (MediaPipe landmarks -> beauty markers + expression)
uv run scripts/process.py

# 2. Train models and evaluate (nested 5-fold CV)
uv run scripts/train.py                    # xgboost (default)
uv run scripts/train.py --model all        # all models
uv run scripts/train.py --model xgboost --tune  # with Optuna hyperparameter tuning

# 3. Cross-dataset generalization & fairness analysis
uv run python scripts/analyze_fairness.py
uv run python scripts/analyze_fairness.py --shap  # with per-ethnicity SHAP

# 4. (Optional) AutoGluon benchmark — automated model selection ceiling
uv run --extra benchmark python scripts/benchmark_autogluon.py --time 600
```

## Results

| Model | MAE (z-score) | Pearson r | vs. Baseline |
|-------|---------------|-----------|--------------|
| Ensemble (stacking) | 0.40 | 0.85 | +49.5% |
| XGBoost | 0.43 | 0.83 | +46.3% |
| Combined 3-dataset CV | 0.57 | 0.68 | +30.5% |

Cross-dataset generalization (train on one, test on another) drops to r=0.28-0.50, confirming that beauty perception is partially culture-specific.

## Fairness

Per-ethnicity performance (combined model, 5-fold CV):

| Ethnicity | N | MAE | Pearson r | Δ MAE vs. overall |
|-----------|---|-----|-----------|-------------------|
| asian | 14,351 | 0.55 | 0.70 | -2.8% |
| caucasian | 2,480 | 0.60 | 0.60 | +5.8% |
| mideastern | 291 | 0.72 | 0.57 | +26.3% |
| black | 296 | 0.62 | 0.47 | +8.3% |
| hispanic | 296 | 0.72 | 0.41 | +26.2% |
| indian | 156 | 0.88 | 0.26 | +54.1% |

Underrepresented groups suffer from data scarcity, not model bias.

## Top Predictors (SHAP)

1. Eye width ratio
2. Nose width ratio
3. Mouth-chin ratio
4. Gonial angle (jaw sharpness)
5. Jaw width ratio

Canthal tilt — despite its popularity in online beauty discourse — ranks near the bottom.

## Project Structure

```
code/
  datasets/           # raw image datasets (gitignored, download separately)
    MEBeauty-database-main/
    SCUT-FBP5500_v2/
    LiveBeauty_public/
  data/
    features.csv      # extracted features (17,870 samples × 50 columns)
    debug/            # landmark overlay images for visual verification
  models/
    base.py           # abstract model class, feature definitions
    train.py          # nested CV training loop
    xgboost/          # XGBoost model + artifacts
    ensemble/         # stacking ensemble
    mlp/              # neural network
    quantile/         # quantile regression
    ranker/           # rank-target XGBoost
    lightgbm/         # LightGBM
    catboost/         # CatBoost
  scripts/
    prepare.py        # download datasets
    process.py        # MediaPipe extraction -> data/features.csv
    train.py          # CLI wrapper for model training
    analyze_fairness.py  # cross-dataset & fairness analysis
    benchmark_autogluon.py  # AutoGluon ceiling
  pyproject.toml
docs/                 # LaTeX project outline
```
