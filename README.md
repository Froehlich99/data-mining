# Facial Attractiveness Prediction from Facial Features

University of Mannheim Data Mining project. We extract 30 geometric beauty markers and 9 expression markers from MediaPipe Face Mesh, then train tree-based and ensemble models to predict attractiveness ratings on three datasets, with cross-dataset and per-ethnicity analyses.

The full write-up is in [`docs/Report/Team_3_Project_Report.pdf`](docs/Report/Team_3_Project_Report.pdf).

## Datasets

| Dataset | N | Scale | Ethnicities | Raters | Source |
|---------|---|-------|-------------|--------|--------|
| MEBeauty | 2,370 | 1-10 | asian, caucasian, black, hispanic, indian, mideastern | ~300 | [github.com/fbplab/MEBeauty-database](https://github.com/fbplab/MEBeauty-database/tree/main) |
| SCUT-FBP5500 | 5,500 | 1-5 | asian, caucasian | 60 | [github.com/HCIILab/SCUT-FBP5500-Database-Release](https://github.com/hciilab/scut-fbp5500-database-release) |
| LiveBeauty | 10,000 | 1-5 | asian | ~20 per face | [tianchi.aliyun.com/dataset/216302](https://tianchi.aliyun.com/dataset/216302) |

Scores are z-normalized per dataset so they can be combined (17,870 total).

### Downloading

MEBeauty and SCUT-FBP5500 are fetched automatically by `uv run scripts/prepare.py` (clones the MEBeauty repo at a pinned commit and downloads SCUT from Google Drive). Both land in `code/datasets/` under `MEBeauty-database-main/` and `SCUT-FBP5500_v2/` respectively.

LiveBeauty requires a manual download (Aliyun Tianchi login). After downloading, extract it so the directory structure looks like:

```
code/datasets/LiveBeauty_public/
  images/
  labels/
  ...
```

## Setup & Usage

```bash
cd code
uv sync

# 0. Download datasets into code/datasets/
uv run scripts/prepare.py
# LiveBeauty: download manually into code/datasets/LiveBeauty_public/

# 1. Extract features (MediaPipe landmarks -> beauty markers + expression)
uv run scripts/process.py

# 2. Train models and evaluate (nested 5-fold CV)
uv run scripts/train.py                           # xgboost (default)
uv run scripts/train.py --model all               # all models
uv run scripts/train.py --model xgboost --tune    # with Optuna tuning

# 3. Cross-dataset generalization & fairness analysis
uv run python scripts/analyze_fairness.py
uv run python scripts/analyze_fairness.py --shap  # with per-ethnicity SHAP

# 4. (Optional) Web demo — drag a face into the browser to see scores
uv run python scripts/app.py
```

## Project Structure

```
code/
  datasets/             # raw image datasets (gitignored, download separately)
  data/
    features.csv        # extracted features (17,870 samples × 50 columns)
    debug/              # landmark overlay images for visual verification
  models/
    base.py             # abstract model class, feature definitions
    train.py            # nested CV training loop
    xgboost/            # XGBoost model + artifacts
    ensemble/           # stacking ensemble
    mlp/                # neural network
    quantile/           # quantile regression
    ranker/             # rank-target XGBoost
    lightgbm/           # LightGBM
    catboost/           # CatBoost
  notebooks/            # exploratory analyses (clustering, ablations, SHAP)
  scripts/
    prepare.py          # download datasets
    process.py          # MediaPipe extraction -> data/features.csv
    train.py            # CLI wrapper for model training
    analyze_fairness.py # cross-dataset & fairness analysis
    app.py              # Flask web demo
  pyproject.toml
docs/
  Report/               # final LaTeX report + PDF
  Outline/              # initial project outline
  presentation/         # final slides
```
