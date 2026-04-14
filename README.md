# Facial Attractiveness Prediction from Geometric Beauty Markers

University data mining project: can facial geometry alone predict how attractive a face is rated?

We extract 42 geometric beauty markers from facial landmarks (MediaPipe Face Mesh), then use XGBoost to predict attractiveness ratings. SHAP analysis reveals which markers actually influence perceived attractiveness.

## Datasets

Download and place these in the `code/` directory:

1. **MEBeauty** — [GitHub](https://github.com/fbplab/MEBeauty-database) — 2,550 multi-ethnic faces rated by ~300 raters (1-10 scale)
   ```
   code/MEBeauty-database-main/
   ```

2. **SCUT-FBP5500** — [GitHub](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) — 5,500 faces rated by 60 raters (1-5 scale)
   ```
   code/SCUT-FBP5500_v2/
   ```

Scores are z-normalized per dataset so they can be combined.

## Setup & Usage

```bash
cd code
uv sync

# 1. Extract features (MediaPipe landmarks -> beauty markers + expression)
uv run python scripts/process.py

# 2. Train XGBoost and evaluate
uv run python scripts/train.py
```

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 0.54 (z-score), ~0.72 on 1-10 / ~0.37 on 1-5 |
| vs. baseline | 33.8% better than predicting the mean |

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
  scripts/
    process.py    # MediaPipe extraction -> data/features.csv + data/debug/
    train.py      # XGBoost training, MAE/RMSE, SHAP feature importance
  pyproject.toml  # uv project (mediapipe, xgboost, shap, etc.)
docs/             # LaTeX project outline
```
