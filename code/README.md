# Facial Attractiveness Prediction from Geometric Beauty Markers

Predicts facial attractiveness ratings using geometric features extracted from MediaPipe Face Mesh landmarks, trained with XGBoost.

## Setup

```bash
cd code
uv sync
```

## Datasets

Download and place these in the `code/` directory:

1. **MEBeauty** — [GitHub](https://github.com/fbplab/MEBeauty-database)
   ```
   code/MEBeauty-database-main/
   ```

2. **SCUT-FBP5500** — [GitHub](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
   ```
   code/SCUT-FBP5500_v2/
   ```

## Usage

```bash
# 1. Extract features (MediaPipe landmarks → beauty markers)
uv run python scripts/process.py

# 2. Train and evaluate
uv run python scripts/train.py
```

`process.py` outputs:
- `data/features.csv` — tabular features + scores for all images
- `data/debug/*.jpg` — landmark overlay images for visual verification

`train.py` outputs MAE, RMSE, and feature importance (XGBoost + SHAP).

## Features (42)

Geometric ratios (33): canthal tilt, eye width/height/area/spacing, eyebrow distance, nose width/length, lip width/fullness, cupid's bow, facial symmetry (overall + per-region), face proportions (thirds, phi deviation), jaw width, gonial angle, chin taper, cheekbone prominence, etc.

Expression blendshapes (9): smile, frown, jaw open, brow up/down, cheek/eye squint, eye wide, mouth pucker.

All geometric features are scale-invariant (ratios) and corrected for head roll.
