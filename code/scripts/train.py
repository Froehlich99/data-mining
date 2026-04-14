"""
Train XGBoost regressor on facial beauty markers extracted by process.py.

Usage:
  uv run python scripts/train.py          # train with default params
  uv run python scripts/train.py --tune   # run Optuna hyperparameter search first
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"

# These are the numeric feature columns produced by process.py
FEATURE_COLS = [
    # Original features
    "canthal_tilt",
    "eye_width_ratio",
    "eye_height_ratio",
    "eyebrow_eye_dist",
    "nose_width_ratio",
    "nose_length_ratio",
    "lip_width_ratio",
    "upper_lip_ratio",
    "facial_symmetry",
    "face_length_width_ratio",
    "midface_ratio",
    "lower_face_ratio",
    "jaw_width_ratio",
    "cheekbone_prominence",
    "interpupillary_ratio",
    # New features
    "eye_spacing_ratio",
    "eye_area_ratio",
    "scleral_show",
    "eye_asymmetry",
    "brow_arch_height",
    "lip_fullness_ratio",
    "mouth_width_face_ratio",
    "cupids_bow_ratio",
    "mouth_chin_ratio",
    "gonial_angle",
    "chin_taper",
    "face_taper_ratio",
    "upper_face_ratio",
    "phi_deviation",
    "facial_thirds_symmetry",
    "eye_symmetry",
    "mouth_symmetry",
    "nose_symmetry",
    # Expression (blendshapes)
    "expr_smile",
    "expr_frown",
    "expr_jaw_open",
    "expr_brow_up",
    "expr_brow_down",
    "expr_cheek_squint",
    "expr_eye_squint",
    "expr_eye_wide",
    "expr_mouth_pucker",
]


def load_data():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run scripts/process.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["score"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["score"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["score"].values

    return X_train, y_train, X_val, y_val, X_test, y_test, df


def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=200):
    """Run Optuna hyperparameter search using 5-fold CV on train+val combined."""
    import optuna
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Combine train+val for cross-validation (avoids overfitting to tiny val set)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "random_state": 42,
        }

        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, X_combined, y_combined,
            cv=5, scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        return -scores.mean()  # minimize MAE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna search complete ({n_trials} trials)")
    print(f"  Best 5-fold CV MAE: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print()

    return study.best_params


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, params=None):
    """Train XGBoost with given params (or defaults) and evaluate."""
    if params is None:
        base_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        # Use early stopping with val set for default params
        model = xgb.XGBRegressor(
            n_estimators=1000,
            early_stopping_rounds=50,
            random_state=42,
            **base_params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"Best iteration: {model.best_iteration}")
    else:
        # Tuned params: train on train+val combined (CV already validated)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        model = xgb.XGBRegressor(random_state=42, **params)
        model.fit(X_combined, y_combined)
        print(f"Trained on train+val ({len(X_combined)} samples) with tuned params")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Baseline
    mean_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, mean_pred)
    improvement = (baseline_mae - mae) / baseline_mae * 100

    print()
    print("=" * 50)
    print(f"  Baseline MAE (predict mean): {baseline_mae:.4f}")
    print(f"  Test MAE:  {mae:.4f}  ({improvement:.1f}% better than baseline)")
    print(f"  Test RMSE: {rmse:.4f}")
    print("=" * 50)
    print()

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("Feature Importance Ranking:")
    print("-" * 40)
    for rank, idx in enumerate(sorted_idx, 1):
        bar = "#" * int(importances[idx] * 50)
        print(f"  {rank:2d}. {FEATURE_COLS[idx]:<28s} {importances[idx]:.4f}  {bar}")

    # SHAP
    try:
        import shap
        print("\nSHAP Summary (mean |SHAP value| per feature):")
        print("-" * 40)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_sorted = np.argsort(mean_abs_shap)[::-1]
        for rank, idx in enumerate(shap_sorted, 1):
            print(f"  {rank:2d}. {FEATURE_COLS[idx]:<28s} {mean_abs_shap[idx]:.4f}")
    except Exception as e:
        print(f"\nSHAP analysis skipped: {e}")

    return model, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, df = load_data()

    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    print(f"  Score range: {df['score'].min():.2f} – {df['score'].max():.2f}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    params = None
    if args.tune:
        params = tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=args.trials)

    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, params=params)


if __name__ == "__main__":
    main()
