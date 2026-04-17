"""
Train beauty prediction models with 5-fold cross-validation.

Usage:
  uv run python -m models.train                        # train xgboost (default)
  uv run python -m models.train --model ensemble       # train stacking ensemble
  uv run python -m models.train --model mlp            # train neural net
  uv run python -m models.train --model quantile       # quantile regression (predicts median)
  uv run python -m models.train --model ranker         # rank-target XGBoost
  uv run python -m models.train --model all            # train all models
  uv run python -m models.train --model xgboost --tune # XGBoost with Optuna
  uv run python -m models.train --augment              # 4x data via feature augmentation
"""

import argparse
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

random.seed(42)
np.random.seed(42)

from models.base import FEATURE_COLS, augment_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"

ALL_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
    "mlp",
    "quantile",
    "ranker",
]

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "xgboost": ("models.xgboost.model", "XGBoostBeautyModel"),
    "lightgbm": ("models.lightgbm.model", "LightGBMBeautyModel"),
    "catboost": ("models.catboost.model", "CatBoostBeautyModel"),
    "ensemble": ("models.ensemble.model", "StackingBeautyModel"),
    "mlp": ("models.mlp.model", "MLPBeautyModel"),
    "quantile": ("models.quantile.model", "QuantileBeautyModel"),
    "ranker": ("models.ranker.model", "RankerBeautyModel"),
}

N_FOLDS = 5
VAL_FRACTION = 0.1  # fraction of train split used for early stopping


def load_data():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run scripts/process.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLS].values
    y = df["score"].values
    return X, y, df


def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute per-dataset mean/std for z-score reconversion."""
    stats = {}
    for ds in ["mebeauty", "scut"]:
        sub = df.loc[df["dataset"] == ds, "score_raw"]
        stats[ds] = {"mean": float(sub.mean()), "std": float(sub.std())}
    return stats


def create_model(name: str):
    """Instantiate a fresh model by name."""
    if name not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{name}'. Available: {', '.join(ALL_MODELS)}")
        sys.exit(1)
    module_path, class_name = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def print_cv_results(name: str, fold_metrics: list[dict]):
    """Print aggregated cross-validation results."""
    print()
    print("=" * 55)
    print(f"  Model:     {name}  ({N_FOLDS}-fold CV)")
    print("-" * 55)

    keys = ["mae", "rmse", "pearson_r", "baseline_mae", "improvement_pct", "std_ratio"]
    for key in keys:
        values = [m[key] for m in fold_metrics]
        mean = np.mean(values)
        std = np.std(values)
        label = {
            "mae": "MAE",
            "rmse": "RMSE",
            "pearson_r": "Pearson r",
            "baseline_mae": "Baseline MAE",
            "improvement_pct": "Improvement %",
            "std_ratio": "Std ratio",
        }[key]
        print(f"  {label:<16s} {mean:.4f} ± {std:.4f}")

    print("=" * 55)


def print_final_model(model, X_all):
    """Print feature importance and SHAP for the final (all-data) model."""
    importances = model.feature_importances()
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance (final model):")
    print("-" * 45)
    for rank, (feat, imp) in enumerate(sorted_feats, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank:2d}. {feat:<28s} {imp:.4f}  {bar}")

    try:
        shap_values = model.shap_analysis(X_all)
        sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        print("\nSHAP (mean |SHAP value|, final model):")
        print("-" * 45)
        for rank, (feat, val) in enumerate(sorted_shap, 1):
            print(f"  {rank:2d}. {feat:<28s} {val:.4f}")
    except Exception as e:
        print(f"\nSHAP analysis skipped: {e}")


def run_cv(name: str, X, y, ds_stats, augment=False, **model_kwargs):
    """Run k-fold cross-validation for a single model type."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\n{'─' * 55}")
    print(f"Cross-validating: {name} ({N_FOLDS}-fold)")
    print(f"{'─' * 55}")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Split off a validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=VAL_FRACTION,
            random_state=42,
        )

        if augment:
            X_train, y_train = augment_features(
                X_train,
                y_train,
                n_copies=3,
                noise_std=0.02,
            )

        model = create_model(name)
        model.dataset_stats = ds_stats
        model.train(X_train, y_train, X_val, y_val, **model_kwargs)

        metrics = model.evaluate(X_test, y_test)
        fold_metrics.append(metrics)
        print(
            f"  Fold {fold + 1}: MAE={metrics['mae']:.4f}  r={metrics['pearson_r']:.4f}"
        )

    print_cv_results(name, fold_metrics)
    return fold_metrics


def train_final(name: str, X, y, ds_stats, augment=False, **model_kwargs):
    """Retrain on all data and save model artifacts for deployment."""
    # Use 10% of all data as val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_FRACTION,
        random_state=42,
    )

    if augment:
        X_train, y_train = augment_features(
            X_train,
            y_train,
            n_copies=3,
            noise_std=0.02,
        )

    model = create_model(name)
    model.dataset_stats = ds_stats

    print(f"\nRetraining {name} on all data ({len(X)} samples) ...")
    model.train(X_train, y_train, X_val, y_val, **model_kwargs)

    # Evaluate on the val slice just for metadata (not a true test metric)
    model.evaluate(X_val, y_val)
    model.save()

    print_final_model(model, X)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train beauty prediction models")
    parser.add_argument("--model", default="xgboost", choices=ALL_MODELS + ["all"])
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search (skipped for ensemble)",
    )
    parser.add_argument(
        "--trials", type=int, default=200, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Augment training data (4x via noise)"
    )
    args = parser.parse_args()

    X, y, df = load_data()
    ds_stats = dataset_stats(df)

    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print(f"  Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"  Score range: {df['score'].min():.2f} to {df['score'].max():.2f}")
    print(f"  Features: {len(FEATURE_COLS)}")

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]

    for name in models_to_train:
        kwargs = {}
        if args.tune and name != "ensemble":
            kwargs = {"tune": True, "n_trials": args.trials}

        # 1. Cross-validation for robust metrics
        run_cv(name, X, y, ds_stats, augment=args.augment, **kwargs)

        # 2. Retrain on all data and save for deployment
        train_final(name, X, y, ds_stats, augment=args.augment, **kwargs)

    print("\nDone!")


if __name__ == "__main__":
    main()
