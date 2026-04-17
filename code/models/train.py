"""
Train beauty prediction models and save to disk.

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
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

from models.base import FEATURE_COLS, augment_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"

ALL_MODELS = ["xgboost", "ensemble", "mlp", "quantile", "ranker"]

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "xgboost": ("models.xgboost.model", "XGBoostBeautyModel"),
    "ensemble": ("models.ensemble.model", "StackingBeautyModel"),
    "mlp": ("models.mlp.model", "MLPBeautyModel"),
    "quantile": ("models.quantile.model", "QuantileBeautyModel"),
    "ranker": ("models.ranker.model", "RankerBeautyModel"),
}


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


def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute per-dataset mean/std for z-score reconversion."""
    stats = {}
    for ds in ["mebeauty", "scut"]:
        sub = df.loc[df["dataset"] == ds, "score_raw"]
        stats[ds] = {"mean": float(sub.mean()), "std": float(sub.std())}
    return stats


def print_results(model, metrics: dict, X_test, feature_cols: list[str]):
    print()
    print("=" * 55)
    print(f"  Model:        {model.name}")
    print(f"  Baseline MAE: {metrics['baseline_mae']:.4f}")
    print(
        f"  Test MAE:     {metrics['mae']:.4f}  ({metrics['improvement_pct']:.1f}% better)"
    )
    print(f"  Test RMSE:    {metrics['rmse']:.4f}")
    print(f"  Pearson r:    {metrics['pearson_r']:.4f}")
    print(f"  Std ratio:    {metrics['std_ratio']:.4f}  (1.0 = perfect spread)")
    print(f"  MAE (bottom quartile): {metrics['mae_bottom_quartile']:.4f}")
    print(f"  MAE (top quartile):    {metrics['mae_top_quartile']:.4f}")
    print(f"  MAE (middle 50%):      {metrics['mae_middle']:.4f}")
    print("=" * 55)

    # Feature importance
    importances = model.feature_importances()
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    print("-" * 45)
    for rank, (feat, imp) in enumerate(sorted_feats, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank:2d}. {feat:<28s} {imp:.4f}  {bar}")

    # SHAP
    try:
        shap_values = model.shap_analysis(X_test)
        sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        print("\nSHAP (mean |SHAP value|):")
        print("-" * 45)
        for rank, (feat, val) in enumerate(sorted_shap, 1):
            print(f"  {rank:2d}. {feat:<28s} {val:.4f}")
    except Exception as e:
        print(f"\nSHAP analysis skipped: {e}")


def train_model(
    name: str, X_train, y_train, X_val, y_val, X_test, y_test, ds_stats, **kwargs
):
    """Instantiate, train, evaluate, and save a model."""
    if name not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{name}'. Available: {', '.join(ALL_MODELS)}")
        sys.exit(1)

    module_path, class_name = MODEL_REGISTRY[name]
    import importlib

    module = importlib.import_module(module_path)
    model = getattr(module, class_name)()

    model.dataset_stats = ds_stats

    print(f"\n{'─' * 55}")
    print(f"Training: {name}")
    print(f"{'─' * 55}")
    model.train(X_train, y_train, X_val, y_val, **kwargs)

    metrics = model.evaluate(X_test, y_test)
    model.save()
    print_results(model, metrics, X_test, FEATURE_COLS)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train beauty prediction models")
    parser.add_argument("--model", default="xgboost", choices=ALL_MODELS + ["all"])
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search (XGBoost only)",
    )
    parser.add_argument(
        "--trials", type=int, default=200, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Augment training data (4x via noise)"
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, df = load_data()
    ds_stats = dataset_stats(df)

    if args.augment:
        X_train, y_train = augment_features(
            X_train, y_train, n_copies=3, noise_std=0.02
        )
        print(f"Augmented training data: {len(X_train)} samples (4x)")

    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    print(f"  Score range: {df['score'].min():.2f} to {df['score'].max():.2f}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]

    for name in models_to_train:
        kwargs = {}
        if name == "xgboost" and args.tune:
            kwargs = {"tune": True, "n_trials": args.trials}
        train_model(
            name, X_train, y_train, X_val, y_val, X_test, y_test, ds_stats, **kwargs
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
