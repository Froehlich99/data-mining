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
from sklearn.model_selection import StratifiedKFold, train_test_split

random.seed(42)
np.random.seed(42)

from models.base import FEATURE_COLS, FEATURE_GROUPS, augment_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"

ALL_MODELS = [
    "ridge",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
    "mlp",
    "quantile",
    "ranker",
]

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "ridge": ("models.linear.model", "RidgeBeautyModel"),
    "decision_tree": ("models.decision_tree.model", "DecisionTreeBeautyModel"),
    "random_forest": ("models.random_forest.model", "RandomForestBeautyModel"),
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
    for ds in df["dataset"].unique():
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
    """Print and save feature importance and SHAP for the final (all-data) model."""
    import json

    importances = model.feature_importances()
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance (final model):")
    print("-" * 45)
    for rank, (feat, imp) in enumerate(sorted_feats, 1):
        bar = "#" * int(imp * 50)
        print(f"  {rank:2d}. {feat:<28s} {imp:.4f}  {bar}")

    print("\nFeature Importance by group:")
    print("-" * 45)
    group_importance = {
        group: sum(importances.get(feature, 0.0) for feature in features)
        for group, features in FEATURE_GROUPS.items()
    }
    for group, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 50)
        print(f"  {group:<28s} {imp:.4f}  {bar}")

    results = {
        "feature_importance": importances,
        "feature_group_importance": group_importance,
    }

    try:
        shap_values = model.shap_analysis(X_all)
        sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        print("\nSHAP (mean |SHAP value|, final model):")
        print("-" * 45)
        for rank, (feat, val) in enumerate(sorted_shap, 1):
            print(f"  {rank:2d}. {feat:<28s} {val:.4f}")
        results["shap_importance"] = shap_values
        results["shap_group_importance"] = {
            group: sum(shap_values.get(feature, 0.0) for feature in features)
            for group, features in FEATURE_GROUPS.items()
        }
    except Exception as e:
        print(f"\nSHAP analysis skipped: {e}")

    # Save to artifacts directory
    importance_path = model.artifacts_dir / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Feature importance saved to {importance_path}")


def run_cv(name: str, X, y, ds_stats, stratify_labels=None, augment=False, **model_kwargs):
    """Run k-fold cross-validation for a single model type."""
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\n{'─' * 55}")
    print(f"Cross-validating: {name} ({N_FOLDS}-fold, stratified)")
    print(f"{'─' * 55}")

    split_args = (X, stratify_labels) if stratify_labels is not None else (X,)
    for fold, (train_idx, test_idx) in enumerate(kf.split(*split_args)):
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


def evaluate_holdout(name: str, X_train_all, y_train_all, X_test, y_test,
                     ds_stats, ethnicity_test=None, augment=False, **model_kwargs):
    """Train on full training split, evaluate on held-out test set."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=VAL_FRACTION, random_state=42
    )

    if augment:
        X_train, y_train = augment_features(
            X_train, y_train, n_copies=3, noise_std=0.02
        )

    model = create_model(name)
    model.dataset_stats = ds_stats
    model.train(X_train, y_train, X_val, y_val, **model_kwargs)

    metrics = model.evaluate(X_test, y_test)

    print()
    print(f"  Held-out test ({len(y_test)} samples):")
    print(f"    MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  r={metrics['pearson_r']:.4f}")

    if ethnicity_test is not None:
        from scipy.stats import pearsonr as _pearsonr
        from sklearn.metrics import mean_absolute_error as _mae

        metrics["per_ethnicity"] = {}
        print(f"    Per-ethnicity:")
        for eth in sorted(set(ethnicity_test)):
            mask = ethnicity_test == eth
            n = mask.sum()
            if n < 3:
                continue
            y_pred = model.predict(X_test[mask])
            eth_mae = _mae(y_test[mask], y_pred)
            eth_r, _ = _pearsonr(y_test[mask], y_pred)
            metrics["per_ethnicity"][eth] = {"mae": round(eth_mae, 4), "pearson_r": round(float(eth_r), 4), "n": int(n)}
            print(f"      {eth:<14s} n={n:>5d}  MAE={eth_mae:.4f}  r={eth_r:.4f}")

    return metrics


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
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Exclude flagged high-variance images from training/eval",
    )
    args = parser.parse_args()

    X, y, df = load_data()

    if args.exclude_outliers and "outlier_flag" in df.columns:
        mask = df["outlier_flag"] != "high_variance"
        df = df[mask].reset_index(drop=True)
        X = df[FEATURE_COLS].values
        y = df["score"].values
        print(f"Excluded outliers: {(~mask).sum()} high-variance images removed")

    ds_stats = dataset_stats(df)

    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print(f"  Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"  Ethnicities: {df['ethnicity'].value_counts().to_dict()}")
    print(f"  Score range: {df['score'].min():.2f} to {df['score'].max():.2f}")
    print(f"  Features: {len(FEATURE_COLS)}")

    # Stratified 80/20 split by ethnicity
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["ethnicity"].values,
        random_state=42,
    )
    X_train_all, X_test = X[train_idx], X[test_idx]
    y_train_all, y_test = y[train_idx], y[test_idx]
    ethnicity_train = df.iloc[train_idx]["ethnicity"].values
    ethnicity_test = df.iloc[test_idx]["ethnicity"].values

    print(f"\n  Train: {len(train_idx)} samples | Test: {len(test_idx)} samples (stratified by ethnicity)")

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]

    all_results = {}
    for name in models_to_train:
        kwargs = {}
        if args.tune and name != "ensemble":
            kwargs = {"tune": True, "n_trials": args.trials}

        # 1. Cross-validation on training split (stratified)
        fold_metrics = run_cv(name, X_train_all, y_train_all, ds_stats, stratify_labels=ethnicity_train, augment=args.augment, **kwargs)

        # 2. Held-out test evaluation
        holdout_metrics = evaluate_holdout(name, X_train_all, y_train_all, X_test, y_test, ds_stats,
                         ethnicity_test=ethnicity_test, augment=args.augment, **kwargs)

        # 3. Retrain on all data and save for deployment
        train_final(name, X, y, ds_stats, augment=args.augment, **kwargs)

        all_results[name] = {
            "cv_mean": {k: round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in ["mae", "rmse", "pearson_r"]},
            "cv_std": {k: round(float(np.std([m[k] for m in fold_metrics])), 4) for k in ["mae", "rmse", "pearson_r"]},
            "holdout": holdout_metrics,
        }

    # Save training results summary
    import json
    results_path = PROJECT_ROOT / "data" / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
