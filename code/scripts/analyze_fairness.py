"""
Cross-dataset generalization and fairness analysis.

Evaluates whether facial beauty markers generalize across datasets and
demographic groups. Produces:
  1. Cross-dataset generalization (train on A, test on B)
  2. Per-ethnicity and per-gender fairness breakdown (out-of-fold predictions)
  3. Optional per-group SHAP analysis

Usage:
  uv run python scripts/analyze_fairness.py
  uv run python scripts/analyze_fairness.py --model ensemble
  uv run python scripts/analyze_fairness.py --shap
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import FEATURE_COLS
from models.train import N_FOLDS, VAL_FRACTION, create_model, dataset_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fairness_results"


def load_data() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run scripts/process.py first.")
        sys.exit(1)
    return pd.read_csv(FEATURES_CSV)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 3:
        return {"mae": np.nan, "rmse": np.nan, "pearson_r": np.nan, "n": len(y_true)}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "pearson_r": round(float(r), 4), "n": len(y_true)}


def cross_dataset_eval(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Train on one dataset, evaluate on the other(s)."""
    ds_stats = dataset_stats(df)
    results = []

    datasets = sorted(df["dataset"].unique())
    experiments = []
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds != test_ds:
                experiments.append((f"{train_ds} → {test_ds}", train_ds, test_ds))

    for label, train_ds, test_ds in experiments:
        train_df = df[df["dataset"] == train_ds]
        test_df = df[df["dataset"] == test_ds]

        X_train_full = train_df[FEATURE_COLS].values
        y_train_full = train_df["score"].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["score"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_FRACTION, random_state=42
        )

        model = create_model(model_name)
        model.dataset_stats = ds_stats
        model.train(X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["experiment"] = label
        metrics["train_n"] = len(train_df)
        metrics["test_n"] = len(test_df)
        results.append(metrics)

        # Per-ethnicity drill-down on test set (skip if all unknown)
        ethnicities = test_df["ethnicity"].unique()
        if not (len(ethnicities) == 1 and ethnicities[0] == "unknown"):
            for eth in sorted(ethnicities):
                if eth == "unknown":
                    continue
                mask = test_df["ethnicity"].values == eth
                if mask.sum() < 3:
                    continue
                eth_metrics = compute_metrics(y_test[mask], y_pred[mask])
                eth_metrics["experiment"] = f"  └ {eth}"
                eth_metrics["train_n"] = ""
                eth_metrics["test_n"] = int(mask.sum())
                results.append(eth_metrics)

    return pd.DataFrame(results)


def combined_cv_baseline(df: pd.DataFrame, model_name: str) -> tuple[dict, np.ndarray]:
    """Run 5-fold CV on combined data, return overall metrics and out-of-fold predictions."""
    ds_stats = dataset_stats(df)
    X = df[FEATURE_COLS].values
    y = df["score"].values

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.full(len(y), np.nan)

    for train_idx, test_idx in kf.split(X):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_FRACTION, random_state=42
        )

        model = create_model(model_name)
        model.dataset_stats = ds_stats
        model.train(X_train, y_train, X_val, y_val)

        oof_preds[test_idx] = model.predict(X_test)

    overall = compute_metrics(y, oof_preds)
    overall["experiment"] = f"Combined ({N_FOLDS}-fold CV)"
    overall["train_n"] = len(df)
    overall["test_n"] = len(df)
    return overall, oof_preds


def fairness_breakdown(df: pd.DataFrame, oof_preds: np.ndarray) -> pd.DataFrame:
    """Per-ethnicity and per-gender metrics from out-of-fold predictions."""
    y = df["score"].values
    overall_mae = mean_absolute_error(y, oof_preds)
    rows = []

    # Per-ethnicity
    for eth in sorted(df["ethnicity"].unique()):
        mask = df["ethnicity"].values == eth
        metrics = compute_metrics(y[mask], oof_preds[mask])
        metrics["group_type"] = "ethnicity"
        metrics["group"] = eth
        metrics["delta_mae_pct"] = round((metrics["mae"] - overall_mae) / overall_mae * 100, 1)
        rows.append(metrics)

    # Per-gender
    for gender in sorted(df["gender"].unique()):
        mask = df["gender"].values == gender
        metrics = compute_metrics(y[mask], oof_preds[mask])
        metrics["group_type"] = "gender"
        metrics["group"] = gender
        metrics["delta_mae_pct"] = round((metrics["mae"] - overall_mae) / overall_mae * 100, 1)
        rows.append(metrics)

    return pd.DataFrame(rows)


def shap_by_group(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Per-ethnicity SHAP analysis using a model trained on all data."""
    ds_stats = dataset_stats(df)
    X = df[FEATURE_COLS].values
    y = df["score"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_FRACTION, random_state=42)

    model = create_model(model_name)
    model.dataset_stats = ds_stats
    model.train(X_train, y_train, X_val, y_val)

    import shap

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)

    rows = []
    for eth in sorted(df["ethnicity"].unique()):
        mask = df["ethnicity"].values == eth
        mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            rows.append({
                "ethnicity": eth,
                "rank": rank,
                "feature": FEATURE_COLS[idx],
                "mean_abs_shap": round(float(mean_abs[idx]), 4),
            })

    return pd.DataFrame(rows)


def print_cross_dataset(cross_df: pd.DataFrame, baseline: dict):
    print()
    print("=" * 70)
    print("  CROSS-DATASET GENERALIZATION")
    print("=" * 70)
    print(f"  {'Experiment':<25s} {'Train N':>8s} {'Test N':>8s} {'MAE':>8s} {'RMSE':>8s} {'r':>8s}")
    print("  " + "─" * 65)

    for _, row in cross_df.iterrows():
        train_n = str(row["train_n"]) if row["train_n"] != "" else ""
        print(
            f"  {row['experiment']:<25s} {train_n:>8s} {str(row['test_n']):>8s} "
            f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['pearson_r']:>8.4f}"
        )

    print("  " + "─" * 65)
    print(
        f"  {baseline['experiment']:<25s} {str(baseline['train_n']):>8s} "
        f"{str(baseline['test_n']):>8s} {baseline['mae']:>8.4f} "
        f"{baseline['rmse']:>8.4f} {baseline['pearson_r']:>8.4f}"
    )
    print("=" * 70)


def print_fairness(fairness_df: pd.DataFrame):
    print()
    print("=" * 70)
    print("  FAIRNESS: PER-GROUP PERFORMANCE (combined model, 5-fold CV)")
    print("=" * 70)

    for group_type in ["ethnicity", "gender"]:
        sub = fairness_df[fairness_df["group_type"] == group_type]
        print(f"\n  By {group_type}:")
        print(f"  {'Group':<14s} {'N':>6s} {'MAE':>8s} {'Pearson r':>10s} {'Δ MAE':>8s}")
        print("  " + "─" * 50)
        for _, row in sub.iterrows():
            delta_str = f"{row['delta_mae_pct']:+.1f}%"
            print(
                f"  {row['group']:<14s} {row['n']:>6d} {row['mae']:>8.4f} "
                f"{row['pearson_r']:>10.4f} {delta_str:>8s}"
            )

    print()
    print("=" * 70)


def print_shap(shap_df: pd.DataFrame):
    print()
    print("=" * 70)
    print("  PER-ETHNICITY SHAP: TOP-5 FEATURES")
    print("=" * 70)

    for eth in sorted(shap_df["ethnicity"].unique()):
        sub = shap_df[shap_df["ethnicity"] == eth]
        features = ", ".join(f"{r['feature']}({r['mean_abs_shap']:.3f})" for _, r in sub.iterrows())
        print(f"  {eth:<12s}: {features}")

    print("=" * 70)


def save_results(cross_df, baseline, fairness_df, shap_df=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-dataset results
    all_cross = pd.concat([cross_df, pd.DataFrame([baseline])], ignore_index=True)
    all_cross.to_csv(OUTPUT_DIR / "cross_dataset.csv", index=False)

    # Fairness breakdown
    fairness_df.to_csv(OUTPUT_DIR / "fairness_breakdown.csv", index=False)

    # SHAP
    if shap_df is not None:
        shap_df.to_csv(OUTPUT_DIR / "shap_by_ethnicity.csv", index=False)

    print(f"\n  Results saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset and fairness analysis")
    parser.add_argument("--model", default="xgboost", help="Model to evaluate (default: xgboost)")
    parser.add_argument("--shap", action="store_true", help="Run per-ethnicity SHAP analysis")
    args = parser.parse_args()

    df = load_data()
    print(f"Loaded {len(df)} samples ({df['dataset'].value_counts().to_dict()})")
    print(f"Ethnicities: {df['ethnicity'].value_counts().to_dict()}")
    print(f"Model: {args.model}")

    # 1. Cross-dataset generalization
    print("\n[1/3] Cross-dataset generalization...")
    cross_df = cross_dataset_eval(df, args.model)

    # 2. Combined CV baseline + out-of-fold predictions for fairness
    print("\n[2/3] Combined CV (out-of-fold predictions for fairness)...")
    baseline, oof_preds = combined_cv_baseline(df, args.model)

    # 3. Fairness breakdown
    print("\n[3/3] Fairness breakdown...")
    fairness_df = fairness_breakdown(df, oof_preds)

    # 4. Optional SHAP
    shap_df = None
    if args.shap:
        print("\n[bonus] Per-ethnicity SHAP...")
        shap_df = shap_by_group(df, args.model)

    # Print results
    print_cross_dataset(cross_df, baseline)
    print_fairness(fairness_df)
    if shap_df is not None:
        print_shap(shap_df)

    # Save CSVs
    save_results(cross_df, baseline, fairness_df, shap_df)


if __name__ == "__main__":
    main()
