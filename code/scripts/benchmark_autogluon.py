"""
AutoGluon benchmark — automated model selection as a performance ceiling.

Runs AutoGluon's TabularPredictor with 5-fold CV to see how far automated
model search can push MAE on the same features. Useful as a reference point:
"AutoGluon achieved MAE X, our interpretable XGBoost achieved MAE Y."

Usage:
  uv run --extra benchmark python scripts/benchmark_autogluon.py
  uv run --extra benchmark python scripts/benchmark_autogluon.py --time 300
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"

N_FOLDS = 5


def main():
    parser = argparse.ArgumentParser(description="AutoGluon benchmark")
    parser.add_argument(
        "--time",
        type=int,
        default=600,
        help="Total time budget in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--presets",
        default="best_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon quality preset (default: best_quality)",
    )
    args = parser.parse_args()

    from autogluon.tabular import TabularPredictor

    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run scripts/process.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    data = df[FEATURE_COLS + ["score"]].copy()

    print(f"AutoGluon benchmark")
    print(f"  Samples: {len(data)}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Time budget: {args.time}s total ({args.time // N_FOLDS}s per fold)")
    print(f"  Presets: {args.presets}")
    print()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_maes = []
    fold_rmses = []
    fold_rs = []

    time_per_fold = args.time // N_FOLDS

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        X_test = test_data[FEATURE_COLS].values
        y_test = test_data["score"].values

        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = TabularPredictor(
                label="score",
                eval_metric="mean_absolute_error",
                path=tmpdir,
                verbosity=1,
            )
            predictor.fit(
                train_data,
                time_limit=time_per_fold,
                presets=args.presets,
            )

            y_pred = predictor.predict(test_data.drop(columns=["score"])).values

            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            from scipy.stats import pearsonr

            r, _ = pearsonr(y_test, y_pred)

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_rs.append(r)

            print(f"\n  Fold {fold + 1}: MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}")

            # Print leaderboard for last fold
            if fold == N_FOLDS - 1:
                print("\n  Leaderboard (last fold):")
                lb = predictor.leaderboard(test_data, silent=True)
                print(
                    lb[["model", "score_test", "pred_time_test", "fit_time"]].to_string(
                        index=False
                    )
                )

    print()
    print("=" * 55)
    print(f"  AutoGluon  ({N_FOLDS}-fold CV, {args.time}s budget)")
    print("-" * 55)
    print(f"  MAE              {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"  RMSE             {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
    print(f"  Pearson r        {np.mean(fold_rs):.4f} ± {np.std(fold_rs):.4f}")

    baseline_mae = data["score"].apply(lambda x: abs(x - data["score"].mean())).mean()
    improvement = (baseline_mae - np.mean(fold_maes)) / baseline_mae * 100
    print(f"  Baseline MAE     {baseline_mae:.4f}")
    print(f"  Improvement      {improvement:.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
