"""
Analyze XGBoost prediction distribution to check for regression-to-the-mean.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

FEATURES_CSV = "data/features.csv"

EXCLUDED_COLS = {"image_path", "dataset", "gender", "ethnicity", "score", "score_raw", "split", "head_roll"}

def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feature_cols].values
    y_train = train_df["score"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score"].values

    print(f"Samples: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Features: {len(feature_cols)}")
    print()

    # Train model with same params as train.py defaults
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration}")
    print()

    y_pred = model.predict(X_test)

    # --- Distribution of actual vs predicted ---
    def describe(arr, label):
        p10, p25, p75, p90 = np.percentile(arr, [10, 25, 75, 90])
        print(f"  {label}:")
        print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
        print(f"    10th pctl={p10:.4f}  90th pctl={p90:.4f}")
        print(f"    25th pctl={p25:.4f}  75th pctl={p75:.4f}")
        return p25, p75

    print("=" * 60)
    print("DISTRIBUTION COMPARISON")
    print("=" * 60)
    _, _ = describe(y_test, "Actual test scores")
    print()
    _, _ = describe(y_pred, "Predicted test scores")
    print()

    # Variance ratio
    print(f"  Variance ratio (pred/actual): {y_pred.var():.4f} / {y_test.var():.4f} = {y_pred.var() / y_test.var():.4f}")
    print(f"  Std ratio (pred/actual):      {y_pred.std():.4f} / {y_test.std():.4f} = {y_pred.std() / y_test.std():.4f}")
    print()

    # --- Correlation ---
    r, p_val = stats.pearsonr(y_test, y_pred)
    rho, _ = stats.spearmanr(y_test, y_pred)
    print(f"  Pearson r:  {r:.4f}  (p={p_val:.2e})")
    print(f"  Spearman r: {rho:.4f}")
    print()

    # --- Non-average predictions ---
    n_extreme_pred = np.sum(np.abs(y_pred) > 1.0)
    n_extreme_actual = np.sum(np.abs(y_test) > 1.0)
    print(f"  Samples with |score| > 1.0:")
    print(f"    Actual:    {n_extreme_actual} / {len(y_test)} ({100*n_extreme_actual/len(y_test):.1f}%)")
    print(f"    Predicted: {n_extreme_pred} / {len(y_pred)} ({100*n_extreme_pred/len(y_pred):.1f}%)")
    print()

    n_extreme_pred2 = np.sum(np.abs(y_pred) > 1.5)
    n_extreme_actual2 = np.sum(np.abs(y_test) > 1.5)
    print(f"  Samples with |score| > 1.5:")
    print(f"    Actual:    {n_extreme_actual2} / {len(y_test)} ({100*n_extreme_actual2/len(y_test):.1f}%)")
    print(f"    Predicted: {n_extreme_pred2} / {len(y_pred)} ({100*n_extreme_pred2/len(y_pred):.1f}%)")
    print()

    # --- MAE by quartile ---
    p25_actual = np.percentile(y_test, 25)
    p75_actual = np.percentile(y_test, 75)

    bottom_mask = y_test < p25_actual
    top_mask = y_test > p75_actual
    mid_mask = ~bottom_mask & ~top_mask

    mae_bottom = np.mean(np.abs(y_test[bottom_mask] - y_pred[bottom_mask]))
    mae_top = np.mean(np.abs(y_test[top_mask] - y_pred[top_mask]))
    mae_mid = np.mean(np.abs(y_test[mid_mask] - y_pred[mid_mask]))
    mae_all = np.mean(np.abs(y_test - y_pred))

    print("=" * 60)
    print("MAE BY ACTUAL-SCORE QUARTILE")
    print("=" * 60)
    print(f"  Bottom quartile (actual < {p25_actual:.3f}, n={bottom_mask.sum()}):  MAE = {mae_bottom:.4f}")
    print(f"  Middle 50%      ({p25_actual:.3f} to {p75_actual:.3f}, n={mid_mask.sum()}):  MAE = {mae_mid:.4f}")
    print(f"  Top quartile    (actual > {p75_actual:.3f}, n={top_mask.sum()}):  MAE = {mae_top:.4f}")
    print(f"  Overall:                                          MAE = {mae_all:.4f}")
    print()

    # --- Mean prediction by quartile (regression to mean diagnostic) ---
    print("=" * 60)
    print("MEAN PREDICTED vs MEAN ACTUAL BY QUARTILE (regression-to-mean check)")
    print("=" * 60)
    print(f"  Bottom quartile:  actual mean = {y_test[bottom_mask].mean():.4f}  predicted mean = {y_pred[bottom_mask].mean():.4f}  (shrinkage toward 0: {abs(y_pred[bottom_mask].mean()) - abs(y_test[bottom_mask].mean()):.4f})")
    print(f"  Middle 50%:       actual mean = {y_test[mid_mask].mean():.4f}  predicted mean = {y_pred[mid_mask].mean():.4f}")
    print(f"  Top quartile:     actual mean = {y_test[top_mask].mean():.4f}  predicted mean = {y_pred[top_mask].mean():.4f}  (shrinkage toward 0: {abs(y_pred[top_mask].mean()) - abs(y_test[top_mask].mean()):.4f})")
    print()

    # --- Prediction range compression summary ---
    print("=" * 60)
    print("REGRESSION-TO-THE-MEAN SUMMARY")
    print("=" * 60)
    actual_range = y_test.max() - y_test.min()
    pred_range = y_pred.max() - y_pred.min()
    print(f"  Actual range:    {actual_range:.4f}  (min={y_test.min():.4f} to max={y_test.max():.4f})")
    print(f"  Predicted range: {pred_range:.4f}  (min={y_pred.min():.4f} to max={y_pred.max():.4f})")
    print(f"  Range compression: {100*(1 - pred_range/actual_range):.1f}%")
    print(f"  Std compression:   {100*(1 - y_pred.std()/y_test.std()):.1f}%")
    print()

    if y_pred.std() / y_test.std() < 0.5:
        print("  ** SEVERE regression to the mean: predicted std is less than half of actual std **")
    elif y_pred.std() / y_test.std() < 0.75:
        print("  ** MODERATE regression to the mean: predicted std is notably compressed **")
    elif y_pred.std() / y_test.std() < 0.9:
        print("  ** MILD regression to the mean: some compression in predicted spread **")
    else:
        print("  Prediction spread looks healthy relative to actual spread.")


if __name__ == "__main__":
    main()
