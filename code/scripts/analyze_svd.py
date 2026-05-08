# %% [markdown]
# # SVD Feature Analysis
#
# Decomposes the feature matrix via PCA/SVD to understand:
# 1. How quickly explained variance drops off (scree plot)
# 2. Which original features load onto the top components (loadings heatmap)
# 3. Which features contribute most overall (contribution ranking)
# 4. Whether high-variance components are also predictive of the target (component-target correlation)

# %%
"""Setup and data loading."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.base import FEATURE_COLS

# Suppress sklearn matmul warnings (numpy 2.x compat noise, results unaffected)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_CSV = DATA_DIR / "features.csv"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# %%
"""Load and prepare data."""

df = pd.read_csv(FEATURES_CSV)
X = df[FEATURE_COLS].values
y = df["score"].values

# Drop rows with NaN or Inf features
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

# Clip extreme outliers (beyond 10 std from mean) to avoid overflow in matmul
for col in range(X.shape[1]):
    col_mean = np.mean(X[:, col])
    col_std = np.std(X[:, col])
    if col_std > 0:
        X[:, col] = np.clip(X[:, col], col_mean - 10 * col_std, col_mean + 10 * col_std)

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Standardize (zero mean, unit variance) so SVD isn't dominated by scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
"""Run PCA (SVD under the hood)."""

pca = PCA(n_components=X_scaled.shape[1])
pca.fit(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components = len(explained_var)

# Key thresholds
n_90 = int(np.searchsorted(cumulative_var, 0.90)) + 1
n_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

# %%
"""Scree plot: explained variance per component + cumulative."""

fig, ax1 = plt.subplots(figsize=(10, 5))

# Bar chart: individual variance
ax1.bar(
    range(1, n_components + 1),
    explained_var * 100,
    alpha=0.7,
    color="steelblue",
    label="Individual",
)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_xticks(range(1, n_components + 1))
ax1.set_xticklabels(range(1, n_components + 1), fontsize=7)

# Cumulative line on secondary axis
ax2 = ax1.twinx()
ax2.plot(
    range(1, n_components + 1),
    cumulative_var * 100,
    color="darkorange",
    marker="o",
    markersize=4,
    linewidth=2,
    label="Cumulative",
)
ax2.set_ylabel("Cumulative Variance (%)")
ax2.set_ylim(0, 105)

# Mark 90% and 95% thresholds
ax2.axhline(90, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.axhline(95, color="darkred", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.axvline(n_90, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
ax2.axvline(n_95, color="darkred", linestyle=":", linewidth=0.8, alpha=0.7)

ax2.annotate(
    f"90% at PC{n_90}",
    xy=(n_90, 90),
    xytext=(n_90 + 1.5, 85),
    fontsize=8,
    color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
)
ax2.annotate(
    f"95% at PC{n_95}",
    xy=(n_95, 95),
    xytext=(n_95 + 1.5, 98),
    fontsize=8,
    color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

ax1.set_title("Scree Plot: Explained Variance by Principal Component")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "scree_plot.png")
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'scree_plot.png'}")

# %%
"""Loading heatmap: top 10 components x all features."""

loadings = pca.components_  # shape: (n_components, n_features)
n_show = min(10, n_components)

loading_df = pd.DataFrame(
    loadings[:n_show].T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(n_show)],
)

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(
    loading_df,
    cmap="RdBu_r",
    center=0,
    annot=False,
    linewidths=0.3,
    ax=ax,
    cbar_kws={"label": "Loading"},
)
ax.set_title("PCA Loadings: Top 10 Components")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Original Feature")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "loadings_heatmap.png")
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'loadings_heatmap.png'}")

# %%
"""Feature contribution ranking (sum of squared loadings across top-k components)."""

# Use components that explain 90% variance
top_loadings = loadings[:n_90]  # shape: (n_90, n_features)
contributions = np.sum(top_loadings**2, axis=0)  # per feature

# Normalize to percentage
contributions_pct = contributions / contributions.sum() * 100

# Sort
sort_idx = np.argsort(contributions_pct)[::-1]
sorted_features = [FEATURE_COLS[i] for i in sort_idx]
sorted_contributions = contributions_pct[sort_idx]

# Bar chart
fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(range(len(sorted_features)), sorted_contributions, color="steelblue", alpha=0.8)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Contribution (%)")
ax.set_title(f"Feature Contribution to Top {n_90} PCs (90% variance)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "feature_contributions.png")
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'feature_contributions.png'}")

# Print table
print(f"\n{'Feature':<30} {'Contribution %':>15}")
print("-" * 47)
for feat, contrib in zip(sorted_features, sorted_contributions):
    print(f"{feat:<30} {contrib:>14.2f}%")

# %%
"""Component-target correlation: are high-variance PCs also predictive?"""

X_pca = pca.transform(X_scaled)  # project all data onto PCs

correlations = []
p_values = []
for i in range(n_components):
    r, p = pearsonr(X_pca[:, i], y)
    correlations.append(r)
    p_values.append(p)

correlations = np.array(correlations)
p_values = np.array(p_values)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if p < 0.05 else "lightgray" for p in p_values]
ax.bar(range(1, n_components + 1), np.abs(correlations), color=colors, alpha=0.8)
ax.set_xlabel("Principal Component")
ax.set_ylabel("|Pearson r| with target score")
ax.set_xticks(range(1, n_components + 1))
ax.set_xticklabels(range(1, n_components + 1), fontsize=7)
ax.set_title("Component-Target Correlation (blue = p < 0.05)")

# Annotate top correlated components
top_corr_idx = np.argsort(np.abs(correlations))[::-1][:5]
for idx in top_corr_idx:
    if np.abs(correlations[idx]) > 0.05:
        ax.annotate(
            f"PC{idx+1}\nr={correlations[idx]:.3f}",
            xy=(idx + 1, np.abs(correlations[idx])),
            xytext=(idx + 1, np.abs(correlations[idx]) + 0.02),
            fontsize=7,
            ha="center",
        )

fig.tight_layout()
fig.savefig(PLOTS_DIR / "component_target_correlation.png")
plt.close(fig)
print(f"\nSaved: {PLOTS_DIR / 'component_target_correlation.png'}")

# Print high-variance but low-predictive components
print(f"\n{'PC':<5} {'Var%':>6} {'|r|':>6} {'p-value':>10} {'Note'}")
print("-" * 45)
for i in range(n_components):
    note = ""
    if explained_var[i] > 0.05 and abs(correlations[i]) < 0.05:
        note = "HIGH VAR, LOW PRED"
    elif explained_var[i] < 0.03 and abs(correlations[i]) > 0.05:
        note = "LOW VAR, HIGH PRED"
    if note or i < 10:
        print(
            f"PC{i+1:<3} {explained_var[i]*100:>5.1f}% {abs(correlations[i]):>5.3f} "
            f"{p_values[i]:>10.2e} {note}"
        )

# %%
"""Summary: combined feature ranking (SVD contribution + direct target correlation)."""

# Direct correlation of each original feature with target
direct_corr = np.array([abs(pearsonr(X_scaled[:, i], y)[0]) for i in range(len(FEATURE_COLS))])

# Build summary dataframe
summary = pd.DataFrame(
    {
        "feature": FEATURE_COLS,
        "svd_contribution_pct": contributions_pct,
        "direct_corr_with_target": direct_corr,
    }
)

# Rank both metrics (lower rank = more important)
summary = summary.assign(
    svd_rank=lambda d: d["svd_contribution_pct"].rank(ascending=False).astype(int),
    corr_rank=lambda d: d["direct_corr_with_target"].rank(ascending=False).astype(int),
)
summary = summary.assign(
    combined_rank=lambda d: (d["svd_rank"] + d["corr_rank"]) / 2,
)
summary = summary.sort_values("combined_rank")

print("\n" + "=" * 75)
print("SUMMARY: Feature Importance (SVD contribution + target correlation)")
print("=" * 75)
print(
    f"{'Feature':<28} {'SVD%':>6} {'SVDRk':>5} {'|r|':>6} {'CorrRk':>6} {'CombRk':>7}"
)
print("-" * 75)
for _, row in summary.iterrows():
    print(
        f"{row['feature']:<28} {row['svd_contribution_pct']:>5.1f}% "
        f"{row['svd_rank']:>5} {row['direct_corr_with_target']:>5.3f} "
        f"{row['corr_rank']:>6} {row['combined_rank']:>7.1f}"
    )

# Highlight candidates to drop (bottom quartile on both)
n_features = len(FEATURE_COLS)
drop_candidates = summary[
    (summary["svd_rank"] > n_features * 0.75)
    & (summary["corr_rank"] > n_features * 0.75)
]

if not drop_candidates.empty:
    print(f"\nDrop candidates (bottom quartile on BOTH metrics):")
    for _, row in drop_candidates.iterrows():
        print(f"  - {row['feature']}")
else:
    print("\nNo features are in the bottom quartile on both metrics.")

print(f"\nAll plots saved to: {PLOTS_DIR}/")
