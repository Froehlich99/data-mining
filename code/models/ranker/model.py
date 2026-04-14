"""XGBoost with rank-transformed targets.

Instead of predicting z-scores directly, this model predicts percentile ranks
(0-1). This forces the model to use the full output range because every
training sample maps to a unique position in [0, 1]. At inference time,
predictions are mapped back to z-scores via the inverse of the training CDF.

This naturally fights mean-regression: the model MUST spread predictions
across the full range to match the uniform-ish rank distribution.
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

from models.base import BeautyModel


class RankerBeautyModel(BeautyModel):
    name = "ranker"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.model: xgb.XGBRegressor | None = None
        # Sorted training z-scores for inverse CDF mapping
        self._sorted_train_scores: np.ndarray | None = None

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        # Store original scores for inverse mapping
        self._sorted_train_scores = np.sort(y_combined)

        # Transform targets to percentile ranks
        y_ranks_train = self._to_ranks(y_train, y_combined)
        y_ranks_val = self._to_ranks(y_val, y_combined)

        self.params = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 50,
        }

        self.model = xgb.XGBRegressor(random_state=42, **self.params)
        self.model.fit(X_train, y_ranks_train, eval_set=[(X_val, y_ranks_val)], verbose=False)
        print(f"  Best iteration: {self.model.best_iteration}")

        return self.params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict percentile ranks, then map back to z-scores."""
        ranks = self.model.predict(X)
        ranks = np.clip(ranks, 0, 1)
        return self._ranks_to_scores(ranks)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.artifacts_dir / "model.json"))
        np.save(self.artifacts_dir / "sorted_scores.npy", self._sorted_train_scores)
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "RankerBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(instance.artifacts_dir / "model.json"))
        instance._sorted_train_scores = np.load(instance.artifacts_dir / "sorted_scores.npy")
        return instance

    def feature_importances(self) -> dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, importances.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))

    @staticmethod
    def _to_ranks(y: np.ndarray, y_reference: np.ndarray) -> np.ndarray:
        """Convert scores to percentile ranks relative to a reference distribution."""
        sorted_ref = np.sort(y_reference)
        ranks = np.searchsorted(sorted_ref, y, side="right") / len(sorted_ref)
        return ranks

    def _ranks_to_scores(self, ranks: np.ndarray) -> np.ndarray:
        """Map predicted percentile ranks back to z-scores via inverse CDF."""
        n = len(self._sorted_train_scores)
        indices = np.clip((ranks * n).astype(int), 0, n - 1)
        return self._sorted_train_scores[indices]
