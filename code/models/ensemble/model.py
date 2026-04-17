"""Stacking ensemble beauty prediction model."""

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BeautyModel


class StackingBeautyModel(BeautyModel):
    name = "ensemble"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.model: StackingRegressor | None = None

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        base_estimators = [
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
            ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ]

        self.model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )

        print("  Training stacking ensemble (4 base models x 5-fold CV)...")
        self.model.fit(X_combined, y_combined)

        self.params = {
            "base_estimators": ["xgb", "rf", "gbr", "ridge"],
            "meta_learner": "ridge",
            "cv_folds": 5,
            "n_train": len(X_combined),
        }
        return self.params

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.artifacts_dir / "model.joblib", compress=3)
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "StackingBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = joblib.load(instance.artifacts_dir / "model.joblib")
        return instance

    def feature_importances(self) -> dict[str, float]:
        """Average feature importances from tree-based base models, weighted by meta-learner."""
        meta_coefs = self.model.final_estimator_.coef_
        n_features = len(self.feature_cols)
        weighted_importance = np.zeros(n_features)
        total_weight = 0.0

        for i, (name, estimator) in enumerate(self.model.named_estimators_.items()):
            if hasattr(estimator, "feature_importances_"):
                weighted_importance += estimator.feature_importances_ * abs(
                    meta_coefs[i]
                )
                total_weight += abs(meta_coefs[i])

        if total_weight > 0:
            weighted_importance /= total_weight

        return dict(zip(self.feature_cols, weighted_importance.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        """SHAP via TreeExplainer on the XGBoost base model (fastest, most informative)."""
        import shap

        xgb_model = self.model.named_estimators_["xgb"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
