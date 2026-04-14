"""XGBoost quantile regression — predicts median instead of mean.

Median regression is more robust to outliers and doesn't penalize extreme
predictions as harshly as MSE, which can reduce mean-regression.
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

from models.base import BeautyModel


class QuantileBeautyModel(BeautyModel):
    name = "quantile"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.model: xgb.XGBRegressor | None = None

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        self.params = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.5,  # median
        }

        self.model = xgb.XGBRegressor(
            random_state=42,
            early_stopping_rounds=50,
            **self.params,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"  Best iteration: {self.model.best_iteration}")

        return self.params

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.artifacts_dir / "model.json"))
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "QuantileBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(instance.artifacts_dir / "model.json"))
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
