"""Shared base class for XGBoost-based beauty models."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from models.base import BeautyModel


class XGBoostBaseModel(BeautyModel):
    """Base for models that wrap a single xgb.XGBRegressor."""

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / self.name / "artifacts"
        super().__init__(artifacts_dir)
        self.model: xgb.XGBRegressor | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.artifacts_dir / "model.json"))
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "XGBoostBaseModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(instance.artifacts_dir / "model.json"))
        return instance

    def feature_importances(self) -> dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, importances.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        """Run SHAP TreeExplainer. Returns feature -> mean |SHAP value|."""
        import shap

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
