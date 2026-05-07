"""Ridge Regression baseline beauty prediction model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BeautyModel


class RidgeBeautyModel(BeautyModel):
    name = "ridge"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.pipeline: Pipeline | None = None

    def train(self, X_train, y_train, X_val, y_val, tune=False, n_trials=200, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        if tune:
            self.params = self._tune(X_combined, y_combined, n_trials)
        else:
            self.params = {"alpha": 1.0}

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.params["alpha"], random_state=42)),
            ]
        )

        self.pipeline.fit(X_combined, y_combined)
        self.params["n_features"] = X_combined.shape[1]
        print(f"  Trained on {len(X_combined)} samples (alpha={self.params['alpha']})")
        return self.params

    def _tune(self, X, y, n_trials):
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, random_state=42)),
                ]
            )
            scores = cross_val_score(
                pipeline, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"  Optuna: {n_trials} trials, best CV MAE: {study.best_value:.4f}")
        return {"alpha": study.best_params["alpha"]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.artifacts_dir / "model.joblib", compress=3)
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "RidgeBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.pipeline = joblib.load(instance.artifacts_dir / "model.joblib")
        return instance

    def feature_importances(self) -> dict[str, float]:
        ridge = self.pipeline.named_steps["ridge"]
        scaler = self.pipeline.named_steps["scaler"]
        importance = np.abs(ridge.coef_) * scaler.scale_
        importance = importance / importance.sum()
        return dict(zip(self.feature_cols, importance.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        import shap

        explainer = shap.LinearExplainer(
            self.pipeline.named_steps["ridge"],
            self.pipeline.named_steps["scaler"].transform(X_test),
        )
        shap_values = explainer.shap_values(
            self.pipeline.named_steps["scaler"].transform(X_test)
        )
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
