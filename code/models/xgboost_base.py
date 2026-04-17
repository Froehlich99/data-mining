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

    def _tune(self, X_train, y_train, X_val, y_val, n_trials, extra_params=None):
        """Optuna hyperparameter search for XGBoost-based models.

        Args:
            extra_params: Fixed params to add to every trial (e.g. objective).
        """
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "random_state": 42,
            }
            if extra_params:
                params.update(extra_params)

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(
                model,
                X_combined,
                y_combined,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"  Optuna: {n_trials} trials, best CV MAE: {study.best_value:.4f}")
        return study.best_params
