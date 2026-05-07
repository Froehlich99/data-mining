"""Decision Tree Regressor beauty prediction model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from models.base import BeautyModel


class DecisionTreeBeautyModel(BeautyModel):
    name = "decision_tree"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.model: DecisionTreeRegressor | None = None

    def train(self, X_train, y_train, X_val, y_val, tune=False, n_trials=200, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        if tune:
            self.params = self._tune(X_combined, y_combined, n_trials)
        else:
            self.params = {
                "max_depth": 7,
                "min_samples_leaf": 10,
                "min_samples_split": 10,
                "random_state": 42,
            }

        self.model = DecisionTreeRegressor(**self.params)
        self.model.fit(X_combined, y_combined)
        print(f"  Trained (depth={self.model.get_depth()}, leaves={self.model.get_n_leaves()})")
        return self.params

    def _tune(self, X, y, n_trials):
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "random_state": 42,
            }
            model = DecisionTreeRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"  Optuna: {n_trials} trials, best CV MAE: {study.best_value:.4f}")
        best = study.best_params
        best["random_state"] = 42
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.artifacts_dir / "model.joblib", compress=3)
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "DecisionTreeBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = joblib.load(instance.artifacts_dir / "model.joblib")
        return instance

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        import shap

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
