"""MLP (neural network) beauty prediction model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BeautyModel


class MLPBeautyModel(BeautyModel):
    name = "mlp"

    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        super().__init__(artifacts_dir)
        self.pipeline: Pipeline | None = None

    def train(self, X_train, y_train, X_val, y_val, tune=False, n_trials=200) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        if tune:
            self.params = self._tune(X_combined, y_combined, n_trials)
        else:
            self.params = {
                "hidden_layer_sizes": (128, 64, 32),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.01,
                "batch_size": 64,
                "learning_rate": "adaptive",
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 30,
                "random_state": 42,
            }

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(**self.params)),
            ]
        )

        layers = self.params["hidden_layer_sizes"]
        print(
            f"  Training MLP ({'-'.join(str(l) for l in layers)}, adam, early stopping)..."
        )
        self.pipeline.fit(X_combined, y_combined)

        mlp = self.pipeline.named_steps["mlp"]
        loss_info = (
            f", best val loss: {mlp.best_loss_:.4f}"
            if mlp.best_loss_ is not None
            else ""
        )
        print(f"  Converged in {mlp.n_iter_} iterations{loss_info}")

        return self.params

    def _tune(self, X, y, n_trials):
        """Optuna hyperparameter search for MLP."""
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        LAYER_CHOICES = [
            (64, 32),
            (128, 64),
            (128, 64, 32),
            (256, 128),
            (256, 128, 64),
        ]

        def objective(trial):
            layers_idx = trial.suggest_categorical(
                "hidden_layer_sizes_idx",
                list(range(len(LAYER_CHOICES))),
            )
            params = {
                "hidden_layer_sizes": LAYER_CHOICES[layers_idx],
                "activation": "relu",
                "solver": "adam",
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "learning_rate": "adaptive",
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init",
                    1e-4,
                    0.01,
                    log=True,
                ),
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 30,
                "random_state": 42,
            }
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(**params)),
                ]
            )
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring="neg_mean_absolute_error",
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"  Optuna: {n_trials} trials, best CV MAE: {study.best_value:.4f}")

        # Reconstruct best params from trial
        best = study.best_params
        return {
            "hidden_layer_sizes": LAYER_CHOICES[best["hidden_layer_sizes_idx"]],
            "activation": "relu",
            "solver": "adam",
            "alpha": best["alpha"],
            "batch_size": best["batch_size"],
            "learning_rate": "adaptive",
            "learning_rate_init": best["learning_rate_init"],
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 30,
            "random_state": 42,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.artifacts_dir / "model.joblib", compress=3)
        self.save_metadata()
        print(f"  Saved to {self.artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "MLPBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.pipeline = joblib.load(instance.artifacts_dir / "model.joblib")
        return instance

    def feature_importances(self) -> dict[str, float]:
        """Permutation-based importance approximation using first-layer weights."""
        mlp = self.pipeline.named_steps["mlp"]
        scaler = self.pipeline.named_steps["scaler"]
        # Weight magnitude of first layer, scaled by feature std
        w = np.abs(mlp.coefs_[0])  # shape: (n_features, hidden_size)
        importance = w.mean(axis=1) * scaler.scale_
        importance = importance / importance.sum()
        return dict(zip(self.feature_cols, importance.tolist()))

    def shap_analysis(self, X_test: np.ndarray) -> dict[str, float]:
        """SHAP via KernelExplainer (slow but model-agnostic)."""
        import shap

        # Use a small background set to keep it tractable
        bg = X_test[
            np.random.RandomState(42).choice(
                len(X_test), min(100, len(X_test)), replace=False
            )
        ]
        explainer = shap.KernelExplainer(self.pipeline.predict, bg)
        shap_values = explainer.shap_values(X_test[:200], silent=True)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
