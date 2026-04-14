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

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

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

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**self.params)),
        ])

        print("  Training MLP (128-64-32, adam, early stopping)...")
        self.pipeline.fit(X_combined, y_combined)

        mlp = self.pipeline.named_steps["mlp"]
        loss_info = f", best val loss: {mlp.best_loss_:.4f}" if mlp.best_loss_ is not None else ""
        print(f"  Converged in {mlp.n_iter_} iterations{loss_info}")

        return self.params

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
        bg = X_test[np.random.RandomState(42).choice(len(X_test), min(100, len(X_test)), replace=False)]
        explainer = shap.KernelExplainer(self.pipeline.predict, bg)
        shap_values = explainer.shap_values(X_test[:200], silent=True)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_cols, mean_abs.tolist()))
