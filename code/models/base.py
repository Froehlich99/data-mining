"""Base class for beauty prediction models."""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

FEATURE_COLS = [
    # Eyes
    "canthal_tilt",
    "eye_width_ratio",
    "eye_height_ratio",
    "eyebrow_eye_dist",
    "eye_spacing_ratio",
    "eye_area_ratio",
    "scleral_show",
    "eye_asymmetry",
    "brow_arch_height",
    # Nose
    "nose_width_ratio",
    "nose_length_ratio",
    "nose_symmetry",
    # Mouth
    "lip_width_ratio",
    "upper_lip_ratio",
    "lip_fullness_ratio",
    "mouth_width_face_ratio",
    "cupids_bow_ratio",
    "mouth_chin_ratio",
    "mouth_symmetry",
    # Face & jaw
    "facial_symmetry",
    "face_length_width_ratio",
    "jaw_width_ratio",
    "cheekbone_prominence",
    "interpupillary_ratio",
    "gonial_angle",
    "chin_taper",
    "face_taper_ratio",
    "eye_symmetry",
    # Proportions
    "midface_ratio",
    "lower_face_ratio",
    "upper_face_ratio",
    "phi_deviation",
    "facial_thirds_symmetry",
    # Expression (blendshapes)
    "expr_smile",
    "expr_frown",
    "expr_jaw_open",
    "expr_brow_up",
    "expr_brow_down",
    "expr_cheek_squint",
    "expr_eye_squint",
    "expr_eye_wide",
    "expr_mouth_pucker",
]

EXCLUDED_COLS = {"image_path", "dataset", "gender", "ethnicity", "score", "score_raw", "split", "head_roll"}


def augment_features(X: np.ndarray, y: np.ndarray, n_copies: int = 3,
                     noise_std: float = 0.02, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create augmented training data by adding gaussian noise to features.

    Simulates measurement variance from slightly different landmark positions.
    Labels stay the same — the noise is small enough that the face is still the same face.
    """
    rng = np.random.RandomState(seed)
    augmented_X = [X]
    augmented_y = [y]
    for i in range(n_copies):
        noise = rng.normal(0, noise_std, size=X.shape)
        augmented_X.append(X + noise * np.abs(X))  # proportional noise
        augmented_y.append(y)
    return np.vstack(augmented_X), np.concatenate(augmented_y)


class BeautyModel(ABC):
    """Abstract base for all beauty prediction models."""

    name: str = "base"

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.feature_cols: list[str] = list(FEATURE_COLS)
        self.dataset_stats: dict = {}
        self.std_ratio: float | None = None
        self.metrics: dict = {}
        self.params: dict = {}

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        """Train the model. Returns metrics dict."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict z-scores for feature matrix X."""

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Predict with variance calibration to reduce mean-regression."""
        z = self.predict(X)
        if self.std_ratio and self.std_ratio > 0:
            z = z / self.std_ratio
        return z

    @abstractmethod
    def save(self):
        """Persist model to artifacts_dir."""

    @classmethod
    @abstractmethod
    def load(cls, artifacts_dir: Path) -> "BeautyModel":
        """Load a saved model from artifacts_dir."""

    @abstractmethod
    def feature_importances(self) -> dict[str, float]:
        """Return feature name -> importance mapping."""

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model on test set. Returns metrics dict."""
        y_pred = self.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r, _ = pearsonr(y_test, y_pred)
        std_ratio = float(np.std(y_pred) / np.std(y_test))

        # Baseline: predict the mean
        baseline_mae = float(mean_absolute_error(y_test, np.full(len(y_test), y_test.mean())))
        improvement = (baseline_mae - mae) / baseline_mae * 100

        # Per-quartile MAE
        q25, q75 = np.percentile(y_test, [25, 75])
        bottom_mask = y_test < q25
        top_mask = y_test > q75
        mid_mask = ~bottom_mask & ~top_mask

        self.std_ratio = std_ratio
        self.metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "baseline_mae": round(baseline_mae, 4),
            "improvement_pct": round(improvement, 1),
            "pearson_r": round(float(r), 4),
            "std_ratio": round(std_ratio, 4),
            "mae_bottom_quartile": round(float(mean_absolute_error(y_test[bottom_mask], y_pred[bottom_mask])), 4),
            "mae_top_quartile": round(float(mean_absolute_error(y_test[top_mask], y_pred[top_mask])), 4),
            "mae_middle": round(float(mean_absolute_error(y_test[mid_mask], y_pred[mid_mask])), 4),
        }
        return self.metrics

    def save_metadata(self):
        """Save metadata.json alongside the model."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.name,
            "feature_cols": self.feature_cols,
            "dataset_stats": self.dataset_stats,
            "metrics": self.metrics,
            "params": self.params,
            "std_ratio": self.std_ratio,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_features": len(self.feature_cols),
        }
        with open(self.artifacts_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self):
        """Load metadata.json from artifacts_dir."""
        with open(self.artifacts_dir / "metadata.json") as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.dataset_stats = meta["dataset_stats"]
        self.metrics = meta.get("metrics", {})
        self.params = meta.get("params", {})
        self.std_ratio = meta.get("std_ratio")
