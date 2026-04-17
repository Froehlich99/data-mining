"""Model registry — load any trained beauty model by name."""

import importlib
from pathlib import Path

from models.base import BeautyModel
from models.train import MODEL_REGISTRY

MODELS_ROOT = Path(__file__).resolve().parent


def load_model(name: str = "xgboost") -> BeautyModel:
    """Load a trained model by name from its artifacts directory."""
    artifacts_dir = MODELS_ROOT / name / "artifacts"
    if not (artifacts_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"No trained {name} model found at {artifacts_dir}. "
            f"Run: uv run python -m models.train --model {name}"
        )

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {', '.join(MODEL_REGISTRY)}"
        )

    module_path, class_name = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.load(artifacts_dir)


def list_models() -> list[str]:
    """Return names of models that have saved artifacts."""
    available = []
    for name in MODEL_REGISTRY:
        artifacts_dir = MODELS_ROOT / name / "artifacts"
        if (artifacts_dir / "metadata.json").exists():
            available.append(name)
    return available
