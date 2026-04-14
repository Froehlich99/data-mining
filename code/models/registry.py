"""Model registry — load any trained beauty model by name."""

from pathlib import Path

from models.base import BeautyModel

MODELS_ROOT = Path(__file__).resolve().parent


def load_model(name: str = "xgboost") -> BeautyModel:
    """Load a trained model by name from its artifacts directory."""
    artifacts_dir = MODELS_ROOT / name / "artifacts"
    if not (artifacts_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"No trained {name} model found at {artifacts_dir}. "
            f"Run: uv run python -m models.train --model {name}"
        )

    if name == "xgboost":
        from models.xgboost.model import XGBoostBeautyModel
        return XGBoostBeautyModel.load(artifacts_dir)
    elif name == "ensemble":
        from models.ensemble.model import StackingBeautyModel
        return StackingBeautyModel.load(artifacts_dir)
    elif name == "mlp":
        from models.mlp.model import MLPBeautyModel
        return MLPBeautyModel.load(artifacts_dir)
    elif name == "quantile":
        from models.quantile.model import QuantileBeautyModel
        return QuantileBeautyModel.load(artifacts_dir)
    elif name == "ranker":
        from models.ranker.model import RankerBeautyModel
        return RankerBeautyModel.load(artifacts_dir)
    else:
        raise ValueError(f"Unknown model: {name}. Available: xgboost, ensemble, mlp, quantile, ranker")


def list_models() -> list[str]:
    """Return names of models that have saved artifacts."""
    available = []
    for name in ["xgboost", "ensemble", "mlp", "quantile", "ranker"]:
        artifacts_dir = MODELS_ROOT / name / "artifacts"
        if (artifacts_dir / "metadata.json").exists():
            available.append(name)
    return available
