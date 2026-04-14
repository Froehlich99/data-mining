"""
Train beauty prediction models. Wrapper for models.train.

Usage:
  uv run python scripts/train.py                        # train xgboost (default)
  uv run python scripts/train.py --model ensemble       # train stacking ensemble
  uv run python scripts/train.py --model all            # train both
  uv run python scripts/train.py --model xgboost --tune # XGBoost with Optuna
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.train import main

if __name__ == "__main__":
    main()
