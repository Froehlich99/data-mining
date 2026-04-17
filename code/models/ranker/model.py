"""XGBoost with rank-transformed targets.

Instead of predicting z-scores directly, this model predicts percentile ranks
(0-1). This forces the model to use the full output range because every
training sample maps to a unique position in [0, 1]. At inference time,
predictions are mapped back to z-scores via the inverse of the training CDF.

This naturally fights mean-regression: the model MUST spread predictions
across the full range to match the uniform-ish rank distribution.
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

from models.xgboost_base import XGBoostBaseModel


class RankerBeautyModel(XGBoostBaseModel):
    name = "ranker"

    def __init__(self, artifacts_dir: Path | None = None):
        super().__init__(artifacts_dir)
        # Sorted training z-scores for inverse CDF mapping
        self._sorted_train_scores: np.ndarray | None = None

    def train(self, X_train, y_train, X_val, y_val, tune=False, n_trials=200) -> dict:
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        # Store original scores for inverse mapping
        self._sorted_train_scores = np.sort(y_combined)

        # Transform targets to percentile ranks
        y_ranks_train = self._to_ranks(y_train, y_combined)
        y_ranks_val = self._to_ranks(y_val, y_combined)

        if tune:
            self.params = self._tune_ranker(
                X_train,
                y_ranks_train,
                X_val,
                y_ranks_val,
                n_trials,
            )
            self.model = xgb.XGBRegressor(random_state=42, **self.params)
            X_ranks_combined = self._to_ranks(y_combined, y_combined)
            self.model.fit(X_combined, X_ranks_combined)
            print(
                f"  Trained on train+val ({len(X_combined)} samples) with tuned params"
            )
        else:
            self.params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 50,
            }
            self.model = xgb.XGBRegressor(random_state=42, **self.params)
            self.model.fit(
                X_train,
                y_ranks_train,
                eval_set=[(X_val, y_ranks_val)],
                verbose=False,
            )
            print(f"  Best iteration: {self.model.best_iteration}")

        return self.params

    def _tune_ranker(self, X_train, y_ranks_train, X_val, y_ranks_val, n_trials):
        """Optuna tuning with rank-transformed targets already applied."""
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_ranks_train, y_ranks_val])

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict percentile ranks, then map back to z-scores."""
        ranks = self.model.predict(X)
        ranks = np.clip(ranks, 0, 1)
        return self._ranks_to_scores(ranks)

    def save(self):
        super().save()
        np.save(self.artifacts_dir / "sorted_scores.npy", self._sorted_train_scores)

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "RankerBeautyModel":
        instance = cls(artifacts_dir)
        instance.load_metadata()
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(instance.artifacts_dir / "model.json"))
        instance._sorted_train_scores = np.load(
            instance.artifacts_dir / "sorted_scores.npy"
        )
        return instance

    @staticmethod
    def _to_ranks(y: np.ndarray, y_reference: np.ndarray) -> np.ndarray:
        """Convert scores to percentile ranks relative to a reference distribution."""
        sorted_ref = np.sort(y_reference)
        ranks = np.searchsorted(sorted_ref, y, side="right") / len(sorted_ref)
        return ranks

    def _ranks_to_scores(self, ranks: np.ndarray) -> np.ndarray:
        """Map predicted percentile ranks back to z-scores via inverse CDF."""
        n = len(self._sorted_train_scores)
        indices = np.clip((ranks * n).astype(int), 0, n - 1)
        return self._sorted_train_scores[indices]
