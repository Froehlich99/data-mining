"""XGBoost beauty prediction model."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from models.xgboost_base import XGBoostBaseModel


class XGBoostBeautyModel(XGBoostBaseModel):
    name = "xgboost"

    def train(self, X_train, y_train, X_val, y_val, tune=False, n_trials=200) -> dict:
        if tune:
            self.params = self._tune(X_train, y_train, X_val, y_val, n_trials)
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            self.model = xgb.XGBRegressor(random_state=42, **self.params)
            self.model.fit(X_combined, y_combined)
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
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            print(f"  Best iteration: {self.model.best_iteration}")

        return self.params

    def _tune(self, X_train, y_train, X_val, y_val, n_trials):
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
