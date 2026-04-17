"""XGBoost beauty prediction model."""

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
