"""XGBoost quantile regression — predicts median instead of mean.

Median regression is more robust to outliers and doesn't penalize extreme
predictions as harshly as MSE, which can reduce mean-regression.
"""

import xgboost as xgb

from models.xgboost_base import XGBoostBaseModel


class QuantileBeautyModel(XGBoostBaseModel):
    name = "quantile"

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> dict:
        self.params = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.5,  # median
        }

        self.model = xgb.XGBRegressor(
            random_state=42,
            early_stopping_rounds=50,
            **self.params,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"  Best iteration: {self.model.best_iteration}")

        return self.params
