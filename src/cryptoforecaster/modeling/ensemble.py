"""
EnsembleModel — weighted average of Prophet + ARIMA forecasts.

Weights are determined by inverse-MAPE on a held-out validation split,
so the better-performing model for a given coin gets higher weight.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from cryptoforecaster.config import settings
from cryptoforecaster.modeling.base import BaseModel
from cryptoforecaster.modeling.arima_model import ARIMAModel
from cryptoforecaster.modeling.prophet_model import ProphetModel


class EnsembleModel(BaseModel):
    """
    Inverse-MAPE weighted ensemble of Prophet + ARIMA.
    """

    name = "ensemble"

    def __init__(self, coin_id: str, val_split: float = 0.1, **kwargs):
        super().__init__(coin_id, val_split=val_split, **kwargs)
        self.val_split   = val_split
        self._prophet    = ProphetModel(coin_id)
        self._arima      = ARIMAModel(coin_id)
        self._w_prophet  = 0.5
        self._w_arima    = 0.5

    def fit(self, df: pd.DataFrame) -> "EnsembleModel":
        df = self.prepare_series(df)
        self.train_start = df["timestamp"].min().to_pydatetime()
        self.train_end   = df["timestamp"].max().to_pydatetime()

        # Split off a small validation set to compute weights
        n_val  = max(int(len(df) * self.val_split), 7)
        df_tr  = df.iloc[:-n_val]
        df_val = df.iloc[-n_val:]

        logger.info(f"[Ensemble] Fitting Prophet + ARIMA for {self.coin_id}…")
        self._prophet.fit(df_tr)
        self._arima.fit(df_tr)

        # Predict over validation window
        p_fc = self._prophet.predict(horizon=n_val, include_history=False)
        a_fc = self._arima.predict(horizon=n_val, include_history=False)

        y_true = df_val["price"].values

        def mape(y_pred):
            mask = y_true != 0
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

        p_mape = mape(p_fc["forecast"].values[: n_val])
        a_mape = mape(a_fc["forecast"].values[: n_val])
        logger.info(f"[Ensemble] val MAPE — Prophet={p_mape:.2%}, ARIMA={a_mape:.2%}")

        # Inverse-MAPE weights
        eps = 1e-6
        inv_p = 1.0 / (p_mape + eps)
        inv_a = 1.0 / (a_mape + eps)
        total = inv_p + inv_a
        self._w_prophet = inv_p / total
        self._w_arima   = inv_a / total
        logger.info(f"[Ensemble] weights — Prophet={self._w_prophet:.3f}, ARIMA={self._w_arima:.3f}")

        # Refit on full data
        self._prophet.fit(df)
        self._arima.fit(df)

        self.metrics = {
            "prophet_mape": p_mape,
            "arima_mape":   a_mape,
            "prophet_weight": self._w_prophet,
            "arima_weight":   self._w_arima,
        }
        self._is_fitted = True
        logger.success(f"[Ensemble] {self.coin_id} fitted ✓")
        return self

    def predict(
        self,
        horizon: int = settings.forecast_horizon,
        include_history: bool = True,
    ) -> pd.DataFrame:
        self.check_fitted()

        p_df = self._prophet.predict(horizon=horizon, include_history=include_history)
        a_df = self._arima.predict(horizon=horizon, include_history=include_history)

        # Align on timestamp
        merged = p_df[["timestamp", "forecast", "lower_bound", "upper_bound", "is_future"]].merge(
            a_df[["timestamp", "forecast", "lower_bound", "upper_bound"]],
            on="timestamp",
            suffixes=("_p", "_a"),
        )

        merged["forecast"]    = (self._w_prophet * merged["forecast_p"]
                                 + self._w_arima  * merged["forecast_a"])
        merged["lower_bound"] = (self._w_prophet * merged["lower_bound_p"].fillna(merged["forecast_p"])
                                 + self._w_arima  * merged["lower_bound_a"].fillna(merged["forecast_a"]))
        merged["upper_bound"] = (self._w_prophet * merged["upper_bound_p"].fillna(merged["forecast_p"])
                                 + self._w_arima  * merged["upper_bound_a"].fillna(merged["forecast_a"]))

        result = merged[["timestamp", "forecast", "lower_bound", "upper_bound", "is_future"]].copy()
        result["coin_id"]       = self.coin_id
        result["symbol"]        = settings.coin_symbols.get(self.coin_id, self.coin_id.upper())
        result["model_name"]    = self.name
        result["model_version"] = self.version
        return result

    def save(self, path: Optional[str] = None) -> str:
        self.check_fitted()
        if path is None:
            path = os.path.join(
                settings.models_dir,
                f"{self.coin_id}_{self.name}_{self.version}.joblib",
            )
        joblib.dump(self, path)
        logger.info(f"[Ensemble] Model saved → {path}")
        return path

    def load(self, path: str) -> "EnsembleModel":
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self
