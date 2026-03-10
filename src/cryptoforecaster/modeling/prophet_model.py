"""
ProphetModel — Facebook/Meta Prophet time-series forecasting.

Prophet excels at crypto data because it handles:
  - Multiple seasonalities (daily, weekly, yearly)
  - Trend changepoints (common in crypto boom/bust cycles)
  - Holiday effects (optional)
  - Multiplicative seasonality for highly volatile series
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from cryptoforecast.config import settings
from cryptoforecast.modeling.base import BaseModel


class ProphetModel(BaseModel):
    """
    Thin wrapper around the Prophet forecasting library.

    The model uses log-transform of price to handle extreme volatility
    and produce better-calibrated prediction intervals.
    """

    name = "prophet"

    def __init__(
        self,
        coin_id: str,
        changepoint_prior_scale: float = settings.prophet_changepoint_prior_scale,
        seasonality_mode: str = settings.prophet_seasonality_mode,
        yearly_seasonality: bool = settings.prophet_yearly_seasonality,
        weekly_seasonality: bool = settings.prophet_weekly_seasonality,
        **kwargs,
    ):
        super().__init__(
            coin_id,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            **kwargs,
        )
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self._model = None
        self._log_transform = True

    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Install prophet: pip install prophet")

        df = self.prepare_series(df)
        self.train_start = df["timestamp"].min().to_pydatetime()
        self.train_end   = df["timestamp"].max().to_pydatetime()

        # Prophet expects columns: ds (datetime), y (value)
        prophet_df = pd.DataFrame({
            "ds": df["timestamp"].dt.tz_localize(None),   # Prophet doesn't handle tz-aware
            "y":  np.log1p(df["price"]) if self._log_transform else df["price"],
        })

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            uncertainty_samples=500,
        )
        # Add monthly seasonality — useful for crypto cycles
        self._model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        logger.info(f"[Prophet] Fitting {self.coin_id} on {len(prophet_df)} rows…")
        self._model.fit(prophet_df)

        # In-sample metrics
        in_sample = self._model.predict(prophet_df[["ds"]])
        y_true = prophet_df["y"].values
        y_pred = in_sample["yhat"].values
        self.metrics = self._calc_metrics(
            np.expm1(y_true) if self._log_transform else y_true,
            np.expm1(y_pred) if self._log_transform else y_pred,
        )
        self._is_fitted = True
        logger.success(f"[Prophet] {self.coin_id} — MAE={self.metrics['mae']:.2f}, "
                       f"MAPE={self.metrics['mape']:.2%}")
        return self

    def predict(
        self,
        horizon: int = settings.forecast_horizon,
        include_history: bool = True,
    ) -> pd.DataFrame:
        self.check_fitted()
        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        raw    = self._model.predict(future)

        if not include_history:
            # Only return the future (beyond training end)
            raw = raw[raw["ds"] > pd.Timestamp(self.train_end)]

        y_hat  = raw["yhat"].values
        y_low  = raw["yhat_lower"].values
        y_high = raw["yhat_upper"].values

        if self._log_transform:
            y_hat  = np.expm1(y_hat)
            y_low  = np.expm1(y_low)
            y_high = np.expm1(y_high)

        # Clip negatives (log transform edge case)
        y_hat  = np.clip(y_hat, 0, None)
        y_low  = np.clip(y_low, 0, None)
        y_high = np.clip(y_high, 0, None)

        train_end_ts = pd.Timestamp(self.train_end)
        result = pd.DataFrame({
            "coin_id":       self.coin_id,
            "model_name":    self.name,
            "model_version": self.version,
            "symbol":        settings.coin_symbols.get(self.coin_id, self.coin_id.upper()),
            "timestamp":     pd.to_datetime(raw["ds"]).dt.tz_localize("UTC"),
            "forecast":      y_hat,
            "lower_bound":   y_low,
            "upper_bound":   y_high,
            "is_future":     pd.to_datetime(raw["ds"]) > train_end_ts,
        })
        return result

    def save(self, path: Optional[str] = None) -> str:
        self.check_fitted()
        if path is None:
            path = os.path.join(
                settings.models_dir,
                f"{self.coin_id}_{self.name}_{self.version}.joblib",
            )
        joblib.dump(self, path)
        logger.info(f"[Prophet] Model saved → {path}")
        return path

    def load(self, path: str) -> "ProphetModel":
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        return {"mae": mae, "rmse": rmse, "mape": mape}
