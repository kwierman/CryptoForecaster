"""
ARIMAModel — Seasonal ARIMA (SARIMA) for crypto price forecasting.

Uses statsmodels SARIMAX under the hood. Auto-differencing is applied
on log prices to achieve approximate stationarity.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from cryptoforecaster.config import settings
from cryptoforecaster.modeling.base import BaseModel

warnings.filterwarnings("ignore")


class ARIMAModel(BaseModel):
    """
    SARIMA model for daily crypto price forecasting.

    Fits on log-transformed close prices.
    Returns point forecasts + 95% prediction intervals.
    """

    name = "arima"

    def __init__(
        self,
        coin_id: str,
        order: Tuple[int, int, int] = settings.arima_order,
        seasonal_order: Tuple[int, int, int, int] = settings.arima_seasonal_order,
        **kwargs,
    ):
        super().__init__(coin_id, order=order, seasonal_order=seasonal_order, **kwargs)
        self.order         = order
        self.seasonal_order = seasonal_order
        self._model_fit    = None
        self._log_prices: Optional[pd.Series] = None
        self._dates: Optional[pd.DatetimeIndex] = None

    def fit(self, df: pd.DataFrame) -> "ARIMAModel":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("Install statsmodels: pip install statsmodels")

        df = self.prepare_series(df)
        self.train_start = df["timestamp"].min().to_pydatetime()
        self.train_end   = df["timestamp"].max().to_pydatetime()

        self._dates = pd.DatetimeIndex(
            df["timestamp"].dt.tz_localize(None), freq="D"
        )
        log_p = np.log1p(df["price"].values)
        self._log_prices = pd.Series(log_p, index=self._dates)

        logger.info(
            f"[ARIMA] Fitting {self.coin_id} — order={self.order}, "
            f"seasonal={self.seasonal_order}, n={len(log_p)}"
        )
        model = SARIMAX(
            self._log_prices,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._model_fit = model.fit(disp=False, maxiter=200)

        # In-sample metrics
        in_sample = self._model_fit.fittedvalues
        y_true = np.expm1(self._log_prices.values)
        y_pred = np.expm1(in_sample.values)
        self.metrics = self._calc_metrics(y_true, y_pred)
        self._is_fitted = True

        logger.success(
            f"[ARIMA] {self.coin_id} — MAE={self.metrics['mae']:.2f}, "
            f"MAPE={self.metrics['mape']:.2%}"
        )
        return self

    def predict(
        self,
        horizon: int = settings.forecast_horizon,
        include_history: bool = True,
    ) -> pd.DataFrame:
        self.check_fitted()

        forecast_obj = self._model_fit.get_forecast(steps=horizon)
        fc_mean      = forecast_obj.predicted_mean
        fc_ci        = forecast_obj.conf_int(alpha=0.05)

        # Future dates
        last_date  = self._dates[-1]
        future_idx = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        fc_prices = np.expm1(fc_mean.values)
        fc_low    = np.expm1(fc_ci.iloc[:, 0].values)
        fc_high   = np.expm1(fc_ci.iloc[:, 1].values)

        future_df = pd.DataFrame({
            "timestamp":   pd.DatetimeIndex(future_idx).tz_localize("UTC"),
            "forecast":    fc_prices,
            "lower_bound": np.clip(fc_low, 0, None),
            "upper_bound": fc_high,
            "is_future":   True,
        })

        if include_history:
            in_sample = self._model_fit.fittedvalues
            hist_df = pd.DataFrame({
                "timestamp":   self._dates.tz_localize("UTC"),
                "forecast":    np.expm1(in_sample.values),
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "is_future":   False,
            })
            result = pd.concat([hist_df, future_df], ignore_index=True)
        else:
            result = future_df

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
        logger.info(f"[ARIMA] Model saved → {path}")
        return path

    def load(self, path: str) -> "ARIMAModel":
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        return self

    @staticmethod
    def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        return {"mae": mae, "rmse": rmse, "mape": mape}
