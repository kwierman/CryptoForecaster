"""
ForecastPredictor — loads trained models from disk and produces
forecast DataFrames for every time point in the ingested data.

This is the "deployment" layer: given a trained model, it:
  1. Loads the latest persisted model for each coin
  2. Runs predict() for the full history + future horizon
  3. Joins actuals with forecasts for comparison
  4. Persists results to the forecasts table in DuckDB
"""

from __future__ import annotations

from typing import Dict, List, Optional

import joblib
import pandas as pd
from loguru import logger

from cryptoforecast.config import settings
from cryptoforecast.storage.database import CryptoDatabase


class ForecastPredictor:
    """
    Load trained models and produce forecasts for every coin.

    Example
    -------
    >>> predictor = ForecastPredictor()
    >>> df = predictor.forecast("bitcoin")
    >>> predictor.forecast_all()
    """

    def __init__(
        self,
        db: Optional[CryptoDatabase] = None,
        model_name: str = settings.default_model,
        horizon: int = settings.forecast_horizon,
    ):
        self.db         = db or CryptoDatabase()
        self.model_name = model_name
        self.horizon    = horizon

    # ── Public API ────────────────────────────────────────────────────────

    def load_model(self, coin_id: str):
        """
        Load the latest trained model for a coin from disk.
        Returns the model object.
        """
        record = self.db.get_latest_model(coin_id, self.model_name)
        if not record:
            raise FileNotFoundError(
                f"No trained model found for {coin_id}/{self.model_name}. "
                "Run ForecastTrainer.train() first."
            )
        path = record["model_path"]
        logger.info(f"Loading {self.model_name} for {coin_id} from {path}")
        return joblib.load(path)

    def forecast(
        self,
        coin_id: str,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Produce a forecast DataFrame for a single coin.

        Returns a DataFrame with columns:
            coin_id, symbol, timestamp, price (actual),
            forecast, lower_bound, upper_bound, is_future
        """
        model = self.load_model(coin_id)
        fc_df = model.predict(horizon=self.horizon, include_history=True)

        # Join with actuals
        actuals = self.db.get_price_series(coin_id)[["timestamp", "price"]]
        actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], utc=True)
        fc_df["timestamp"]   = pd.to_datetime(fc_df["timestamp"], utc=True)

        result = fc_df.merge(actuals, on="timestamp", how="left")
        result = result.sort_values("timestamp").reset_index(drop=True)

        if save_to_db:
            cols = [
                "coin_id", "symbol", "model_name", "model_version",
                "timestamp", "forecast", "lower_bound", "upper_bound", "is_future",
            ]
            self.db.upsert_forecasts(result[cols])

        logger.success(
            f"Forecast complete for {coin_id}: "
            f"{result['is_future'].sum()} future, "
            f"{(~result['is_future']).sum()} historical rows"
        )
        return result

    def forecast_all(
        self,
        coin_ids: Optional[List[str]] = None,
        save_to_db: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run forecast() for every coin that has a trained model.

        Returns
        -------
        dict mapping coin_id → forecast DataFrame
        """
        coins = coin_ids or self.db.get_all_coins()
        results: Dict[str, pd.DataFrame] = {}

        for coin in coins:
            try:
                results[coin] = self.forecast(coin, save_to_db=save_to_db)
            except FileNotFoundError:
                logger.warning(f"No trained model for {coin}. Skipping.")
            except Exception as exc:
                logger.error(f"Forecast failed for {coin}: {exc}")

        logger.success(f"Forecasting complete: {len(results)}/{len(coins)} coins.")
        return results

    def get_current_forecast(self, coin_id: str) -> pd.DataFrame:
        """
        Return the most recent saved forecast for a coin from DuckDB.
        """
        return self.db.get_forecasts(coin_id, self.model_name)

    def compare_actuals(self, coin_id: str) -> pd.DataFrame:
        """
        Pull saved forecasts and join with actuals for evaluation.
        Returns a DataFrame useful for plotting / reporting.
        """
        fc  = self.db.get_forecasts(coin_id, self.model_name)
        act = self.db.get_price_series(coin_id)[["timestamp", "price"]]

        fc["timestamp"]  = pd.to_datetime(fc["timestamp"], utc=True)
        act["timestamp"] = pd.to_datetime(act["timestamp"], utc=True)

        merged = fc.merge(act, on="timestamp", how="left")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged
