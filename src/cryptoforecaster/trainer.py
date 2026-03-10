"""
ForecastTrainer — orchestrates model training across multiple coins.

Loads price data from DuckDB, fits a model for each coin, evaluates
on a hold-out test split, and persists the model + metrics to disk/DB.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from cryptoforecast.config import settings
from cryptoforecast.modeling.base import BaseModel
from cryptoforecast.modeling.prophet_model import ProphetModel
from cryptoforecast.modeling.arima_model import ARIMAModel
from cryptoforecast.modeling.ensemble import EnsembleModel
from cryptoforecast.storage.database import CryptoDatabase


_MODEL_REGISTRY: Dict[str, type] = {
    "prophet":  ProphetModel,
    "arima":    ARIMAModel,
    "ensemble": EnsembleModel,
}


class ForecastTrainer:
    """
    High-level trainer that:
      1. Pulls data from DuckDB for each coin
      2. Splits train/test
      3. Fits the chosen model
      4. Evaluates on test set
      5. Saves model file + registers in DB

    Example
    -------
    >>> trainer = ForecastTrainer(model_name="ensemble")
    >>> results = trainer.train_all()
    """

    def __init__(
        self,
        db: Optional[CryptoDatabase] = None,
        model_name: str = settings.default_model,
        train_test_split: float = settings.train_test_split,
    ):
        self.db              = db or CryptoDatabase()
        self.model_name      = model_name.lower()
        self.train_test_split = train_test_split

        if self.model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}"
            )
        self._model_cls = _MODEL_REGISTRY[self.model_name]

    # ── Public API ────────────────────────────────────────────────────────

    def train(self, coin_id: str) -> BaseModel:
        """Train a model for a single coin. Returns the fitted model."""
        df = self.db.get_price_series(coin_id)
        if df.empty:
            raise ValueError(f"No price data found for '{coin_id}'. Run ingestion first.")

        df_train, df_test = self._split(df)
        logger.info(
            f"Training {self.model_name} for {coin_id}: "
            f"train={len(df_train)}, test={len(df_test)}"
        )

        model = self._model_cls(coin_id=coin_id)
        model.fit(df_train)

        # Evaluate on held-out test set
        test_metrics = self._evaluate(model, df_test)
        model.metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        logger.info(
            f"{coin_id} test metrics — "
            f"MAE={test_metrics['mae']:.2f}, MAPE={test_metrics['mape']:.2%}"
        )

        # Persist model
        model_path = model.save()

        # Register in database
        self.db.register_model(
            coin_id=coin_id,
            model_name=model.name,
            model_version=model.version,
            model_path=model_path,
            train_start=model.train_start,
            train_end=model.train_end,
            metrics=model.metrics,
            hyperparams=model.hyperparams,
        )
        return model

    def train_all(
        self,
        coin_ids: Optional[List[str]] = None,
    ) -> Dict[str, BaseModel]:
        """
        Train a model for every coin in the database (or the provided list).

        Returns
        -------
        dict mapping coin_id → fitted BaseModel
        """
        coins = coin_ids or self.db.get_all_coins()
        if not coins:
            logger.warning("No coins found in database. Run ingestion first.")
            return {}

        trained: Dict[str, BaseModel] = {}
        for coin in coins:
            try:
                trained[coin] = self.train(coin)
            except Exception as exc:
                logger.error(f"Training failed for {coin}: {exc}")

        logger.success(
            f"Training complete: {len(trained)}/{len(coins)} models trained."
        )
        return trained

    # ── Helpers ───────────────────────────────────────────────────────────

    def _split(self, df: pd.DataFrame):
        n = len(df)
        split = int(n * self.train_test_split)
        return df.iloc[:split].copy(), df.iloc[split:].copy()

    @staticmethod
    def _evaluate(model: BaseModel, df_test: pd.DataFrame) -> dict:
        """
        Get predictions over the test window and compute error metrics.
        """
        import numpy as np

        n = len(df_test)
        if n == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

        fc_df = model.predict(horizon=n, include_history=False)
        # Align forecast with test dates
        test_prices = df_test["price"].values
        pred_prices = fc_df["forecast"].values[: n]

        mae  = float(np.mean(np.abs(test_prices - pred_prices)))
        rmse = float(np.sqrt(np.mean((test_prices - pred_prices) ** 2)))
        mask = test_prices != 0
        mape = float(np.mean(np.abs(
            (test_prices[mask] - pred_prices[mask]) / test_prices[mask]
        )))
        return {"mae": mae, "rmse": rmse, "mape": mape}
