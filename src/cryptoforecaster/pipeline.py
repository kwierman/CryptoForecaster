"""
pipeline.py — Programmatic entry point for the full data → forecast pipeline.

Useful for running from Python scripts, notebooks, or schedulers.

Example
-------
>>> from cryptoforecaster.pipeline import run_pipeline
>>> results = run_pipeline(coins=["bitcoin", "ethereum"], model="ensemble")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from cryptoforecaster.config import settings
from cryptoforecaster.modeling.base import BaseModel
from cryptoforecaster.ingestion.fetcher import CryptoFetcher
from cryptoforecaster.storage.database import CryptoDatabase
from cryptoforecaster.modeling.trainer import ForecastTrainer
from cryptoforecaster.forecasting.predictor import ForecastPredictor


def ingest(
    coins: Optional[List[str]] = None,
    days: int = settings.default_days,
    db: Optional[CryptoDatabase] = None,
) -> CryptoDatabase:
    """Pull data from CoinGecko and upsert into DuckDB."""
    db = db or CryptoDatabase()
    fetcher = CryptoFetcher()
    data = fetcher.fetch_all(coin_ids=coins, days=days)

    db.upsert_market_prices(data["market_charts"])
    db.upsert_ohlcv(data["ohlcv"])
    db.upsert_market_snapshot(data["snapshot"])
    return db


def train(
    coins: Optional[List[str]] = None,
    model_name: str = settings.default_model,
    db: Optional[CryptoDatabase] = None,
) -> Dict[str, BaseModel]:
    """Train forecast models for all (or specified) coins."""
    db = db or CryptoDatabase()
    trainer = ForecastTrainer(db=db, model_name=model_name)
    return trainer.train_all(coin_ids=coins)


def forecast(
    coins: Optional[List[str]] = None,
    model_name: str = settings.default_model,
    horizon: int = settings.forecast_horizon,
    db: Optional[CryptoDatabase] = None,
) -> Dict[str, pd.DataFrame]:
    """Load trained models and produce forecasts."""
    db = db or CryptoDatabase()
    predictor = ForecastPredictor(db=db, model_name=model_name, horizon=horizon)
    return predictor.forecast_all(coin_ids=coins)


def run_pipeline(
    coins: Optional[List[str]] = None,
    days: int = settings.default_days,
    model_name: str = settings.default_model,
    horizon: int = settings.forecast_horizon,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete pipeline in one call:
      1. Ingest data from CoinGecko
      2. Train forecast models
      3. Generate forecasts

    Returns
    -------
    dict mapping coin_id → forecast DataFrame
    """
    logger.info("=" * 60)
    logger.info("CryptoForecast Pipeline Starting")
    logger.info("=" * 60)

    db = CryptoDatabase()

    logger.info("Step 1/3 — Ingestion")
    ingest(coins=coins, days=days, db=db)

    logger.info("Step 2/3 — Training")
    train(coins=coins, model_name=model_name, db=db)

    logger.info("Step 3/3 — Forecasting")
    forecasts = forecast(coins=coins, model_name=model_name, horizon=horizon, db=db)

    logger.success(f"Pipeline complete. {len(forecasts)} coins forecasted.")
    return forecasts
