"""
CryptoForecast — Cryptocurrency Price Forecasting Package
==========================================================
Pull, store, train, and forecast crypto prices using DuckDB + Prophet/ARIMA.
"""

from cryptoforecast.ingestion.fetcher import CryptoFetcher
from cryptoforecast.storage.database import CryptoDatabase
from cryptoforecast.modeling.trainer import ForecastTrainer
from cryptoforecast.forecasting.predictor import ForecastPredictor
from cryptoforecast.config import settings

__version__ = "0.1.0"
__all__ = [
    "CryptoFetcher",
    "CryptoDatabase",
    "ForecastTrainer",
    "ForecastPredictor",
    "settings",
]
