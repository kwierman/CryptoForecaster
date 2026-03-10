"""
Global configuration and settings for CryptoForecast.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = str(DATA_DIR / "crypto.duckdb")

    # ── Supported coins (CoinGecko IDs) ──────────────────────────────────────
    default_coins: List[str] = field(default_factory=lambda: [
        "bitcoin",
        "ethereum",
        "binancecoin",
        "solana",
        "ripple",
        "cardano",
        "dogecoin",
        "avalanche-2",
        "polkadot",
        "chainlink",
    ])

    # Coin display symbols map
    coin_symbols: dict = field(default_factory=lambda: {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "binancecoin": "BNB",
        "solana": "SOL",
        "ripple": "XRP",
        "cardano": "ADA",
        "dogecoin": "DOGE",
        "avalanche-2": "AVAX",
        "polkadot": "DOT",
        "chainlink": "LINK",
    })

    # ── Data ingestion ────────────────────────────────────────────────────────
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    default_vs_currency: str = "usd"
    default_days: int = 365          # days of historical data to pull
    request_timeout: int = 30
    request_delay: float = 1.5       # seconds between API calls (rate limiting)

    # ── Modeling ──────────────────────────────────────────────────────────────
    models_dir: str = str(MODELS_DIR)
    default_model: str = "prophet"   # "prophet" | "arima" | "ensemble"
    forecast_horizon: int = 30       # days ahead to forecast
    train_test_split: float = 0.85   # fraction of data for training

    # Prophet hyperparameters
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "multiplicative"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True

    # ARIMA order
    arima_order: tuple = (5, 1, 0)
    arima_seasonal_order: tuple = (1, 1, 0, 7)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = str(LOGS_DIR / "cryptoforecast.log")


settings = Settings()
