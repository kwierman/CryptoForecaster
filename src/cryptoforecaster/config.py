import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_", case_sensitive=False)

    db_path: str = str(DATA_DIR / "crypto.duckdb")
    default_coins: List[str] = Field(
        default_factory=lambda: [
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
        ]
    )
    coin_symbols: dict = Field(
        default_factory=lambda: {
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
        }
    )
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    default_vs_currency: str = "usd"
    default_days: int = 365
    request_timeout: int = 30
    request_delay: float = 1.5
    models_dir: str = str(MODELS_DIR)
    default_model: str = "prophet"
    forecast_horizon: int = 30
    train_test_split: float = 0.85
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "multiplicative"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    arima_order: tuple = (5, 1, 0)
    arima_seasonal_order: tuple = (1, 1, 0, 7)
    log_level: str = "INFO"
    log_file: str = str(LOGS_DIR / "cryptoforecaster.log")


settings = Settings()
