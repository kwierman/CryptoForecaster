from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field
from pydantic import ConfigDict


class MarketPrice(SQLModel, table=True):
    __tablename__ = "market_prices"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: str = Field(index=True)
    symbol: str
    currency: str = Field(default="USD")
    timestamp: datetime = Field(index=True)
    price: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class OHLCV(SQLModel, table=True):
    __tablename__ = "ohlcv"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: str = Field(index=True)
    symbol: str
    timestamp: datetime = Field(index=True)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class MarketSnapshot(SQLModel, table=True):
    __tablename__ = "market_snapshot"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: Optional[str] = None
    symbol: Optional[str] = None
    name: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    total_volume: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    price_change_pct_1h: Optional[float] = None
    price_change_pct_24h: Optional[float] = None
    price_change_pct_7d: Optional[float] = None
    price_change_pct_30d: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    ath: Optional[float] = None
    ath_date: Optional[str] = None
    fetched_at: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class Forecast(SQLModel, table=True):
    __tablename__ = "forecasts"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: str = Field(index=True)
    symbol: Optional[str] = None
    model_name: str
    model_version: Optional[str] = None
    timestamp: datetime = Field(index=True)
    forecast: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    is_future: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class ModelRegistry(SQLModel, table=True):
    __tablename__ = "model_registry"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: str = Field(index=True)
    model_name: str
    model_version: str
    model_path: Optional[str] = None
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    metrics: Optional[str] = None
    hyperparams: Optional[str] = None
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id: Optional[int] = Field(default=None, primary_key=True)
    coin_id: str = Field(index=True)
    symbol: str
    timestamp: datetime = Field(index=True)
    action: str
    quantity: float
    price: float
    portfolio_value: float
    strategy_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]


class PortfolioSnapshot(SQLModel, table=True):
    __tablename__ = "portfolio_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(index=True)
    total_value: float
    holdings: str
    cash: float
    strategy_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
