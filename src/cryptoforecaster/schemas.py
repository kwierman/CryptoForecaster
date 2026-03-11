from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class CoinInput(BaseModel):
    coin_id: str = Field(..., description="CoinGecko coin ID")
    symbol: str = Field(..., description="Trading symbol (e.g., BTC)")


class IngestInput(BaseModel):
    coins: List[str] = Field(
        default_factory=lambda: ["bitcoin", "ethereum"],
        description="List of coin IDs to fetch",
    )
    days: int = Field(
        default=365, ge=1, le=365, description="Number of days of historical data"
    )

    @field_validator("coins")
    @classmethod
    def validate_coins(cls, v):
        if not v:
            raise ValueError("At least one coin must be specified")
        return v


class TrainInput(BaseModel):
    coin: str = Field(..., description="Coin ID to train on")
    model_type: str = Field(
        default="prophet", description="Model type: prophet, arima, ensemble"
    )
    train_start: Optional[datetime] = Field(
        default=None, description="Training start date"
    )
    train_end: Optional[datetime] = Field(default=None, description="Training end date")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        allowed = ["prophet", "arima", "ensemble"]
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


class PredictInput(BaseModel):
    coins: List[str] = Field(..., description="List of coin IDs to predict")
    horizon: int = Field(
        default=30, ge=1, le=90, description="Number of days to forecast"
    )
    model_name: Optional[str] = Field(default=None, description="Specific model to use")

    @field_validator("coins")
    @classmethod
    def validate_coins(cls, v):
        if not v:
            raise ValueError("At least one coin must be specified")
        return v


class BacktestInput(BaseModel):
    coins: List[str] = Field(..., description="List of coin IDs to trade")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(
        default=10000.0, gt=0, description="Initial capital in USD"
    )
    strategy: str = Field(
        default="momentum",
        description="Strategy: momentum, mean_reversion, trend_following",
    )

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v, info):
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class StrategyParams(BaseModel):
    momentum_threshold: float = Field(
        default=0.02, description="Price change threshold for momentum"
    )
    rsi_oversold: float = Field(default=30, description="RSI oversold threshold")
    rsi_overbought: float = Field(default=70, description="RSI overbought threshold")
    short_window: int = Field(default=20, description="Short moving average window")
    long_window: int = Field(default=50, description="Long moving average window")
    position_size: float = Field(
        default=0.1, gt=0, le=1, description="Position size as fraction of portfolio"
    )
    stop_loss: float = Field(
        default=0.05, gt=0, le=1, description="Stop loss percentage"
    )
    take_profit: float = Field(default=0.15, gt=0, description="Take profit percentage")


class PortfolioAllocation(BaseModel):
    coin: str
    weight: float = Field(..., ge=0, le=1)

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Weight must be between 0 and 1")
        return v


class TradeSignal(BaseModel):
    timestamp: datetime
    coin_id: str
    action: str = Field(..., description="buy, sell, or hold")
    quantity: float = 0
    price: float = 0
    reason: str = ""

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        allowed = ["buy", "sell", "hold"]
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v


class BacktestResult(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    portfolio_value: float
    holdings: dict = {}
    trades: List[dict] = []
