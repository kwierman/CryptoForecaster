from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from cryptoforecaster.schemas import StrategyParams, TradeSignal


class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()

    def generate_signals(self, data: pd.DataFrame, coin_id: str) -> List[TradeSignal]:
        raise NotImplementedError("Subclasses must implement generate_signals")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""

    def generate_signals(self, data: pd.DataFrame, coin_id: str) -> List[TradeSignal]:
        signals = []

        if len(data) < 2:
            return signals

        prices = data.set_index("timestamp")["price"].sort_index()

        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            prev_price = prices.iloc[i - 1]
            price_change = (current_price - prev_price) / prev_price

            if price_change > self.params.momentum_threshold:
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="buy",
                        price=current_price,
                        reason=f"momentum_up_{price_change:.2%}",
                    )
                )
            elif price_change < -self.params.momentum_threshold:
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="sell",
                        price=current_price,
                        reason=f"momentum_down_{price_change:.2%}",
                    )
                )

        return signals


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy using RSI."""

    def generate_signals(self, data: pd.DataFrame, coin_id: str) -> List[TradeSignal]:
        signals = []

        if len(data) < 30:
            return signals

        prices = data.set_index("timestamp")["price"].sort_index()
        rsi = self.calculate_rsi(prices)

        for i in range(len(prices)):
            if pd.isna(rsi.iloc[i]):
                continue

            if rsi.iloc[i] < self.params.rsi_oversold:
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="buy",
                        price=prices.iloc[i],
                        reason=f"rsi_oversold_{rsi.iloc[i]:.1f}",
                    )
                )
            elif rsi.iloc[i] > self.params.rsi_overbought:
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="sell",
                        price=prices.iloc[i],
                        reason=f"rsi_overbought_{rsi.iloc[i]:.1f}",
                    )
                )

        return signals


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using dual moving averages."""

    def generate_signals(self, data: pd.DataFrame, coin_id: str) -> List[TradeSignal]:
        signals = []

        if len(data) < self.params.long_window:
            return signals

        prices = data.set_index("timestamp")["price"].sort_index()
        short_ma = self.calculate_sma(prices, self.params.short_window)
        long_ma = self.calculate_sma(prices, self.params.long_window)

        position = None
        for i in range(self.params.long_window, len(prices)):
            if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
                continue

            if short_ma.iloc[i] > long_ma.iloc[i] and position != "long":
                if position == "long":
                    pass
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="buy",
                        price=prices.iloc[i],
                        reason=f"trend_bullish_ma{self.params.short_window}_{self.params.long_window}",
                    )
                )
                position = "long"
            elif short_ma.iloc[i] < long_ma.iloc[i] and position == "long":
                signals.append(
                    TradeSignal(
                        timestamp=prices.index[i],
                        coin_id=coin_id,
                        action="sell",
                        price=prices.iloc[i],
                        reason=f"trend_bearish_ma{self.params.short_window}_{self.params.long_window}",
                    )
                )
                position = None

        return signals


class AITradingStrategy(BaseStrategy):
    """AI-based trading strategy using predictions."""

    def __init__(
        self,
        params: Optional[StrategyParams] = None,
        predictions: Optional[pd.DataFrame] = None,
    ):
        super().__init__(params)
        self.predictions = predictions

    def set_predictions(self, predictions: pd.DataFrame):
        """Set prediction data from AI model."""
        self.predictions = predictions

    def generate_signals(self, data: pd.DataFrame, coin_id: str) -> List[TradeSignal]:
        signals = []

        if self.predictions is None or self.predictions.empty:
            logger.warning(f"No predictions available for {coin_id}")
            return signals

        prices = data.set_index("timestamp")["price"].sort_index()

        coin_preds = self.predictions[self.predictions["coin_id"] == coin_id]
        if coin_preds.empty:
            return signals

        coin_preds = coin_preds.set_index("timestamp").sort_index()

        for ts in prices.index:
            if ts not in coin_preds.index:
                continue

            pred = coin_preds.loc[ts]
            current_price = prices.loc[ts]

            predicted_change = (pred["forecast"] - current_price) / current_price

            if predicted_change > self.params.momentum_threshold:
                signals.append(
                    TradeSignal(
                        timestamp=ts,
                        coin_id=coin_id,
                        action="buy",
                        price=current_price,
                        reason=f"ai_prediction_up_{predicted_change:.2%}",
                    )
                )
            elif predicted_change < -self.params.momentum_threshold:
                signals.append(
                    TradeSignal(
                        timestamp=ts,
                        coin_id=coin_id,
                        action="sell",
                        price=current_price,
                        reason=f"ai_prediction_down_{predicted_change:.2%}",
                    )
                )

        return signals


def get_strategy(
    strategy_name: str,
    params: Optional[StrategyParams] = None,
    predictions: Optional[pd.DataFrame] = None,
) -> BaseStrategy:
    """Factory function to get trading strategy."""
    strategies = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "trend_following": TrendFollowingStrategy,
        "ai": lambda p=None, pred=None: AITradingStrategy(p, pred),
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}"
        )

    if strategy_name == "ai":
        return strategies[strategy_name](params, predictions)
    return strategies[strategy_name](params)
