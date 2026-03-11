"""Tests for trading strategies."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from cryptoforecaster.strategy import (
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    get_strategy,
)
from cryptoforecaster.schemas import StrategyParams, TradeSignal


def make_price_data(coin_id="bitcoin", n=100, start_price=40000):
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc), periods=n, freq="D"
    )
    prices = [start_price]
    for i in range(1, n):
        change = prices[-1] * (0.01 * (1 if i % 3 == 0 else -1))
        prices.append(prices[-1] + change)

    return pd.DataFrame(
        {
            "coin_id": coin_id,
            "symbol": "BTC",
            "timestamp": dates,
            "price": prices,
        }
    )


class TestMomentumStrategy:
    def test_generate_signals(self):
        params = StrategyParams(momentum_threshold=0.02)
        strategy = MomentumStrategy(params)

        data = make_price_data()
        signals = strategy.generate_signals(data, "bitcoin")

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradeSignal)
            assert signal.coin_id == "bitcoin"
            assert signal.action in ["buy", "sell"]

    def test_no_signals_on_insufficient_data(self):
        strategy = MomentumStrategy()
        data = make_price_data(n=1)
        signals = strategy.generate_signals(data, "bitcoin")
        assert signals == []


class TestMeanReversionStrategy:
    def test_generate_signals(self):
        params = StrategyParams(rsi_oversold=30, rsi_overbought=70)
        strategy = MeanReversionStrategy(params)

        data = make_price_data()
        signals = strategy.generate_signals(data, "bitcoin")

        assert isinstance(signals, list)

    def test_insufficient_data(self):
        strategy = MeanReversionStrategy()
        data = make_price_data(n=10)
        signals = strategy.generate_signals(data, "bitcoin")
        assert signals == []


class TestTrendFollowingStrategy:
    def test_generate_signals(self):
        params = StrategyParams(short_window=10, long_window=20)
        strategy = TrendFollowingStrategy(params)

        data = make_price_data(n=50)
        signals = strategy.generate_signals(data, "bitcoin")

        assert isinstance(signals, list)

    def test_insufficient_data(self):
        strategy = TrendFollowingStrategy()
        data = make_price_data(n=15)
        signals = strategy.generate_signals(data, "bitcoin")
        assert signals == []


class TestStrategyFactory:
    def test_get_momentum_strategy(self):
        strategy = get_strategy("momentum")
        assert isinstance(strategy, MomentumStrategy)

    def test_get_mean_reversion_strategy(self):
        strategy = get_strategy("mean_reversion")
        assert isinstance(strategy, MeanReversionStrategy)

    def test_get_trend_following_strategy(self):
        strategy = get_strategy("trend_following")
        assert isinstance(strategy, TrendFollowingStrategy)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            get_strategy("invalid_strategy")
