"""Tests for backtesting engine."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from cryptoforecaster.backtest import Portfolio, BacktestEngine
from cryptoforecaster.schemas import StrategyParams


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


class TestPortfolio:
    def test_initialization(self):
        portfolio = Portfolio(10000)
        assert portfolio.cash == 10000
        assert portfolio.initial_capital == 10000
        assert portfolio.holdings == {}

    def test_buy(self):
        portfolio = Portfolio(10000)
        result = portfolio.buy("bitcoin", 0.1, 40000, datetime.now(timezone.utc))

        assert result is True
        assert portfolio.cash == 6000
        assert portfolio.holdings["bitcoin"] == 0.1

    def test_buy_insufficient_funds(self):
        portfolio = Portfolio(1000)
        result = portfolio.buy("bitcoin", 1, 40000, datetime.now(timezone.utc))

        assert result is False

    def test_sell(self):
        portfolio = Portfolio(10000)
        portfolio.buy("bitcoin", 0.1, 40000, datetime.now(timezone.utc))

        result = portfolio.sell("bitcoin", 0.1, 45000, datetime.now(timezone.utc))

        assert result is True
        assert "bitcoin" not in portfolio.holdings

    def test_sell_insufficient_holdings(self):
        portfolio = Portfolio(10000)
        result = portfolio.sell("bitcoin", 1, 45000, datetime.now(timezone.utc))

        assert result is False

    def test_get_value(self):
        portfolio = Portfolio(10000)
        portfolio.buy("bitcoin", 0.1, 40000, datetime.now(timezone.utc))

        prices = {"bitcoin": 50000}
        value = portfolio.get_value(prices)

        # Cash after buy: 10000 - (0.1 * 40000) = 6000
        # Holdings: 0.1 * 50000 = 5000
        # Total: 6000 + 5000 = 11000
        assert value == 11000

    def test_get_holdings_dict(self):
        portfolio = Portfolio(10000)
        portfolio.buy("bitcoin", 0.1, 40000, datetime.now(timezone.utc))

        holdings = portfolio.get_holdings_dict()

        assert holdings == {"bitcoin": 0.1}


class TestBacktestEngine:
    def test_initialization(self):
        engine = BacktestEngine(initial_capital=5000, strategy_name="momentum")

        assert engine.initial_capital == 5000
        assert engine.strategy_name == "momentum"

    def test_run_backtest(self):
        engine = BacktestEngine(initial_capital=10000, strategy_name="momentum")

        data = {"bitcoin": make_price_data()}

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 31, tzinfo=timezone.utc)

        result = engine.run(data, start, end)

        assert result.total_return is not None
        assert result.portfolio_value >= 0
        assert result.total_trades >= 0

    def test_run_backtest_multiple_coins(self):
        engine = BacktestEngine(initial_capital=10000, strategy_name="momentum")

        data = {
            "bitcoin": make_price_data("bitcoin"),
            "ethereum": make_price_data("ethereum", start_price=2000),
        }

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 31, tzinfo=timezone.utc)

        result = engine.run(data, start, end)

        assert result.total_return is not None

    def test_calculate_sharpe_ratio(self):
        engine = BacktestEngine()

        values = [10000, 10500, 10200, 10800, 11000]
        sharpe = engine._calculate_sharpe_ratio(values)

        assert isinstance(sharpe, float)

    def test_calculate_max_drawdown(self):
        engine = BacktestEngine()

        values = [10000, 11000, 10500, 12000, 11500]
        dd = engine._calculate_max_drawdown(values)

        assert isinstance(dd, float)
        assert 0 <= dd <= 1
