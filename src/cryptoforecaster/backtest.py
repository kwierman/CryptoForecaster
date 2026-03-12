from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from cryptoforecaster.schemas import BacktestResult, TradeSignal, StrategyParams
from cryptoforecaster.strategy import get_strategy


class Portfolio:
    """Portfolio manager for backtesting."""

    def __init__(self, initial_capital: float, position_size: float = 0.1):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.holdings: Dict[str, float] = {}
        self.position_size = position_size
        self.history: List[Dict] = []

    def buy(
        self, coin_id: str, quantity: float, price: float, timestamp: datetime
    ) -> bool:
        cost = quantity * price
        if cost > self.cash:
            return False

        self.cash -= cost
        self.holdings[coin_id] = self.holdings.get(coin_id, 0) + quantity
        self.history.append(
            {
                "timestamp": timestamp,
                "action": "buy",
                "coin_id": coin_id,
                "quantity": quantity,
                "price": price,
                "value": cost,
            }
        )
        return True

    def sell(
        self, coin_id: str, quantity: float, price: float, timestamp: datetime
    ) -> bool:
        if self.holdings.get(coin_id, 0) < quantity:
            return False

        proceeds = quantity * price
        self.cash += proceeds
        self.holdings[coin_id] -= quantity
        if self.holdings[coin_id] <= 0:
            del self.holdings[coin_id]

        self.history.append(
            {
                "timestamp": timestamp,
                "action": "sell",
                "coin_id": coin_id,
                "quantity": quantity,
                "price": price,
                "value": proceeds,
            }
        )
        return True

    def get_value(self, prices: Dict[str, float]) -> float:
        holdings_value = sum(
            self.holdings.get(coin, 0) * prices.get(coin, 0) for coin in self.holdings
        )
        return self.cash + holdings_value

    def get_holdings_dict(self) -> Dict[str, float]:
        return self.holdings.copy()


class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        strategy_name: str = "momentum",
        strategy_params: Optional[StrategyParams] = None,
    ):
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or StrategyParams()
        self.portfolio: Optional[Portfolio] = None

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        self.portfolio = Portfolio(
            self.initial_capital, self.strategy_params.position_size
        )

        strategy = get_strategy(self.strategy_name, self.strategy_params)

        all_signals: List[TradeSignal] = []
        for coin_id, coin_data in data.items():
            filtered = coin_data[
                (coin_data["timestamp"] >= start_date)
                & (coin_data["timestamp"] <= end_date)
            ]
            signals = strategy.generate_signals(filtered, coin_id)
            all_signals.extend(signals)

        all_signals.sort(key=lambda x: x.timestamp)

        prices_at_time = {}
        portfolio_values = []

        for signal in all_signals:
            coin_id = signal.coin_id

            df = data[coin_id]
            price_row = df[df["timestamp"] == signal.timestamp]
            if price_row.empty:
                continue

            current_price = price_row["price"].iloc[0]
            prices_at_time[coin_id] = current_price

            if signal.action == "buy":
                max_quantity = (
                    self.portfolio.cash * self.portfolio.position_size
                ) / current_price
                self.portfolio.buy(
                    coin_id, max_quantity, current_price, signal.timestamp
                )

            elif signal.action == "sell":
                quantity = self.portfolio.holdings.get(coin_id, 0)
                if quantity > 0:
                    self.portfolio.sell(
                        coin_id, quantity, current_price, signal.timestamp
                    )

            current_value = self.portfolio.get_value(prices_at_time)
            portfolio_values.append(current_value)

        for coin_id in data.keys():
            if coin_id not in prices_at_time:
                df = data[coin_id]
                latest = (
                    df[df["timestamp"] <= end_date].iloc[-1] if not df.empty else None
                )
                if latest is not None:
                    prices_at_time[coin_id] = latest["price"]

        final_value = self.portfolio.get_value(prices_at_time)
        total_return = (final_value - self.initial_capital) / self.initial_capital

        trades = self.portfolio.history
        winning_trades = [
            t for t in trades if t["action"] == "sell" and t.get("value", 0) > 0
        ]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        logger.info(
            f"Backtest complete. Final value: ${final_value:.2f}, Return: {total_return:.2%}"
        )

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            portfolio_value=final_value,
            holdings=self.portfolio.get_holdings_dict(),
            trades=trades,
        )

    def _calculate_sharpe_ratio(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0

        returns = np.diff(values) / values[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0

        values_arr = np.array(values)
        running_max = np.maximum.accumulate(values_arr)
        drawdown = (values_arr - running_max) / running_max
        return abs(np.min(drawdown))


class AIBacktestEngine(BacktestEngine):
    """Backtesting engine that uses AI predictions."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        strategy_params: Optional[StrategyParams] = None,
        predictions: Optional[pd.DataFrame] = None,
    ):
        super().__init__(initial_capital, "ai", strategy_params)
        self.predictions = predictions

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        from cryptoforecaster.strategy import AITradingStrategy

        logger.info(f"Starting AI backtest from {start_date} to {end_date}")

        self.portfolio = Portfolio(
            self.initial_capital, self.strategy_params.position_size
        )

        strategy = AITradingStrategy(self.strategy_params, self.predictions)

        all_signals: List[TradeSignal] = []
        for coin_id, coin_data in data.items():
            filtered = coin_data[
                (coin_data["timestamp"] >= start_date)
                & (coin_data["timestamp"] <= end_date)
            ]
            signals = strategy.generate_signals(filtered, coin_id)
            all_signals.extend(signals)

        all_signals.sort(key=lambda x: x.timestamp)

        prices_at_time = {}

        for signal in all_signals:
            coin_id = signal.coin_id

            df = data[coin_id]
            price_row = df[df["timestamp"] == signal.timestamp]
            if price_row.empty:
                continue

            current_price = price_row["price"].iloc[0]
            prices_at_time[coin_id] = current_price

            if signal.action == "buy":
                max_quantity = (
                    self.portfolio.cash * self.portfolio.position_size
                ) / current_price
                self.portfolio.buy(
                    coin_id, max_quantity, current_price, signal.timestamp
                )

            elif signal.action == "sell":
                quantity = self.portfolio.holdings.get(coin_id, 0)
                if quantity > 0:
                    self.portfolio.sell(
                        coin_id, quantity, current_price, signal.timestamp
                    )

        for coin_id in data.keys():
            if coin_id not in prices_at_time:
                df = data[coin_id]
                latest = (
                    df[df["timestamp"] <= end_date].iloc[-1] if not df.empty else None
                )
                if latest is not None:
                    prices_at_time[coin_id] = latest["price"]

        final_value = self.portfolio.get_value(prices_at_time)
        total_return = (final_value - self.initial_capital) / self.initial_capital

        trades = self.portfolio.history
        winning_trades = [t for t in trades if t["action"] == "sell"]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        portfolio_values = [self.initial_capital] + [
            self.portfolio.get_value(prices_at_time)
        ]
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            portfolio_value=final_value,
            holdings=self.portfolio.get_holdings_dict(),
            trades=trades,
        )
