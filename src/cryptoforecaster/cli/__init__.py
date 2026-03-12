import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import logging
from datetime import datetime, timezone
from typing import Optional, List

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=Console(log_path=False))],
)

console = Console(log_path=False)

app = typer.Typer(help="CryptoForecast - Cryptocurrency price forecasting CLI")


@app.command()
def ingest(
    coins: Optional[List[str]] = typer.Option(
        None, "--coins", "-c", help="Comma-separated list of coin IDs"
    ),
    days: int = typer.Option(
        365, "--days", "-d", help="Number of days of historical data"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to DuckDB database"
    ),
):
    """Ingest historical cryptocurrency data from CoinGecko."""
    from cryptoforecaster.ingestion.fetcher import CryptoFetcher
    from cryptoforecaster.storage.database import CryptoDatabase
    from cryptoforecaster.config import settings

    if coins:
        coin_list = (
            [c.strip() for c in coins[0].split(",")]
            if isinstance(coins, list)
            else coins.split(",")
        )
    else:
        coin_list = settings.default_coins

    fetcher = CryptoFetcher()
    db = CryptoDatabase(db_path=db_path) if db_path else CryptoDatabase()

    console.print(f"[bold]Fetching data for: {', '.join(coin_list)}[/bold]")

    try:
        result = fetcher.fetch_all(coin_list, days=days)

        if not result["market_charts"].empty:
            db.upsert_market_prices(result["market_charts"])

        if not result["ohlcv"].empty:
            db.upsert_ohlcv(result["ohlcv"])

        if not result["snapshot"].empty:
            db.upsert_market_snapshot(result["snapshot"])

        console.print("[green]Data ingestion complete![/green]")

    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    coin: str = typer.Option(..., "--coin", "-c", help="Coin ID to train on"),
    model_type: str = typer.Option(
        "prophet", "--model", "-m", help="Model type: prophet, arima, ensemble"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to DuckDB database"
    ),
):
    """Train a forecasting model on historical data."""
    from cryptoforecaster.modeling.trainer import ForecastTrainer
    from cryptoforecaster.storage.database import CryptoDatabase

    db = CryptoDatabase(db_path=db_path) if db_path else CryptoDatabase()
    trainer = ForecastTrainer(db=db, model_name=model_type)

    console.print(f"[bold]Training {model_type} model for {coin}[/bold]")

    try:
        model = trainer.train(coin)
        console.print("[green]Model trained successfully![/green]")
        console.print(f"Model saved to: {model.model_path}")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    coins: Optional[List[str]] = typer.Option(
        None, "--coins", "-c", help="Comma-separated list of coin IDs"
    ),
    horizon: int = typer.Option(
        30, "--horizon", "-h", help="Number of days to forecast"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Specific model to use"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to DuckDB database"
    ),
):
    """Generate price predictions using trained models."""
    from cryptoforecaster.forecasting.predictor import ForecastPredictor
    from cryptoforecaster.storage.database import CryptoDatabase
    from cryptoforecaster.config import settings

    if coins:
        coin_list = (
            [c.strip() for c in coins[0].split(",")]
            if isinstance(coins, list)
            else coins.split(",")
        )
    else:
        coin_list = settings.default_coins

    db = CryptoDatabase(db_path=db_path) if db_path else CryptoDatabase()
    predictor = ForecastPredictor(
        db=db,
        model_name=model_name or settings.default_model,
        horizon=horizon,
    )

    console.print(f"[bold]Generating predictions for: {', '.join(coin_list)}[/bold]")

    try:
        predictions = predictor.forecast(coin_list[0])

        table = Table(title="Price Predictions")
        table.add_column("Coin")
        table.add_column("Date")
        table.add_column("Forecast")
        table.add_column("Lower Bound")
        table.add_column("Upper Bound")

        for _, row in predictions.head(20).iterrows():
            table.add_row(
                row["coin_id"],
                row["timestamp"].strftime("%Y-%m-%d"),
                f"${row['forecast']:.2f}",
                f"${row.get('lower_bound', 0):.2f}",
                f"${row.get('upper_bound', 0):.2f}",
            )

        console.print(table)
        console.print(f"[green]Generated {len(predictions)} predictions[/green]")

    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def backtest(
    coins: List[str] = typer.Option(
        ..., "--coins", "-c", help="Comma-separated list of coin IDs"
    ),
    start_date: str = typer.Option(
        ..., "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(
        10000.0, "--capital", help="Initial capital in USD"
    ),
    strategy: str = typer.Option(
        "momentum",
        "--strategy",
        help="Strategy: momentum, mean_reversion, trend_following",
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to DuckDB database"
    ),
):
    """Backtest a trading strategy on historical data."""
    from cryptoforecaster.storage.database import CryptoDatabase
    from cryptoforecaster.backtest import BacktestEngine

    coin_list = (
        [c.strip() for c in coins[0].split(",")]
        if isinstance(coins, list)
        else coins.split(",")
    )

    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    db = CryptoDatabase(db_path=db_path) if db_path else CryptoDatabase()

    console.print(f"[bold]Running backtest from {start_date} to {end_date}[/bold]")
    console.print(f"Coins: {', '.join(coin_list)}")
    console.print(f"Strategy: {strategy}")
    console.print(f"Initial Capital: ${initial_capital:,.2f}")

    try:
        data = {}
        for coin in coin_list:
            df = db.get_price_series(coin, start=start_date, end=end_date)
            if not df.empty:
                data[coin] = df
            else:
                console.print(f"[yellow]No data found for {coin}[/yellow]")

        if not data:
            console.print("[red]No data available for backtesting[/red]")
            raise typer.Exit(1)

        engine = BacktestEngine(
            initial_capital=initial_capital,
            strategy_name=strategy,
        )

        result = engine.run(data, start, end)

        console.print("\n[bold green]Backtest Results[/bold green]")
        console.print(f"Total Return: {result.total_return:.2%}")
        console.print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        console.print(f"Max Drawdown: {result.max_drawdown:.2%}")
        console.print(f"Win Rate: {result.win_rate:.2%}")
        console.print(f"Total Trades: {result.total_trades}")
        console.print(f"Final Portfolio Value: ${result.portfolio_value:,.2f}")

    except Exception as e:
        console.print(f"[red]Error during backtest: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def summary(
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Path to DuckDB database"
    ),
):
    """Show summary of stored data."""
    from cryptoforecaster.storage.database import CryptoDatabase
    from rich.table import Table

    db = CryptoDatabase(db_path=db_path) if db_path else CryptoDatabase()

    try:
        summary_df = db.summary()

        if summary_df.empty:
            console.print("[yellow]No data in database[/yellow]")
            return

        table = Table(title="Database Summary")
        table.add_column("Coin")
        table.add_column("Symbol")
        table.add_column("Price Rows")
        table.add_column("Earliest")
        table.add_column("Latest")
        table.add_column("Min Price")
        table.add_column("Max Price")

        for _, row in summary_df.iterrows():
            table.add_row(
                row["coin_id"],
                row["symbol"],
                str(row["price_rows"]),
                row["earliest"].strftime("%Y-%m-%d") if row["earliest"] else "N/A",
                row["latest"].strftime("%Y-%m-%d") if row["latest"] else "N/A",
                f"${row['min_price']:.2f}" if row["min_price"] else "N/A",
                f"${row['max_price']:.2f}" if row["max_price"] else "N/A",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
