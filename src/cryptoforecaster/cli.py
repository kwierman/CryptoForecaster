"""
CryptoForecast CLI
==================
Usage:
  cryptoforecast ingest  [--coins BTC,ETH,...] [--days 365]
  cryptoforecast train   [--coins BTC,ETH,...] [--model prophet|arima|ensemble]
  cryptoforecast predict [--coins BTC,ETH,...] [--horizon 30]
  cryptoforecast pipeline  (runs ingest → train → predict in one shot)
  cryptoforecast summary
"""

from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="🚀 CryptoForecast — CLI for crypto price forecasting")
console = Console()

_COIN_CHOICES = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
    "cardano", "dogecoin", "avalanche-2", "polkadot", "chainlink",
]


def _parse_coins(coins_str: Optional[str]) -> Optional[List[str]]:
    if coins_str is None:
        return None
    return [c.strip().lower() for c in coins_str.split(",")]


@app.command()
def ingest(
    coins: Optional[str] = typer.Option(None, "--coins", help="Comma-separated coin IDs"),
    days:  int           = typer.Option(365,   "--days",  help="Historical days to fetch"),
):
    """Pull data from CoinGecko and store in DuckDB."""
    from cryptoforecast.ingestion.fetcher import CryptoFetcher
    from cryptoforecast.storage.database  import CryptoDatabase
    from cryptoforecast.utils.logger      import setup_logger
    setup_logger()

    coin_list = _parse_coins(coins)
    rprint(f"[bold cyan]🌐 Ingesting data for: {coin_list or 'all default coins'}[/bold cyan]")

    fetcher = CryptoFetcher()
    db      = CryptoDatabase()
    data    = fetcher.fetch_all(coin_ids=coin_list, days=days)

    db.upsert_market_prices(data["market_charts"])
    db.upsert_ohlcv(data["ohlcv"])
    db.upsert_market_snapshot(data["snapshot"])

    rprint("[bold green]✅ Ingestion complete![/bold green]")
    _print_summary(db)


@app.command()
def train(
    coins: Optional[str] = typer.Option(None, "--coins", help="Comma-separated coin IDs"),
    model: str           = typer.Option("prophet", "--model", help="prophet | arima | ensemble"),
):
    """Train a forecast model for each coin."""
    from cryptoforecast.modeling.trainer import ForecastTrainer
    from cryptoforecast.storage.database import CryptoDatabase
    from cryptoforecast.utils.logger     import setup_logger
    setup_logger()

    coin_list = _parse_coins(coins)
    rprint(f"[bold cyan]🧠 Training [{model}] for: {coin_list or 'all coins in DB'}[/bold cyan]")

    db      = CryptoDatabase()
    trainer = ForecastTrainer(db=db, model_name=model)
    results = trainer.train_all(coin_ids=coin_list)

    table = Table(title="Training Results", style="bold")
    table.add_column("Coin",       style="cyan")
    table.add_column("Model",      style="magenta")
    table.add_column("Test MAE",   style="yellow")
    table.add_column("Test MAPE",  style="red")

    for coin_id, m in results.items():
        mae  = m.metrics.get("test_mae", float("nan"))
        mape = m.metrics.get("test_mape", float("nan"))
        table.add_row(coin_id, m.name, f"{mae:.2f}", f"{mape:.2%}")

    console.print(table)
    rprint("[bold green]✅ Training complete![/bold green]")


@app.command()
def predict(
    coins:   Optional[str] = typer.Option(None, "--coins",   help="Comma-separated coin IDs"),
    model:   str           = typer.Option("prophet", "--model",   help="prophet | arima | ensemble"),
    horizon: int           = typer.Option(30,        "--horizon", help="Forecast days ahead"),
):
    """Run inference and store forecasts in DuckDB."""
    from cryptoforecast.forecasting.predictor import ForecastPredictor
    from cryptoforecast.storage.database      import CryptoDatabase
    from cryptoforecast.utils.logger          import setup_logger
    setup_logger()

    coin_list = _parse_coins(coins)
    rprint(f"[bold cyan]🔮 Forecasting [{model}] +{horizon}d for: {coin_list or 'all coins'}[/bold cyan]")

    db        = CryptoDatabase()
    predictor = ForecastPredictor(db=db, model_name=model, horizon=horizon)
    results   = predictor.forecast_all(coin_ids=coin_list)

    for coin_id, df in results.items():
        future_rows = df[df["is_future"]]
        if not future_rows.empty:
            last = future_rows.iloc[-1]
            rprint(
                f"  {coin_id}: forecast {horizon}d → "
                f"[yellow]${last['forecast']:,.2f}[/yellow] "
                f"(CI: ${last['lower_bound']:,.2f} – ${last['upper_bound']:,.2f})"
            )

    rprint("[bold green]✅ Forecasts saved![/bold green]")


@app.command()
def pipeline(
    coins:   Optional[str] = typer.Option(None,      "--coins",   help="Comma-separated coin IDs"),
    days:    int           = typer.Option(365,        "--days",    help="Historical days"),
    model:   str           = typer.Option("prophet",  "--model",   help="prophet | arima | ensemble"),
    horizon: int           = typer.Option(30,         "--horizon", help="Forecast days"),
):
    """Run the full pipeline: ingest → train → predict."""
    from cryptoforecast.utils.logger import setup_logger
    setup_logger()

    rprint("[bold magenta]🚀 Running full pipeline…[/bold magenta]")
    ctx = typer.Context(app)

    # Ingest
    typer.echo("\n─── Step 1: Ingestion ───")
    ingest(coins=coins, days=days)

    # Train
    typer.echo("\n─── Step 2: Training ───")
    train(coins=coins, model=model)

    # Predict
    typer.echo("\n─── Step 3: Forecasting ───")
    predict(coins=coins, model=model, horizon=horizon)

    rprint("\n[bold green]🎉 Pipeline complete![/bold green]")


@app.command()
def summary():
    """Show a summary of data stored in DuckDB."""
    from cryptoforecast.storage.database import CryptoDatabase
    db = CryptoDatabase()
    _print_summary(db)


def _print_summary(db):
    df = db.summary()
    if df.empty:
        rprint("[yellow]No data in database yet.[/yellow]")
        return

    table = Table(title="Database Summary", style="bold")
    for col in df.columns:
        table.add_column(col, style="cyan")
    for _, row in df.iterrows():
        table.add_row(*[str(round(v, 2)) if isinstance(v, float) else str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    app()
