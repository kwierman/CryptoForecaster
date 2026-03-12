"""Tests for CryptoDatabase storage layer (uses in-memory DuckDB)."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from cryptoforecaster.storage.database import CryptoDatabase


@pytest.fixture
def db(tmp_path):
    """In-memory DuckDB database for testing."""
    db_path = str(tmp_path / "test.duckdb")
    return CryptoDatabase(db_path=db_path)


def make_price_df(coin_id="bitcoin", n=10):
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "coin_id": coin_id,
            "symbol": "BTC",
            "currency": "USD",
            "timestamp": dates,
            "price": [40000 + i * 100 for i in range(n)],
            "market_cap": [8e11 + i * 1e9 for i in range(n)],
            "volume": [2e10] * n,
        }
    )


class TestCryptoDatabase:
    def test_upsert_market_prices(self, db):
        df = make_price_df(n=5)
        rows = db.upsert_market_prices(df)
        assert rows == 5

    def test_get_price_series(self, db):
        df = make_price_df(n=10)
        db.upsert_market_prices(df)
        result = db.get_price_series("bitcoin")
        assert len(result) == 10
        assert "price" in result.columns

    def test_upsert_idempotent(self, db):
        df = make_price_df(n=5)
        db.upsert_market_prices(df)
        db.upsert_market_prices(df)  # should not duplicate
        result = db.get_price_series("bitcoin")
        assert len(result) == 5

    def test_get_all_coins(self, db):
        db.upsert_market_prices(make_price_df("bitcoin", 3))
        db.upsert_market_prices(make_price_df("ethereum", 3))
        coins = db.get_all_coins()
        assert "bitcoin" in coins
        assert "ethereum" in coins

    def test_summary(self, db):
        db.upsert_market_prices(make_price_df("bitcoin", 5))
        s = db.summary()
        assert not s.empty
        assert "coin_id" in s.columns

    def test_upsert_forecasts(self, db):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        fc_df = pd.DataFrame(
            {
                "coin_id": "bitcoin",
                "symbol": "BTC",
                "model_name": "prophet",
                "model_version": "v1",
                "timestamp": dates,
                "forecast": [40000.0, 41000, 42000, 43000, 44000],
                "lower_bound": [39000.0] * 5,
                "upper_bound": [45000.0] * 5,
                "is_future": [False, False, True, True, True],
            }
        )
        rows = db.upsert_forecasts(fc_df)
        assert rows == 5

        stored = db.get_forecasts("bitcoin", "prophet")
        assert len(stored) == 5

    def test_register_model(self, db):
        db.register_model(
            coin_id="bitcoin",
            model_name="prophet",
            model_version="20240101_120000",
            model_path="/tmp/model.joblib",
            train_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metrics={"mae": 500.0, "mape": 0.02},
            hyperparams={"changepoint_prior_scale": 0.05},
        )
        record = db.get_latest_model("bitcoin", "prophet")
        assert record is not None
        assert record["coin_id"] == "bitcoin"

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "ctx.duckdb")
        with CryptoDatabase(db_path=path) as db:
            df = make_price_df(n=3)
            db.upsert_market_prices(df)
            result = db.get_price_series("bitcoin")
        assert len(result) == 3
