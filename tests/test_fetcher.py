"""Tests for CryptoFetcher using mock HTTP responses."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from cryptoforecast.ingestion.fetcher import CryptoFetcher


MOCK_CHART = {
    "prices":       [[1700000000000, 35000.0], [1700086400000, 36000.0]],
    "market_caps":  [[1700000000000, 6.8e11],  [1700086400000, 7.0e11]],
    "total_volumes":[[1700000000000, 2.0e10],  [1700086400000, 2.1e10]],
}

MOCK_OHLCV = [
    [1700000000000, 34000, 36000, 33000, 35500],
    [1700086400000, 35500, 37000, 35000, 36500],
]

MOCK_MARKETS = [
    {
        "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
        "current_price": 37000, "market_cap": 7.2e11, "market_cap_rank": 1,
        "total_volume": 2.2e10, "high_24h": 37500, "low_24h": 36000,
        "price_change_percentage_1h_in_currency": 0.1,
        "price_change_percentage_24h_in_currency": 1.2,
        "price_change_percentage_7d_in_currency": -2.3,
        "price_change_percentage_30d_in_currency": 5.0,
        "circulating_supply": 19e6, "total_supply": 21e6,
        "ath": 69000, "ath_date": "2021-11-10T14:24:11.849Z",
    }
]


@pytest.fixture
def fetcher():
    return CryptoFetcher(request_delay=0)


def mock_get(endpoint, params=None, retries=3):
    if "market_chart" in endpoint:
        return MOCK_CHART
    if "ohlc" in endpoint:
        return MOCK_OHLCV
    if "markets" in endpoint:
        return MOCK_MARKETS
    return {}


class TestCryptoFetcher:

    def test_fetch_market_chart(self, fetcher):
        with patch.object(fetcher, "_get", side_effect=mock_get):
            df = fetcher.fetch_market_chart("bitcoin", days=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(["coin_id", "timestamp", "price", "market_cap", "volume"]).issubset(df.columns)
        assert df["coin_id"].iloc[0] == "bitcoin"
        assert df["price"].iloc[0] == pytest.approx(35000.0)

    def test_fetch_ohlcv(self, fetcher):
        with patch.object(fetcher, "_get", side_effect=mock_get):
            df = fetcher.fetch_ohlcv("bitcoin", days=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "open" in df.columns and "close" in df.columns

    def test_fetch_market_snapshot(self, fetcher):
        with patch.object(fetcher, "_get", side_effect=mock_get):
            df = fetcher.fetch_market_snapshot(["bitcoin"])
        assert isinstance(df, pd.DataFrame)
        assert "current_price" in df.columns
        assert df["coin_id"].iloc[0] == "bitcoin"

    def test_fetch_all_returns_dict(self, fetcher):
        with patch.object(fetcher, "_get", side_effect=mock_get):
            result = fetcher.fetch_all(coin_ids=["bitcoin"])
        assert "market_charts" in result
        assert "ohlcv" in result
        assert "snapshot" in result
        assert isinstance(result["market_charts"], pd.DataFrame)

    def test_empty_on_missing_data(self, fetcher):
        with patch.object(fetcher, "_get", return_value={}):
            df = fetcher.fetch_market_chart("bitcoin")
        assert df.empty
