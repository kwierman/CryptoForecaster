"""Tests for Prophet and ARIMA models using synthetic price data."""

import pytest
import numpy as np
import pandas as pd

from cryptoforecast.modeling.base import BaseModel


def make_price_series(n=200, seed=42):
    """Synthetic daily price series with trend + seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    trend = np.linspace(30000, 50000, n)
    noise = rng.normal(0, 500, n)
    seasonality = 2000 * np.sin(np.linspace(0, 4 * np.pi, n))
    prices = trend + seasonality + noise
    return pd.DataFrame({"timestamp": dates, "price": prices})


class TestBaseModel:
    def test_prepare_series(self):
        df = make_price_series(10)
        # Add a duplicate
        df = pd.concat([df, df.iloc[:2]])
        result = BaseModel.prepare_series(df)
        assert len(result) == 10  # duplicates removed
        assert result["timestamp"].is_monotonic_increasing


@pytest.mark.slow
class TestARIMAModel:
    """Integration tests (slower, require statsmodels)."""

    def test_fit_and_predict(self):
        from cryptoforecast.modeling.arima_model import ARIMAModel
        df = make_price_series(120)
        model = ARIMAModel(coin_id="bitcoin", order=(2, 1, 0), seasonal_order=(0, 0, 0, 0))
        model.fit(df)
        assert model._is_fitted
        assert "mae" in model.metrics

        fc = model.predict(horizon=7)
        assert isinstance(fc, pd.DataFrame)
        assert "forecast" in fc.columns
        future_rows = fc[fc["is_future"]]
        assert len(future_rows) == 7

    def test_unfitted_raises(self):
        from cryptoforecast.modeling.arima_model import ARIMAModel
        model = ARIMAModel(coin_id="bitcoin")
        with pytest.raises(RuntimeError):
            model.predict(horizon=5)

    def test_save_and_load(self, tmp_path):
        from cryptoforecast.modeling.arima_model import ARIMAModel
        df = make_price_series(120)
        model = ARIMAModel(coin_id="bitcoin", order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
        model.fit(df)
        path = model.save(str(tmp_path / "arima.joblib"))

        loaded = ARIMAModel(coin_id="bitcoin")
        loaded.load(path)
        assert loaded._is_fitted


@pytest.mark.slow
class TestProphetModel:
    """Integration tests (slower, require prophet)."""

    def test_fit_and_predict(self):
        pytest.importorskip("prophet")
        from cryptoforecast.modeling.prophet_model import ProphetModel
        df = make_price_series(200)
        model = ProphetModel(coin_id="bitcoin")
        model.fit(df)
        assert model._is_fitted

        fc = model.predict(horizon=14)
        future = fc[fc["is_future"]]
        assert len(future) == 14
        assert (fc["forecast"] >= 0).all()
        assert (fc["upper_bound"] >= fc["forecast"]).all()
