"""
Abstract base class for all time-series forecast models.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


class BaseModel(abc.ABC):
    """
    All forecast models must implement fit(), predict(), and save()/load().
    """

    name: str = "base"

    def __init__(self, coin_id: str, **hyperparams):
        self.coin_id = coin_id
        self.hyperparams = hyperparams
        self._is_fitted = False
        self.train_start: Optional[datetime] = None
        self.train_end: Optional[datetime] = None
        self.metrics: Dict[str, float] = {}
        self.version: str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.model_path: Optional[str] = None

    # ── Abstract interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseModel":
        """
        Train on a DataFrame with at minimum columns [timestamp, price].
        Must set self._is_fitted = True and populate self.metrics.
        """

    @abc.abstractmethod
    def predict(
        self,
        horizon: int,
        include_history: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            [timestamp, forecast, lower_bound, upper_bound, is_future]
        horizon = number of future steps to forecast.
        """

    @abc.abstractmethod
    def save(self, path: str) -> str:
        """Persist model to disk, return the path used."""

    @abc.abstractmethod
    def load(self, path: str) -> "BaseModel":
        """Load model from disk."""

    # ── Shared utilities ──────────────────────────────────────────────────

    @staticmethod
    def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise input DataFrame:
          - ensure 'timestamp' is datetime
          - sort ascending
          - drop rows where price is NaN
          - remove duplicates
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.dropna(subset=["price"])
        df = df.reset_index(drop=True)
        return df

    def check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. Call .fit() first."
            )

    def __repr__(self):
        status = "fitted" if self._is_fitted else "unfitted"
        return f"<{self.__class__.__name__}(coin={self.coin_id}, {status})>"
