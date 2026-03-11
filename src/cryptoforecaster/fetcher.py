"""
CryptoFetcher — pulls OHLCV + market data from the CoinGecko public API.

Endpoints used (all free / no key required):
  GET /coins/{id}/market_chart          → price, market_cap, volume timeseries
  GET /coins/{id}/ohlc                  → OHLCV candles
  GET /coins/markets                    → snapshot metadata (rank, supply, etc.)
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from cryptoforecaster.config import settings


class CryptoFetcher:
    """
    Fetch historical and current cryptocurrency data from CoinGecko.

    Example
    -------
    >>> fetcher = CryptoFetcher()
    >>> df = fetcher.fetch_market_chart("bitcoin", days=365)
    >>> print(df.head())
    """

    def __init__(
        self,
        base_url: str = settings.coingecko_base_url,
        vs_currency: str = settings.default_vs_currency,
        request_delay: float = settings.request_delay,
        timeout: int = settings.request_timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.vs_currency = vs_currency
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoForecast/0.1.0 (open-source research tool)",
        })

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: Optional[dict] = None, retries: int = 3) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {wait}s before retry…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(self.request_delay)
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
                if attempt == retries:
                    raise
                time.sleep(self.request_delay * attempt)
        return {}

    @staticmethod
    def _ts_to_dt(ts_ms: int) -> datetime:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_market_chart(
        self,
        coin_id: str,
        days: int = settings.default_days,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch daily price / market_cap / volume timeseries.

        Parameters
        ----------
        coin_id : CoinGecko coin id, e.g. "bitcoin"
        days    : number of historical days (max 365 for free tier without key)
        interval: "daily" (for >90 days, always daily granularity on free tier)

        Returns
        -------
        pd.DataFrame with columns: [coin_id, timestamp, price, market_cap, volume]
        """
        logger.info(f"Fetching market chart for {coin_id} ({days} days)")
        data = self._get(
            f"coins/{coin_id}/market_chart",
            params={
                "vs_currency": self.vs_currency,
                "days": days,
                "interval": interval,
            },
        )

        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            logger.warning(f"No price data returned for {coin_id}")
            return pd.DataFrame()

        df_price = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
        df_mcap  = pd.DataFrame(market_caps, columns=["timestamp_ms", "market_cap"])
        df_vol   = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])

        df = df_price.merge(df_mcap, on="timestamp_ms").merge(df_vol, on="timestamp_ms")
        df["timestamp"] = df["timestamp_ms"].apply(self._ts_to_dt)
        df["coin_id"]   = coin_id
        df["symbol"]    = settings.coin_symbols.get(coin_id, coin_id.upper())
        df["currency"]  = self.vs_currency.upper()

        df = df[["coin_id", "symbol", "currency", "timestamp", "price", "market_cap", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.success(f"  → {len(df)} rows fetched for {coin_id}")
        return df

    def fetch_ohlcv(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.

        CoinGecko returns candles at different granularities:
          1-2 days   → 30-min candles
          3-30 days  → 4-hour candles
          31-90 days → 4-hour candles
          91-365 days→ 4-day candles  (free tier)

        Returns
        -------
        pd.DataFrame: [coin_id, timestamp, open, high, low, close]
        """
        logger.info(f"Fetching OHLCV for {coin_id} ({days} days)")
        data = self._get(
            f"coins/{coin_id}/ohlc",
            params={"vs_currency": self.vs_currency, "days": days},
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close"])
        df["timestamp"] = df["timestamp_ms"].apply(self._ts_to_dt)
        df["coin_id"]   = coin_id
        df["symbol"]    = settings.coin_symbols.get(coin_id, coin_id.upper())
        df = df[["coin_id", "symbol", "timestamp", "open", "high", "low", "close"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.success(f"  → {len(df)} OHLCV rows for {coin_id}")
        return df

    def fetch_market_snapshot(self, coin_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch current market snapshot for a list of coins.

        Returns
        -------
        pd.DataFrame with rank, price, 24h change, market cap, etc.
        """
        coins = coin_ids or settings.default_coins
        logger.info(f"Fetching market snapshot for {len(coins)} coins")
        data = self._get(
            "coins/markets",
            params={
                "vs_currency": self.vs_currency,
                "ids": ",".join(coins),
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "1h,24h,7d,30d",
            },
        )
        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append({
                "coin_id":              item.get("id"),
                "symbol":               item.get("symbol", "").upper(),
                "name":                 item.get("name"),
                "current_price":        item.get("current_price"),
                "market_cap":           item.get("market_cap"),
                "market_cap_rank":      item.get("market_cap_rank"),
                "total_volume":         item.get("total_volume"),
                "high_24h":             item.get("high_24h"),
                "low_24h":              item.get("low_24h"),
                "price_change_pct_1h":  item.get("price_change_percentage_1h_in_currency"),
                "price_change_pct_24h": item.get("price_change_percentage_24h_in_currency"),
                "price_change_pct_7d":  item.get("price_change_percentage_7d_in_currency"),
                "price_change_pct_30d": item.get("price_change_percentage_30d_in_currency"),
                "circulating_supply":   item.get("circulating_supply"),
                "total_supply":         item.get("total_supply"),
                "ath":                  item.get("ath"),
                "ath_date":             item.get("ath_date"),
                "fetched_at":           datetime.now(tz=timezone.utc),
            })

        df = pd.DataFrame(records)
        logger.success(f"  → snapshot for {len(df)} coins")
        return df

    def fetch_all(
        self,
        coin_ids: Optional[List[str]] = None,
        days: int = settings.default_days,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience wrapper — fetches market_chart + ohlcv for every coin.

        Returns
        -------
        dict with keys:
            "market_charts" : concatenated DataFrame of all coins' daily series
            "ohlcv"         : concatenated OHLCV DataFrame
            "snapshot"      : current market snapshot
        """
        coins = coin_ids or settings.default_coins
        chart_frames, ohlcv_frames = [], []

        for coin in coins:
            try:
                chart_frames.append(self.fetch_market_chart(coin, days=days))
            except Exception as exc:
                logger.error(f"Failed market_chart for {coin}: {exc}")

            try:
                # OHLCV limited to 365 days max on free tier
                ohlcv_days = min(days, 365)
                ohlcv_frames.append(self.fetch_ohlcv(coin, days=ohlcv_days))
            except Exception as exc:
                logger.error(f"Failed OHLCV for {coin}: {exc}")

        snapshot = pd.DataFrame()
        try:
            snapshot = self.fetch_market_snapshot(coins)
        except Exception as exc:
            logger.error(f"Failed snapshot: {exc}")

        return {
            "market_charts": pd.concat(chart_frames, ignore_index=True) if chart_frames else pd.DataFrame(),
            "ohlcv":         pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame(),
            "snapshot":      snapshot,
        }
