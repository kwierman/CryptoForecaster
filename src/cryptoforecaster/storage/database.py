"""
CryptoDatabase — DuckDB-backed storage for all crypto market data and forecasts.

Schema
------
  market_prices   : daily price / market_cap / volume timeseries
  ohlcv           : OHLCV candle data
  market_snapshot : periodic market snapshots
  forecasts       : model forecast outputs
  model_registry  : trained model metadata
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd
from loguru import logger

from cryptoforecast.config import settings


_SCHEMA_SQL = """
-- ── Raw price timeseries ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_prices (
    id           INTEGER PRIMARY KEY,
    coin_id      VARCHAR NOT NULL,
    symbol       VARCHAR NOT NULL,
    currency     VARCHAR NOT NULL DEFAULT 'USD',
    timestamp    TIMESTAMPTZ NOT NULL,
    price        DOUBLE,
    market_cap   DOUBLE,
    volume       DOUBLE,
    ingested_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (coin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_mp_coin_ts ON market_prices (coin_id, timestamp);

-- ── OHLCV candles ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    id          INTEGER PRIMARY KEY,
    coin_id     VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    ingested_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (coin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_coin_ts ON ohlcv (coin_id, timestamp);

-- ── Market snapshots ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_snapshot (
    id                   INTEGER PRIMARY KEY,
    coin_id              VARCHAR,
    symbol               VARCHAR,
    name                 VARCHAR,
    current_price        DOUBLE,
    market_cap           DOUBLE,
    market_cap_rank      INTEGER,
    total_volume         DOUBLE,
    high_24h             DOUBLE,
    low_24h              DOUBLE,
    price_change_pct_1h  DOUBLE,
    price_change_pct_24h DOUBLE,
    price_change_pct_7d  DOUBLE,
    price_change_pct_30d DOUBLE,
    circulating_supply   DOUBLE,
    total_supply         DOUBLE,
    ath                  DOUBLE,
    ath_date             VARCHAR,
    fetched_at           TIMESTAMPTZ
);

-- ── Forecasts ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY,
    coin_id         VARCHAR NOT NULL,
    symbol          VARCHAR,
    model_name      VARCHAR NOT NULL,
    model_version   VARCHAR,
    timestamp       TIMESTAMPTZ NOT NULL,
    forecast        DOUBLE,
    lower_bound     DOUBLE,
    upper_bound     DOUBLE,
    is_future       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (coin_id, model_name, model_version, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_fc_coin_model ON forecasts (coin_id, model_name);

-- ── Model registry ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              INTEGER PRIMARY KEY,
    coin_id         VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    model_version   VARCHAR NOT NULL,
    model_path      VARCHAR,
    train_start     TIMESTAMPTZ,
    train_end       TIMESTAMPTZ,
    metrics         JSON,
    hyperparams     JSON,
    trained_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (coin_id, model_name, model_version)
);
"""

_SEQUENCES_SQL = """
CREATE SEQUENCE IF NOT EXISTS seq_market_prices  START 1;
CREATE SEQUENCE IF NOT EXISTS seq_ohlcv          START 1;
CREATE SEQUENCE IF NOT EXISTS seq_market_snapshot START 1;
CREATE SEQUENCE IF NOT EXISTS seq_forecasts      START 1;
CREATE SEQUENCE IF NOT EXISTS seq_model_registry START 1;
"""


class CryptoDatabase:
    """
    Thin wrapper around DuckDB that manages the crypto schema,
    upserts DataFrames, and exposes convenient query helpers.

    Example
    -------
    >>> db = CryptoDatabase()
    >>> db.upsert_market_prices(df)
    >>> df = db.get_price_series("bitcoin")
    """

    def __init__(self, db_path: str = settings.db_path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self.connect()

    # ── Schema init ───────────────────────────────────────────────────────

    def _init_schema(self):
        conn = self.connect()
        conn.execute(_SEQUENCES_SQL)
        # Execute each CREATE TABLE statement separately
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        logger.debug(f"Schema initialised at {self.db_path}")

    # ── Upserts ───────────────────────────────────────────────────────────

    def upsert_market_prices(self, df: pd.DataFrame) -> int:
        """Insert-or-replace daily price rows. Returns rows written."""
        if df.empty:
            return 0
        conn = self.conn
        # Register df as a view, then INSERT OR REPLACE
        conn.register("_mp_staging", df)
        conn.execute("""
            INSERT OR REPLACE INTO market_prices
                (id, coin_id, symbol, currency, timestamp, price, market_cap, volume)
            SELECT
                nextval('seq_market_prices'),
                coin_id, symbol, currency, timestamp, price, market_cap, volume
            FROM _mp_staging
        """)
        conn.unregister("_mp_staging")
        logger.info(f"Upserted {len(df)} rows → market_prices")
        return len(df)

    def upsert_ohlcv(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        conn = self.conn
        conn.register("_ohlcv_staging", df)
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv
                (id, coin_id, symbol, timestamp, open, high, low, close)
            SELECT
                nextval('seq_ohlcv'),
                coin_id, symbol, timestamp, open, high, low, close
            FROM _ohlcv_staging
        """)
        conn.unregister("_ohlcv_staging")
        logger.info(f"Upserted {len(df)} rows → ohlcv")
        return len(df)

    def upsert_market_snapshot(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        conn = self.conn
        conn.register("_snap_staging", df)
        conn.execute("""
            INSERT INTO market_snapshot
            SELECT nextval('seq_market_snapshot'), * FROM _snap_staging
        """)
        conn.unregister("_snap_staging")
        logger.info(f"Inserted {len(df)} rows → market_snapshot")
        return len(df)

    def upsert_forecasts(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        conn = self.conn
        conn.register("_fc_staging", df)
        conn.execute("""
            INSERT OR REPLACE INTO forecasts
                (id, coin_id, symbol, model_name, model_version,
                 timestamp, forecast, lower_bound, upper_bound, is_future)
            SELECT
                nextval('seq_forecasts'),
                coin_id, symbol, model_name, model_version,
                timestamp, forecast, lower_bound, upper_bound, is_future
            FROM _fc_staging
        """)
        conn.unregister("_fc_staging")
        logger.info(f"Upserted {len(df)} forecast rows → forecasts")
        return len(df)

    def register_model(
        self,
        coin_id: str,
        model_name: str,
        model_version: str,
        model_path: str,
        train_start: datetime,
        train_end: datetime,
        metrics: dict,
        hyperparams: dict,
    ):
        conn = self.conn
        conn.execute("""
            INSERT OR REPLACE INTO model_registry
                (id, coin_id, model_name, model_version, model_path,
                 train_start, train_end, metrics, hyperparams)
            VALUES (nextval('seq_model_registry'), ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            coin_id, model_name, model_version, model_path,
            train_start, train_end,
            json.dumps(metrics), json.dumps(hyperparams),
        ])
        logger.info(f"Registered model {model_name} v{model_version} for {coin_id}")

    # ── Queries ───────────────────────────────────────────────────────────

    def get_price_series(
        self,
        coin_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return daily price series for a coin, optionally filtered by date range."""
        conditions = ["coin_id = ?"]
        params: list = [coin_id]
        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)
        where = " AND ".join(conditions)
        df = self.conn.execute(
            f"SELECT * FROM market_prices WHERE {where} ORDER BY timestamp",
            params,
        ).df()
        return df

    def get_all_coins(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT coin_id FROM market_prices ORDER BY coin_id"
        ).fetchall()
        return [r[0] for r in rows]

    def get_forecasts(
        self,
        coin_id: str,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        conditions = ["coin_id = ?"]
        params: list = [coin_id]
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        where = " AND ".join(conditions)
        return self.conn.execute(
            f"SELECT * FROM forecasts WHERE {where} ORDER BY timestamp",
            params,
        ).df()

    def get_latest_model(self, coin_id: str, model_name: str) -> Optional[dict]:
        row = self.conn.execute("""
            SELECT * FROM model_registry
            WHERE coin_id = ? AND model_name = ?
            ORDER BY trained_at DESC LIMIT 1
        """, [coin_id, model_name]).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, row))

    def summary(self) -> pd.DataFrame:
        """Return a summary of all stored data by coin."""
        return self.conn.execute("""
            SELECT
                coin_id,
                symbol,
                COUNT(*)            AS price_rows,
                MIN(timestamp)      AS earliest,
                MAX(timestamp)      AS latest,
                MIN(price)          AS min_price,
                MAX(price)          AS max_price,
                AVG(price)          AS avg_price
            FROM market_prices
            GROUP BY coin_id, symbol
            ORDER BY coin_id
        """).df()

    def run_query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        return self.conn.execute(sql, params or []).df()
