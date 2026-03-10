"""
ForecastVisualizer — interactive Plotly charts for price and forecast data.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ForecastVisualizer:
    """
    Produces interactive Plotly figures for forecast analysis.

    Example
    -------
    >>> viz = ForecastVisualizer()
    >>> fig = viz.plot_forecast(forecast_df, coin_id="bitcoin")
    >>> fig.show()
    """

    DEFAULT_COLORS = {
        "actual":   "#00b4d8",
        "forecast": "#ff6b6b",
        "ci":       "rgba(255,107,107,0.15)",
        "future":   "rgba(255,165,0,0.10)",
    }

    def plot_forecast(
        self,
        df: pd.DataFrame,
        coin_id: str,
        title: Optional[str] = None,
        show_ci: bool = True,
    ) -> go.Figure:
        """
        Plot actual prices against model forecast with confidence intervals.

        Parameters
        ----------
        df       : output of ForecastPredictor.forecast()  (has 'price' + 'forecast')
        coin_id  : coin identifier string
        title    : optional chart title override
        show_ci  : whether to draw upper/lower confidence bands
        """
        fig = go.Figure()

        # Split history vs future
        hist = df[~df["is_future"]].dropna(subset=["price"])
        fut  = df[df["is_future"]]
        all_fc = df.copy()

        # ── Confidence interval band ──────────────────────────────────────
        if show_ci and "upper_bound" in all_fc.columns:
            ci_df = all_fc.dropna(subset=["upper_bound", "lower_bound"])
            fig.add_trace(go.Scatter(
                x=pd.concat([ci_df["timestamp"], ci_df["timestamp"][::-1]]),
                y=pd.concat([ci_df["upper_bound"], ci_df["lower_bound"][::-1]]),
                fill="toself",
                fillcolor=self.DEFAULT_COLORS["ci"],
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True,
                name="95% CI",
            ))

        # ── Historical actuals ────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["price"],
            mode="lines",
            name="Actual Price",
            line=dict(color=self.DEFAULT_COLORS["actual"], width=1.5),
        ))

        # ── Historical forecast (in-sample) ──────────────────────────────
        fig.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["forecast"],
            mode="lines",
            name="Model Fit",
            line=dict(color=self.DEFAULT_COLORS["forecast"], width=1, dash="dot"),
        ))

        # ── Future forecast ───────────────────────────────────────────────
        if not fut.empty:
            # Shade the forecast region
            fig.add_vrect(
                x0=fut["timestamp"].min(),
                x1=fut["timestamp"].max(),
                fillcolor=self.DEFAULT_COLORS["future"],
                layer="below",
                line_width=0,
            )
            fig.add_trace(go.Scatter(
                x=fut["timestamp"],
                y=fut["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="orange", width=2.5),
                marker=dict(size=4),
            ))

        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else coin_id.upper()
        fig.update_layout(
            title=title or f"{symbol} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=60, r=20, t=60, b=60),
        )
        return fig

    def plot_multi_coin(
        self,
        forecasts: dict,
        normalise: bool = True,
    ) -> go.Figure:
        """
        Overlay forecasts for multiple coins on a single chart.

        Parameters
        ----------
        forecasts  : dict coin_id → forecast DataFrame
        normalise  : if True, index all prices to 100 at the first observation
        """
        fig = go.Figure()
        for coin_id, df in forecasts.items():
            hist = df[~df["is_future"]].dropna(subset=["price"])
            fut  = df[df["is_future"]]
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else coin_id.upper()

            y_hist = hist["price"]
            y_fut  = fut["forecast"]

            if normalise and len(y_hist) > 0:
                base   = y_hist.iloc[0]
                y_hist = y_hist / base * 100
                y_fut  = y_fut  / base * 100

            fig.add_trace(go.Scatter(
                x=hist["timestamp"], y=y_hist,
                mode="lines", name=f"{symbol} actual",
            ))
            if not fut.empty:
                fig.add_trace(go.Scatter(
                    x=fut["timestamp"], y=y_fut,
                    mode="lines", name=f"{symbol} forecast",
                    line=dict(dash="dash"),
                ))

        fig.update_layout(
            title="Multi-Coin Price Comparison (Indexed = 100)",
            xaxis_title="Date",
            yaxis_title="Indexed Price" if normalise else "Price (USD)",
            template="plotly_dark",
            hovermode="x unified",
        )
        return fig

    def plot_model_metrics(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Bar chart of model performance (MAPE) across all coins.

        metrics_df must have columns: coin_id, mape (and optionally mae, rmse).
        """
        metrics_df = metrics_df.sort_values("mape")
        fig = go.Figure(go.Bar(
            x=metrics_df["coin_id"],
            y=(metrics_df["mape"] * 100).round(2),
            text=(metrics_df["mape"] * 100).round(2).astype(str) + "%",
            textposition="outside",
            marker_color="#00b4d8",
        ))
        fig.update_layout(
            title="Model MAPE by Coin",
            xaxis_title="Coin",
            yaxis_title="MAPE (%)",
            template="plotly_dark",
        )
        return fig
