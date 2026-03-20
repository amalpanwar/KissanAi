from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from statsmodels.tsa.statespace.sarimax import SARIMAX


class PriceLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


@dataclass
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame


def _auto_sarima_forecast(values: np.ndarray, horizon_days: int) -> np.ndarray:
    # Simple auto-SARIMA via small AIC grid search.
    best_aic = None
    best_model = None
    best_order = None
    best_seasonal = None

    # Small, fast grid to avoid heavy compute.
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    m = 7  # weekly seasonality for daily data

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal = (P, D, Q, m)
                            try:
                                model = SARIMAX(
                                    values,
                                    order=order,
                                    seasonal_order=seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                res = model.fit(disp=False)
                                aic = res.aic
                                if best_aic is None or aic < best_aic:
                                    best_aic = aic
                                    best_model = res
                                    best_order = order
                                    best_seasonal = seasonal
                            except Exception:
                                continue

    if best_model is None:
        raise ValueError("SARIMA model fitting failed.")

    forecast = best_model.forecast(steps=horizon_days)
    preds = np.asarray(forecast, dtype=np.float32)
    # Guard against negative prices
    preds[preds < 0] = 0.0
    return preds


def prepare_daily_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    commodity: str | None = None,
    state: str | None = None,
    district: str | None = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    cols_map = {c.lower(): c for c in out.columns}

    def _col(name: str) -> str:
        if name in out.columns:
            return name
        lowered = name.lower()
        if lowered in cols_map:
            return cols_map[lowered]
        raise ValueError(f"Column '{name}' not found. Available: {list(out.columns)}")

    date_col = _col(date_col)
    value_col = _col(value_col)

    if commodity and "commodity" in cols_map:
        c = cols_map["commodity"]
        out = out[out[c].astype(str).str.lower() == commodity.lower()]
    if state and "state" in cols_map:
        c = cols_map["state"]
        out = out[out[c].astype(str).str.lower() == state.lower()]
    if district and "district" in cols_map:
        c = cols_map["district"]
        out = out[out[c].astype(str).str.lower() == district.lower()]

    if out.empty:
        raise ValueError("No rows left after applying commodity/state/district filters.")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=dayfirst)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col])

    if out.empty:
        raise ValueError("No valid date/value rows found after cleaning.")

    series = (
        out.groupby(out[date_col].dt.date)[value_col]
        .mean()
        .reset_index()
        .rename(columns={date_col: "date", value_col: "value"})
    )
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Daily frequency with interpolation for missing days
    series = series.set_index("date").asfreq("D")
    series["value"] = series["value"].interpolate(method="linear").ffill().bfill()
    series = series.reset_index().sort_values("date").reset_index(drop=True)
    if not series["date"].is_monotonic_increasing:
        raise ValueError("Date series is not sorted in ascending order after preprocessing.")
    return series


def _make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i : i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


def train_and_forecast(
    series_df: pd.DataFrame,
    horizon_days: int = 15,
    lookback: int = 30,
    epochs: int = 120,
    lr: float = 0.001,
    seed: int = 42,
    train_window_days: int = 1095,
    max_daily_change_pct: float = 0.05,
    model: str = "sarima",
) -> ForecastResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = series_df.sort_values("date").reset_index(drop=True).copy()
    if train_window_days > 0 and len(data) > 0:
        cutoff = data["date"].max() - pd.Timedelta(days=int(train_window_days))
        recent = data[data["date"] >= cutoff].copy()
        if len(recent) > lookback + 5:
            data = recent

    values = data["value"].values.astype(np.float32)
    if len(values) <= lookback + 5:
        raise ValueError(
            f"Not enough history for lookback={lookback}. Need > {lookback + 5} daily points."
        )

    # Prefer SARIMA unless explicitly disabled; fallback to LSTM on failure.
    if model.lower() != "lstm":
        try:
            preds = _auto_sarima_forecast(values, horizon_days)
            start = data["date"].max() + pd.Timedelta(days=1)
            f_dates = pd.date_range(start=start, periods=horizon_days, freq="D")
            forecast_df = pd.DataFrame({"date": f_dates, "predicted_value": preds})
            return ForecastResult(history=data, forecast=forecast_df)
        except Exception:
            pass

    vmin, vmax = float(values.min()), float(values.max())
    denom = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
    scaled = (values - vmin) / denom

    X, y = _make_sequences(scaled, lookback)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = PriceLSTM(input_size=1, hidden_size=64, num_layers=1)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optim.step()

    # Recursive forecasting
    model.eval()
    window = scaled[-lookback:].copy()
    preds_scaled: list[float] = []
    for _ in range(horizon_days):
        x_in = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            nxt = model(x_in).item()
        nxt = float(max(0.0, min(1.0, nxt)))
        preds_scaled.append(nxt)
        window = np.concatenate([window[1:], np.array([nxt], dtype=np.float32)])

    preds = np.array(preds_scaled) * denom + vmin
    # Keep predictions anchored to current regime by capping daily jump.
    anchored_preds: list[float] = []
    prev = float(values[-1])
    max_pct = max(0.0, float(max_daily_change_pct))
    for p in preds.tolist():
        low = prev * (1.0 - max_pct)
        high = prev * (1.0 + max_pct)
        clipped = min(max(float(p), low), high)
        anchored_preds.append(clipped)
        prev = clipped

    preds = np.array(anchored_preds, dtype=np.float32)
    # If predictions are flat, fall back to recent trend.
    if float(np.std(preds)) < 1e-3:
        recent = values[-14:] if len(values) >= 14 else values
        if len(recent) >= 2:
            deltas = np.diff(recent)
            mean_delta = float(np.mean(deltas))
            if abs(mean_delta) < 1e-6 and len(recent) >= 8:
                mean_delta = float((recent[-1] - recent[-8]) / 7.0)
            trend = np.array([values[-1] + mean_delta * (i + 1) for i in range(horizon_days)])
            preds = trend.astype(np.float32)

    start = data["date"].max() + pd.Timedelta(days=1)
    f_dates = pd.date_range(start=start, periods=horizon_days, freq="D")

    forecast_df = pd.DataFrame({"date": f_dates, "predicted_value": preds})
    return ForecastResult(history=data, forecast=forecast_df)
