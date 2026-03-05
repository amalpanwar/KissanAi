from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lstm_forecast import prepare_daily_series, train_and_forecast


def _guess_date_col(df: pd.DataFrame) -> str:
    candidates = ["arrival_date", "date", "timestamp", "recorded_at"]
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    raise ValueError(f"Could not infer date column. Available columns: {list(df.columns)}")


def _guess_value_col(df: pd.DataFrame) -> str:
    candidates = ["modal_price", "price", "value", "avg_price", "mean_price"]
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    raise ValueError(f"Could not infer value column. Available columns: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from data.gov fetch")
    parser.add_argument("--commodity", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--district", default=None)
    parser.add_argument("--date_col", default=None)
    parser.add_argument("--value_col", default=None)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--out", default="data/processed/forecast_15d.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    date_col = args.date_col or _guess_date_col(df)
    value_col = args.value_col or _guess_value_col(df)

    series = prepare_daily_series(
        df,
        date_col=date_col,
        value_col=value_col,
        commodity=args.commodity,
        state=args.state,
        district=args.district,
    )
    result = train_and_forecast(
        series_df=series,
        horizon_days=args.horizon,
        lookback=args.lookback,
        epochs=args.epochs,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.forecast.to_csv(out_path, index=False)

    print("Forecast saved:", out_path)
    print(result.forecast.to_string(index=False))


if __name__ == "__main__":
    main()
