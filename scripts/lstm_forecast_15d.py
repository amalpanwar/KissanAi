from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lstm_forecast import prepare_daily_series, train_and_forecast


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


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
    load_local_env(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.getenv("DATA_GOV_OUT_CSV", "data/raw/live/datagov_commodity.csv"),
        help="CSV from data.gov fetch",
    )
    parser.add_argument("--commodity", default=os.getenv("DATA_GOV_COMMODITY"))
    parser.add_argument("--state", default=os.getenv("DATA_GOV_STATE"))
    parser.add_argument("--district", default=os.getenv("DATA_GOV_DISTRICT"))
    parser.add_argument("--date_col", default=None)
    parser.add_argument("--value_col", default=None)
    parser.add_argument("--dayfirst", action="store_true", default=True)
    parser.add_argument("--monthfirst", action="store_true")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--min_points", type=int, default=90)
    parser.add_argument("--no_auto_broaden", action="store_true")
    parser.add_argument(
        "--out",
        default=os.getenv("FORECAST_OUT_CSV", "data/processed/forecast_15d.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    date_col = args.date_col or _guess_date_col(df)
    value_col = args.value_col or _guess_value_col(df)

    parse_dayfirst = False if args.monthfirst else args.dayfirst

    # Try strict filter first; if too few points, broaden scope progressively.
    candidates = [
        ("district", args.commodity, args.state, args.district),
        ("state", args.commodity, args.state, None),
        ("commodity", args.commodity, None, None),
        ("all", None, None, None),
    ]
    if args.no_auto_broaden:
        candidates = [("strict", args.commodity, args.state, args.district)]

    selected_scope = "strict"
    series = None
    last_err: Exception | None = None
    for scope, commodity, state, district in candidates:
        try:
            candidate = prepare_daily_series(
                df,
                date_col=date_col,
                value_col=value_col,
                commodity=commodity,
                state=state,
                district=district,
                dayfirst=parse_dayfirst,
            )
            if len(candidate) >= args.min_points:
                series = candidate
                selected_scope = scope
                break
            # Keep last candidate as fallback if none reach threshold
            if series is None or len(candidate) > len(series):
                series = candidate
                selected_scope = scope
        except Exception as e:
            last_err = e
            continue

    if series is None:
        raise ValueError(f"Could not build time series. Last error: {last_err}")
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
    print(
        "Series diagnostics:",
        {
            "scope_used": selected_scope,
            "points": int(len(series)),
            "start_date": str(series["date"].min().date()),
            "end_date": str(series["date"].max().date()),
            "lookback": args.lookback,
            "horizon": args.horizon,
        },
    )
    print(result.forecast.to_string(index=False))


if __name__ == "__main__":
    main()
