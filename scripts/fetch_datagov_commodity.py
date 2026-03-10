from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.datagov_client import DataGovClient


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


def main() -> None:
    load_local_env(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource_id",
        default=os.getenv("DATA_GOV_RESOURCE_ID", ""),
        help="Data.gov.in resource id (or set DATA_GOV_RESOURCE_ID in .env)",
    )
    parser.add_argument("--api_key", default=os.getenv("DATA_GOV_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max_records", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--state", default=os.getenv("DATA_GOV_STATE"))
    parser.add_argument("--district", default=os.getenv("DATA_GOV_DISTRICT"))
    parser.add_argument("--commodity", default=os.getenv("DATA_GOV_COMMODITY"))
    parser.add_argument("--keep_years", type=int, default=3)
    parser.add_argument("--recent_only", action="store_true", default=True)
    parser.add_argument(
        "--out",
        default=os.getenv("DATA_GOV_OUT_CSV", "data/raw/live/datagov_commodity.csv"),
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Provide --api_key or set DATA_GOV_API_KEY env var.")
    if not args.resource_id:
        raise ValueError("Provide --resource_id or set DATA_GOV_RESOURCE_ID env var.")

    client = DataGovClient(api_key=args.api_key, timeout_sec=args.timeout, retries=args.retries)
    extra_params: dict[str, str] = {}
    if args.state:
        extra_params["filters[State]"] = args.state
    if args.district:
        extra_params["filters[District]"] = args.district
    if args.commodity:
        extra_params["filters[Commodity]"] = args.commodity
    extra_params["sort[Arrival_Date]"] = "desc"
    cutoff_date = None
    if args.keep_years > 0:
        cutoff_date = date.today() - timedelta(days=365 * args.keep_years)

    records = client.fetch_records(
        resource_id=args.resource_id,
        limit=args.limit,
        max_records=args.max_records,
        extra_params=extra_params,
        stop_date=cutoff_date,
        date_field="Arrival_Date",
        dayfirst=True,
    )
    if not records:
        print("No records returned.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if out_path.exists():
        try:
            old_df = pd.read_csv(out_path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            merged = new_df.copy()
    else:
        merged = new_df.copy()

    if "Arrival_Date" in merged.columns:
        merged["Arrival_Date_dt"] = pd.to_datetime(
            merged["Arrival_Date"], errors="coerce", dayfirst=True
        )
        merged = merged.dropna(subset=["Arrival_Date_dt"])

        if args.keep_years > 0:
            cutoff = merged["Arrival_Date_dt"].max() - pd.Timedelta(days=365 * args.keep_years)
            merged = merged[merged["Arrival_Date_dt"] >= cutoff]

    key_cols = [
        c
        for c in [
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade",
            "Arrival_Date",
            "Modal_Price",
        ]
        if c in merged.columns
    ]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep="last")

    merged = merged.drop(columns=["Arrival_Date_dt"], errors="ignore")
    merged.to_csv(out_path, index=False)
    print(f"Fetched {len(new_df)} rows; stored {len(merged)} rows to {out_path}")
    print("Columns:", ", ".join(merged.columns))


if __name__ == "__main__":
    main()
