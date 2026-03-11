from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

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


def merge_and_trim(out_path: Path, new_df: pd.DataFrame, keep_years: int) -> pd.DataFrame:
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
        if keep_years > 0:
            cutoff = merged["Arrival_Date_dt"].max() - pd.Timedelta(days=365 * keep_years)
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
    return merged


def main() -> None:
    load_local_env(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource_id",
        default=os.getenv("DATA_GOV_RESOURCE_ID", ""),
        help="Data.gov.in resource id",
    )
    parser.add_argument("--api_key", default=os.getenv("DATA_GOV_API_KEY", ""))
    parser.add_argument("--state", default="Uttar Pradesh")
    parser.add_argument(
        "--districts",
        default=(
            "Meerut,Muzaffarnagar,Baghpat,Saharanpur,Shamli,Bulandshahr,"
            "Aligarh,Greater Noida,Gautam Buddha Nagar"
        ),
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max_records", type=int, default=50000)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--keep_years", type=int, default=3)
    parser.add_argument(
        "--out_dir",
        default="data/raw/live/by_district",
        help="Output directory for per-district CSVs",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Provide --api_key or set DATA_GOV_API_KEY env var.")
    if not args.resource_id:
        raise ValueError("Provide --resource_id or set DATA_GOV_RESOURCE_ID env var.")

    districts = [d.strip() for d in args.districts.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_date = None
    if args.keep_years > 0:
        cutoff_date = date.today() - timedelta(days=365 * args.keep_years)

    client = DataGovClient(api_key=args.api_key, timeout_sec=args.timeout, retries=args.retries)

    for d in districts:
        extra_params: dict[str, str] = {
            "filters[State]": args.state,
            "filters[District]": d,
            "sort[Arrival_Date]": "desc",
        }
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
            print(f"{d}: no records returned")
            continue

        new_df = pd.DataFrame(records)
        out_path = out_dir / f"datagov_{d.lower().replace(' ', '_')}.csv"
        merged = merge_and_trim(out_path, new_df, args.keep_years)
        merged.to_csv(out_path, index=False)
        print(f"{d}: fetched {len(new_df)} rows, stored {len(merged)} rows -> {out_path}")


if __name__ == "__main__":
    main()
