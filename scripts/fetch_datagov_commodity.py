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

DISTRICT_ALIASES = {
    "Baghpat": ["Bagpat"],
    "Muzaffarnagar": ["Mujaffarnagar", "Muzaffar Nagar"],
    "Gautam Buddha Nagar": ["Gautam Budh Nagar"],
    "Greater Noida": ["Gautam Buddha Nagar", "Gautam Budh Nagar"],
    "Bulandshahr": ["Buland Shahar"],
    "Saharanpur": ["Saharan Pur"],
}


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
    parser.add_argument("--max_records", type=int, default=50000)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--state", default=os.getenv("DATA_GOV_STATE", "Uttar Pradesh"))
    parser.add_argument("--district", default=None)
    parser.add_argument("--commodity", default=None)
    parser.add_argument(
        "--districts",
        default="Meerut,Muzaffarnagar,Baghpat,Saharanpur,Shamli,Bulandshahr,Aligarh,Greater Noida,Gautam Buddha Nagar",
        help="Comma-separated districts for Western UP",
    )
    parser.add_argument("--use_env_filters", action="store_true", default=False)
    parser.add_argument("--use_aliases", action="store_true", default=False)
    parser.add_argument("--state_only", action="store_true", default=False)
    parser.add_argument("--no_timeout", action="store_true", default=True)
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

    timeout_val = None if args.no_timeout else args.timeout
    client = DataGovClient(api_key=args.api_key, timeout_sec=timeout_val, retries=args.retries)

    cutoff_date = None
    if args.keep_years > 0:
        cutoff_date = date.today() - timedelta(days=365 * args.keep_years)

    if args.use_env_filters:
        if args.district is None:
            args.district = os.getenv("DATA_GOV_DISTRICT")
        if args.commodity is None:
            args.commodity = os.getenv("DATA_GOV_COMMODITY")

    district_list = [d.strip() for d in (args.districts or "").split(",") if d.strip()]

    all_records: list[dict] = []
    if args.district:
        district_list = [args.district]

    if not district_list:
        district_list = [""]

    if args.state_only:
        district_list = [""]

    for d in district_list:
        candidates = [d]
        if args.use_aliases:
            candidates = [d] + DISTRICT_ALIASES.get(d, [])
        for cand in candidates:
            extra_params: dict[str, str] = {}
            if args.state:
                extra_params["filters[State]"] = args.state
            if cand:
                extra_params["filters[District]"] = cand
            if args.commodity:
                extra_params["filters[Commodity]"] = args.commodity
            extra_params["sort[Arrival_Date]"] = "desc"

            records = client.fetch_records(
                resource_id=args.resource_id,
                limit=args.limit,
                max_records=args.max_records,
                extra_params=extra_params,
                stop_date=cutoff_date,
                date_field="Arrival_Date",
                dayfirst=True,
            )
            if records:
                all_records.extend(records)
                break
    if not all_records:
        # Fallback: try state-only (no district/commodity) once
        extra_params = {"filters[State]": args.state, "sort[Arrival_Date]": "desc"} if args.state else {}
        fallback = client.fetch_records(
            resource_id=args.resource_id,
            limit=args.limit,
            max_records=args.max_records,
            extra_params=extra_params,
            stop_date=cutoff_date,
            date_field="Arrival_Date",
            dayfirst=True,
        )
        if not fallback:
            print("No records returned (even after state-only fallback).")
            return
        all_records = fallback

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(all_records)

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
