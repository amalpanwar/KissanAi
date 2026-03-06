from __future__ import annotations

import argparse
import os
import sys
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


def main() -> None:
    load_local_env(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource_id",
        default=os.getenv("DATA_GOV_RESOURCE_ID", ""),
        help="Data.gov.in resource id (or set DATA_GOV_RESOURCE_ID in .env)",
    )
    parser.add_argument("--api_key", default=os.getenv("DATA_GOV_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_records", type=int, default=100000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--state", default=os.getenv("DATA_GOV_STATE"))
    parser.add_argument("--district", default=os.getenv("DATA_GOV_DISTRICT"))
    parser.add_argument("--commodity", default=os.getenv("DATA_GOV_COMMODITY"))
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
    records = client.fetch_records(
        resource_id=args.resource_id,
        limit=args.limit,
        max_records=args.max_records,
        extra_params=extra_params,
    )
    if not records:
        print("No records returned.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
