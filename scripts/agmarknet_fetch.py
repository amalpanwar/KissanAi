from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agmarknet_client import extract_rows, fetch_page


def parse_ids(val: str) -> list[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


def load_id_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # accept CSV with one column or raw lines
    if "," in text and "\n" in text:
        # try CSV first column
        rows = []
        with p.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                rows.append(row[0].strip())
        return [x for x in rows if x]
    return [x.strip() for x in text.splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--state_ids", required=True, help="Comma list of state IDs")
    parser.add_argument("--district_ids", default="", help="Comma list of district IDs")
    parser.add_argument("--district_ids_file", default="", help="File with district IDs, one per line")
    parser.add_argument("--group_ids", required=True, help="Comma list of commodity group IDs")
    parser.add_argument("--commodity_ids", default="", help="Comma list of commodity IDs")
    parser.add_argument("--commodity_ids_file", default="", help="File with commodity IDs, one per line")
    parser.add_argument("--period", default="date")
    parser.add_argument("--type", default="3")
    parser.add_argument("--msp", default="0")
    parser.add_argument("--options", default="2", help="Comma list of options, e.g. 2 for price, 1 for arrivals")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_pages", type=int, default=200)
    parser.add_argument("--sleep_sec", type=float, default=0.3)
    parser.add_argument("--out", default="data/raw/live/agmarknet_report.csv")
    args = parser.parse_args()

    state_ids = parse_ids(args.state_ids)
    district_ids = parse_ids(args.district_ids) + load_id_list(args.district_ids_file)
    group_ids = parse_ids(args.group_ids)
    commodity_ids = parse_ids(args.commodity_ids) + load_id_list(args.commodity_ids_file)
    option_ids = parse_ids(args.options)

    if not district_ids:
        district_ids = [""]
    if not commodity_ids:
        commodity_ids = [""]

    all_rows: list[dict[str, Any]] = []

    for state_id in state_ids:
        for district_id in district_ids:
            for group_id in group_ids:
                for commodity_id in commodity_ids:
                    for option in option_ids:
                        page = 1
                        while page <= args.max_pages:
                            params = {
                                "type": args.type,
                                "from_date": args.from_date,
                                "to_date": args.to_date,
                                "msp": args.msp,
                                "period": args.period,
                                "group": f"[{group_id}]",
                                "commodity": f"[{commodity_id}]",
                                "state": f"[{state_id}]",
                                "district": f"[{district_id}]",
                                "market": "[]",
                                "page": page,
                                "options": option,
                                "limit": args.limit,
                            }
                            payload = fetch_page(params)
                            rows = extract_rows(payload)
                            if not rows:
                                break
                            for r in rows:
                                r = dict(r)
                                r["_state_id"] = state_id
                                r["_district_id"] = district_id
                                r["_group_id"] = group_id
                                r["_commodity_id"] = commodity_id
                                r["_option"] = option
                                all_rows.append(r)
                            if len(rows) < args.limit:
                                break
                            page += 1
                            time.sleep(args.sleep_sec)

    if not all_rows:
        print("No records returned.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
