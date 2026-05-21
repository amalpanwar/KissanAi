from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_MAP = ROOT / "data" / "raw" / "agmarknet_group_commodities.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agmarknet_client import ALL_DISTRICTS, build_report_payload, build_snapshot_payload, build_url, extract_rows, fetch_page


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


def load_group_map(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = p.read_text(encoding="utf-8")
        raw = json.loads(data)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            continue
        ids = [str(x).strip() for x in v if str(x).strip()]
        out[str(k)] = ids
    return out


def iter_dates(from_date: str, to_date: str) -> list[str]:
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    if end < start:
        start, end = end, start
    cur = start
    out: list[str] = []
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def extract_units(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    data = payload.get("data") or {}
    columns = data.get("columns") if isinstance(data, dict) else None
    if columns is None and isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("columns"), list):
                columns = item.get("columns")
                break
    if not isinstance(columns, list):
        return None, None

    def unit_from_title(title: str) -> str | None:
        if not isinstance(title, str):
            return None
        m = re.search(r"\(([^()]+)\)", title)
        if not m:
            return None
        val = m.group(1).strip()
        return val or None

    price_unit: str | None = None
    arrival_unit: str | None = None
    for col in columns:
        if not isinstance(col, dict):
            continue
        key = str(col.get("key") or "")
        title = str(col.get("title") or "")
        if key == "wt_avg_price" and not price_unit:
            price_unit = unit_from_title(title)
        if key == "arrival" and not arrival_unit:
            arrival_unit = unit_from_title(title)
        if price_unit and arrival_unit:
            break
    return price_unit, arrival_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_date", default="", help="YYYY-MM-DD (optional if --keep_years set)")
    parser.add_argument("--to_date", default="", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--keep_years", type=int, default=0, help="If set, use rolling window ending today")
    parser.add_argument("--lookback_days", type=int, default=0, help="If set, use N-day window ending today")
    parser.add_argument("--state_ids", required=True, help="Comma list of state IDs")
    parser.add_argument("--district_ids", default="", help="Comma list of district IDs")
    parser.add_argument("--district_ids_file", default="", help="File with district IDs, one per line")
    parser.add_argument(
        "--districts_as_list",
        action="store_true",
        help="Send all district_ids as a single list param instead of looping each district",
    )
    parser.add_argument("--group_ids", default="", help="Comma list of commodity group IDs")
    parser.add_argument("--group_ids_file", default="", help="File with group IDs, one per line")
    parser.add_argument("--commodity_ids", default="", help="Comma list of commodity IDs")
    parser.add_argument("--commodity_ids_file", default="", help="File with commodity IDs, one per line")
    parser.add_argument(
        "--group_commodities_json",
        default=str(DEFAULT_GROUP_MAP),
        help="JSON map of group_id -> [commodity_id,...]",
    )
    parser.add_argument("--period", default="date")
    parser.add_argument("--type", default="3")
    parser.add_argument("--msp", default="0")
    parser.add_argument("--options", default="2", help="Comma list of options, e.g. 2 for price, 1 for arrivals")
    parser.add_argument(
        "--mode",
        choices=["report", "dashboard"],
        default="report",
        help="Agmarknet source mode. 'report' uses all-type-of-report at district level; 'dashboard' uses the narrower snapshot endpoint.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_pages", type=int, default=200)
    parser.add_argument("--sleep_sec", type=float, default=0.3)
    parser.add_argument("--timeout_sec", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--all_districts",
        action="store_true",
        help="Use Agmarknet's live all-district sentinel instead of local district IDs.",
    )
    parser.add_argument("--fail_log", default="data/raw/live/agmarknet_failures.csv")
    parser.add_argument("--merge_existing", action="store_true", help="Merge with existing CSV at --out")
    parser.add_argument("--trim_years", type=int, default=0, help="After merge, keep last N years only")
    parser.add_argument("--debug", action="store_true", help="Log failed URLs and continue")
    parser.add_argument("--out", default="data/raw/live/agmarknet_report.csv")
    args = parser.parse_args()

    today = date.today()
    to_date = args.to_date.strip() or today.isoformat()
    if args.lookback_days and args.lookback_days > 0:
        from_date = (today - timedelta(days=int(args.lookback_days))).isoformat()
    elif args.keep_years and not args.from_date:
        from_date = today.replace(year=today.year - args.keep_years).isoformat()
    else:
        from_date = args.from_date.strip()
    if not from_date:
        raise SystemExit("Provide --from_date or use --keep_years/--lookback_days.")

    state_ids = parse_ids(args.state_ids)
    district_ids = parse_ids(args.district_ids) + load_id_list(args.district_ids_file)
    group_ids = parse_ids(args.group_ids) + load_id_list(args.group_ids_file)
    commodity_ids = parse_ids(args.commodity_ids) + load_id_list(args.commodity_ids_file)
    option_ids = parse_ids(args.options)
    group_map = load_group_map(args.group_commodities_json)

    all_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    date_values = iter_dates(from_date, to_date)

    if args.mode == "dashboard":
        use_all_districts = args.all_districts or not district_ids
        live_district_ids = None if use_all_districts else district_ids

        for state_id in state_ids:
            for date_str in date_values:
                page = 1
                while page <= args.max_pages:
                    params = build_snapshot_payload(
                        state_id=state_id,
                        date_str=date_str,
                        page=page,
                        limit=args.limit,
                        district_ids=live_district_ids,
                    )
                    try:
                        payload = fetch_page(
                            params,
                            timeout_sec=args.timeout_sec,
                            retries=args.retries,
                        )
                    except Exception as exc:
                        if args.debug:
                            print(f"Fetch failed: {exc} | {build_url(params)}", file=sys.stderr)
                        failed_rows.append(
                            {
                                "mode": args.mode,
                                "state_id": state_id,
                                "date": date_str,
                                "page": page,
                                "error": str(exc),
                                "url": build_url(params),
                                "request_json": json.dumps(params, ensure_ascii=False),
                            }
                        )
                        break
                    rows = extract_rows(payload)
                    if not rows:
                        break
                    price_unit, arrival_unit = extract_units(payload)
                    for r in rows:
                        r = dict(r)
                        r["_state_id"] = state_id
                        r["_requested_date"] = date_str
                        if "reported_date" in r and "rep_date" not in r:
                            r["rep_date"] = r["reported_date"]
                        if "as_on" in r and "model_price_wt" not in r:
                            r["model_price_wt"] = r["as_on"]
                        if price_unit and (not r.get("unit_name_price") or str(r.get("unit_name_price")).strip().lower() == "nan"):
                            r["unit_name_price"] = price_unit
                        if arrival_unit and (not r.get("unit_name_arrival") or str(r.get("unit_name_arrival")).strip().lower() == "nan"):
                            r["unit_name_arrival"] = arrival_unit
                        all_rows.append(r)
                    pagination = payload.get("pagination") or {}
                    total_pages = int(pagination.get("total_pages") or 0)
                    if total_pages and page >= total_pages:
                        break
                    if len(rows) < args.limit:
                        break
                    page += 1
                    time.sleep(args.sleep_sec)
    else:
        if not group_ids and group_map:
            group_ids = sorted(group_map.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if not group_ids:
            raise SystemExit("Report mode requires --group_ids/--group_ids_file or a valid group-commodity map.")
        district_batch = district_ids or [""]
        if not option_ids:
            option_ids = ["2"]

        for state_id in state_ids:
            for group_id in group_ids:
                group_commodity_ids = commodity_ids or group_map.get(str(group_id), [])
                commodity_batch = [str(x).strip() for x in group_commodity_ids if str(x).strip()]
                if not commodity_batch:
                    commodity_batch = [""]
                for option in option_ids:
                    page = 1
                    while page <= args.max_pages:
                        params = build_report_payload(
                            state_id=state_id,
                            district_id=",".join(district_batch) if district_batch and district_batch != [""] else "",
                            group_id=group_id,
                            commodity_id=",".join(commodity_batch) if commodity_batch and commodity_batch != [""] else "",
                            from_date=from_date,
                            to_date=to_date,
                            option=option,
                            page=page,
                            limit=args.limit,
                            period=args.period,
                            type_value=args.type,
                            msp=args.msp,
                        )
                        if district_batch and district_batch != [""]:
                            params["district"] = [int(x) for x in district_batch if str(x).strip()]
                        if commodity_batch and commodity_batch != [""]:
                            params["commodity"] = [int(x) for x in commodity_batch if str(x).strip()]
                        try:
                            payload = fetch_page(
                                params,
                                timeout_sec=args.timeout_sec,
                                retries=args.retries,
                            )
                        except Exception as exc:
                            if args.debug:
                                print(f"Fetch failed: {exc} | {build_url(params)}", file=sys.stderr)
                            failed_rows.append(
                                {
                                    "mode": args.mode,
                                    "state_id": state_id,
                                    "district_ids": ",".join(district_batch),
                                    "group_id": group_id,
                                    "commodity_ids": ",".join(commodity_batch),
                                    "option": option,
                                    "page": page,
                                    "error": str(exc),
                                    "url": build_url(params),
                                    "request_json": json.dumps(params, ensure_ascii=False),
                                }
                            )
                            break
                        rows = extract_rows(payload)
                        if not rows:
                            break
                        for r in rows:
                            r = dict(r)
                            r["_state_id"] = state_id
                            r["_district_ids"] = ",".join(district_batch)
                            r["_group_id"] = group_id
                            r["_commodity_ids"] = ",".join(commodity_batch)
                            r["_option"] = option
                            if "reported_date" in r and "rep_date" not in r:
                                r["rep_date"] = r["reported_date"]
                            if "as_on" in r and "model_price_wt" not in r:
                                r["model_price_wt"] = r["as_on"]
                            all_rows.append(r)
                        pagination = payload.get("pagination") or {}
                        total_pages = int(pagination.get("total_pages") or 0)
                        if total_pages and page >= total_pages:
                            break
                        if len(rows) < args.limit:
                            break
                        page += 1
                        time.sleep(args.sleep_sec)

    if not all_rows:
        print("No records returned.")
        if failed_rows:
            fail_path = Path(args.fail_log)
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failed_rows).to_csv(fail_path, index=False)
            print(f"Wrote failures to {fail_path}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    fetched_price_units = (
        [u for u in df.get("unit_name_price", pd.Series(dtype=object)).dropna().astype(str) if u and u.lower() != "nan"]
        if "unit_name_price" in df.columns
        else []
    )
    fetched_arrival_units = (
        [u for u in df.get("unit_name_arrival", pd.Series(dtype=object)).dropna().astype(str) if u and u.lower() != "nan"]
        if "unit_name_arrival" in df.columns
        else []
    )
    inferred_price_unit = fetched_price_units[0] if fetched_price_units else None
    inferred_arrival_unit = fetched_arrival_units[0] if fetched_arrival_units else None
    if args.merge_existing and out_path.exists():
        try:
            old = pd.read_csv(out_path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    if inferred_price_unit:
        if "unit_name_price" not in df.columns:
            df["unit_name_price"] = inferred_price_unit
        else:
            mask = df["unit_name_price"].isna() | (df["unit_name_price"].astype(str).str.strip() == "") | (
                df["unit_name_price"].astype(str).str.lower() == "nan"
            )
            df.loc[mask, "unit_name_price"] = inferred_price_unit
    if inferred_arrival_unit:
        if "unit_name_arrival" not in df.columns:
            df["unit_name_arrival"] = inferred_arrival_unit
        else:
            mask = df["unit_name_arrival"].isna() | (df["unit_name_arrival"].astype(str).str.strip() == "") | (
                df["unit_name_arrival"].astype(str).str.lower() == "nan"
            )
            df.loc[mask, "unit_name_arrival"] = inferred_arrival_unit
    if not df.empty:
        # Deduplicate on common key columns if present.
        key_cols = [
            c
            for c in [
                "state_name",
                "district_name",
                "market_name",
                "cmdt_name",
                "rep_date",
                "model_price_wt",
            ]
            if c in df.columns
        ]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="last")
    if args.trim_years and not df.empty and "rep_date" in df.columns:
        df["_rep_dt"] = pd.to_datetime(df["rep_date"], errors="coerce", dayfirst=True)
        cutoff = pd.Timestamp(today) - pd.Timedelta(days=int(args.trim_years) * 365)
        df = df[df["_rep_dt"] >= cutoff].drop(columns=["_rep_dt"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    if failed_rows:
        fail_path = Path(args.fail_log)
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed_rows).to_csv(fail_path, index=False)
        print(f"Wrote failures to {fail_path}")


if __name__ == "__main__":
    main()
