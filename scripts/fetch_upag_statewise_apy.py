from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


URL = "https://dash.upag.gov.in/_dash-update-component"
REFERER = "https://dash.upag.gov.in/statewiseapy?hide=true"
GENERAL_NOTE = "Area in Lakh Ha, Production in Lakh Tonnes & Yield in Kg/Ha"
ESTIMATE_NOTE = "Data for the year 2025-26 is of 2nd Advance Estimates"
SPECIAL_PRODUCTION_NOTES = {
    "Cotton": "Cotton production is in Lakh Bales; 1 Bale = 170 Kg",
    "Jute": "Jute production is in Lakh Bales; 1 Bale = 180 Kg",
    "Mesta": "Mesta production is in Lakh Bales; 1 Bale = 180 Kg",
    "Jute & Mesta": "Jute & Mesta production is in Lakh Bales; 1 Bale = 180 Kg",
    "Sannhemp": "Sannhemp production is in Lakh Bales; 1 Bale = 180 Kg",
}


def fetch_swapy(
    crops: list[str],
    from_year: int,
    to_year: int,
    metrics: list[str],
    uom: str,
) -> list[dict]:
    payload = {
        "output": "..swapy-store.data...swapy-suffix-title.children...swapy-notification1.children...swapy-notification2.children...swapy-notification4.children...swapy-sheetname.children..",
        "outputs": [
            {"id": "swapy-store", "property": "data"},
            {"id": "swapy-suffix-title", "property": "children"},
            {"id": "swapy-notification1", "property": "children"},
            {"id": "swapy-notification2", "property": "children"},
            {"id": "swapy-notification4", "property": "children"},
            {"id": "swapy-sheetname", "property": "children"},
        ],
        "inputs": [
            {
                "id": "swapy-filters-store",
                "property": "data",
                "value": {
                    "crop": crops,
                    "fromyear": from_year,
                    "toyear": to_year,
                    "metric": metrics,
                    "uom": uom,
                },
            }
        ],
        "state": [{"id": "url", "property": "search", "value": "?hide=true"}],
        "changedPropIds": ["swapy-filters-store.data"],
    }
    req = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Referer": REFERER,
            "Origin": "https://dash.upag.gov.in",
            "User-Agent": "KisaanAi/1.0",
        },
    )
    with urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["response"]["swapy-store"]["data"]


def normalize_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "State": "state",
            "Season": "season",
            "Crop": "crop",
            "Crop Year": "crop_year",
            "Crop Year Code": "crop_year_code",
            "Value": "value",
            "Unit Of Measure": "unit_of_measure",
            "Metric": "metric",
            "Latest Estimate": "latest_estimate",
            "Estimation Cycle": "estimation_cycle",
            "Estimation Cycle Code": "estimation_cycle_code",
            "RecordType": "record_type",
        }
    )
    keep = [
        "state",
        "season",
        "crop",
        "crop_year",
        "crop_year_code",
        "metric",
        "value",
        "unit_of_measure",
        "latest_estimate",
        "estimation_cycle",
        "estimation_cycle_code",
        "record_type",
        "general_note",
        "estimate_note",
        "special_production_note",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df["general_note"] = GENERAL_NOTE
    df["estimate_note"] = df["crop_year"].astype(str).eq("2025-26").map(
        lambda x: ESTIMATE_NOTE if x else ""
    )
    df["special_production_note"] = df["crop"].map(SPECIAL_PRODUCTION_NOTES).fillna("")
    return df[keep].copy()


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["crop_year_code"] = pd.to_numeric(out["crop_year_code"], errors="coerce")
    out["latest_estimate"] = pd.to_numeric(out["latest_estimate"], errors="coerce").fillna(0)
    out = out.sort_values(
        by=["crop_year_code", "latest_estimate", "estimation_cycle_code"],
        ascending=[False, False, False],
    )
    out = out.drop_duplicates(subset=["state", "crop", "season", "metric"], keep="first")
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_year", type=int, default=2021)
    parser.add_argument("--to_year", type=int, default=2025)
    parser.add_argument("--uom", default="Lakh")
    parser.add_argument("--crops", default="All")
    parser.add_argument("--metrics", default="Area,Production,Yield")
    parser.add_argument("--out_csv", default="data/raw/live/upag_statewise_apy.csv")
    parser.add_argument("--latest_csv", default="data/processed/upag_statewise_apy_latest.csv")
    parser.add_argument("--meta_json", default="data/processed/upag_statewise_apy_meta.json")
    args = parser.parse_args()

    crops = [c.strip() for c in args.crops.split(",") if c.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = fetch_swapy(crops, args.from_year, args.to_year, metrics, args.uom)
    df = normalize_rows(rows)
    latest = latest_snapshot(df)

    out_csv = Path(args.out_csv)
    latest_csv = Path(args.latest_csv)
    meta_json = Path(args.meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    latest_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    latest.to_csv(latest_csv, index=False)
    meta = {
        "general_note": GENERAL_NOTE,
        "estimate_note": ESTIMATE_NOTE,
        "special_production_notes": SPECIAL_PRODUCTION_NOTES,
        "source": REFERER,
        "filters": {
            "crops": crops,
            "from_year": args.from_year,
            "to_year": args.to_year,
            "metrics": metrics,
            "uom": args.uom,
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(df)} rows -> {out_csv}")
    print(f"Saved {len(latest)} latest rows -> {latest_csv}")
    print(f"Saved metadata -> {meta_json}")


if __name__ == "__main__":
    main()
