from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import init_db


def main() -> None:
    cfg = load_config()
    init_db(cfg.paths["sqlite_db"])
    csv_path = Path("data/raw/live/agmarknet_report.csv")
    if not csv_path.exists():
        print("agmarknet_report.csv not found.")
        return
    df = pd.read_csv(csv_path)
    # Support both cleaned and raw agmarknet schemas
    if {"District", "Commodity", "Modal_Price", "Arrival_Date"}.issubset(df.columns):
        df = df.rename(
            columns={
                "District": "district",
                "Commodity": "commodity",
                "Modal_Price": "modal_price",
                "Arrival_Date": "arrival_date",
                "Price_Unit": "price_unit",
            }
        )
    elif {"district_name", "cmdt_name", "as_on", "reported_date"}.issubset(df.columns):
        df = df.rename(
            columns={
                "district_name": "district",
                "cmdt_name": "commodity",
                "as_on": "modal_price",
                "reported_date": "arrival_date",
            }
        )
    elif {"district_name", "cmdt_name", "model_price_wt", "rep_date"}.issubset(df.columns):
        df = df.rename(
            columns={
                "district_name": "district",
                "cmdt_name": "commodity",
                "model_price_wt": "modal_price",
                "rep_date": "arrival_date",
                "unit_name_price": "price_unit",
            }
        )
    else:
        print("agmarknet_report.csv missing required columns.")
        return
    df["source"] = "agmarknet_report.csv"
    keep = ["district", "commodity", "modal_price", "arrival_date", "price_unit", "source"]
    df = df[keep].dropna(subset=["district", "commodity", "modal_price"])

    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM market_prices")
        conn.commit()
        df.to_sql("market_prices", conn, if_exists="append", index=False)
        print(f"Loaded {len(df)} rows into market_prices.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
