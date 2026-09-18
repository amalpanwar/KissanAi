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
        raise SystemExit(2)
    df = pd.read_csv(csv_path, low_memory=False)
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
    elif {"district_name", "cmdt_name"}.issubset(df.columns):
        df = df.rename(columns={"district_name": "district", "cmdt_name": "commodity", "unit_name_price": "price_unit"})
        df["modal_price"] = pd.to_numeric(df.get("model_price_wt", pd.Series(index=df.index, dtype=float)), errors="coerce")
        df["modal_price"] = df["modal_price"].fillna(pd.to_numeric(df.get("as_on", pd.Series(index=df.index, dtype=float)), errors="coerce"))
        df["arrival_date"] = df.get("rep_date", pd.Series(index=df.index, dtype=object))
        df["arrival_date"] = df["arrival_date"].fillna(df.get("reported_date", pd.Series(index=df.index, dtype=object)))
    else:
        print("agmarknet_report.csv missing required columns.")
        raise SystemExit(2)
    if "price_unit" not in df:
        df["price_unit"] = "Rs./Quintal"
    df["source"] = "agmarknet_report.csv"
    keep = ["district", "commodity", "modal_price", "arrival_date", "price_unit", "source"]
    df = df[keep].dropna(subset=["district", "commodity", "modal_price"])

    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    try:
        cur = conn.cursor()
        with conn:
            cur.execute("DELETE FROM market_prices")
            cur.executemany(
                "INSERT INTO market_prices (district, commodity, modal_price, arrival_date, price_unit, source) VALUES (?, ?, ?, ?, ?, ?)",
                df.where(pd.notna(df), None).itertuples(index=False, name=None),
            )
        print(f"Loaded {len(df)} rows into market_prices.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
