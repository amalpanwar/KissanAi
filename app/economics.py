from __future__ import annotations

import sqlite3
from pathlib import Path


def estimate_crop_profit(
    db_path: str | Path,
    district: str,
    season: str,
    crop_name: str,
) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cost_min_inr_per_acre, cost_max_inr_per_acre,
                   market_price_inr_per_qtl, avg_yield_qtl_per_acre
            FROM crop_economics
            WHERE lower(district) = lower(?)
              AND lower(season) = lower(?)
              AND lower(crop_name) = lower(?)
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (district, season, crop_name),
        )
        row = cur.fetchone()
        if not row:
            return None

        revenue = row["market_price_inr_per_qtl"] * row["avg_yield_qtl_per_acre"]
        return {
            "cost_min": row["cost_min_inr_per_acre"],
            "cost_max": row["cost_max_inr_per_acre"],
            "expected_revenue": revenue,
            "expected_profit_min": revenue - row["cost_max_inr_per_acre"],
            "expected_profit_max": revenue - row["cost_min_inr_per_acre"],
        }
    finally:
        conn.close()
