from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


DDL = [
    """
    CREATE TABLE IF NOT EXISTS research_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT NOT NULL,
        title TEXT,
        publication_year INTEGER,
        domain TEXT,
        district TEXT,
        text_content TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS advisories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id TEXT,
        district TEXT NOT NULL,
        season TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        recommendation_text TEXT NOT NULL,
        budget_min_inr REAL,
        budget_max_inr REAL,
        expected_yield_qtl_per_acre REAL,
        expected_revenue_inr_per_acre REAL,
        confidence REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        advisory_id INTEGER,
        farmer_id TEXT,
        actual_crop TEXT,
        actual_yield_qtl_per_acre REAL,
        actual_revenue_inr_per_acre REAL,
        weather_summary TEXT,
        notes TEXT,
        recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(advisory_id) REFERENCES advisories(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS crop_economics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT NOT NULL,
        season TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        cost_min_inr_per_acre REAL,
        cost_max_inr_per_acre REAL,
        market_price_inr_per_qtl REAL,
        avg_yield_qtl_per_acre REAL,
        source TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        for stmt in DDL:
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def insert_research_document(db_path: str | Path, row: dict[str, Any]) -> None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO research_documents (
                source_file, title, publication_year, domain, district, text_content
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("source_file"),
                row.get("title"),
                row.get("publication_year"),
                row.get("domain"),
                row.get("district"),
                row.get("text_content", ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()
