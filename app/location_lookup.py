from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_place(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


@lru_cache(maxsize=1)
def _load_lookup() -> pd.DataFrame:
    path = Path("data/processed/location_lookup.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "place_norm" not in df.columns and "place" in df.columns:
        df["place_norm"] = df["place"].astype(str).map(_normalize_place)
    return df


def lookup_place(place: str) -> dict[str, Any] | None:
    if not place:
        return None
    df = _load_lookup()
    if df.empty or "place_norm" not in df.columns:
        return None
    key = _normalize_place(place)
    if not key:
        return None
    matches = df[df["place_norm"] == key]
    if matches.empty:
        return None
    row = matches.iloc[0].to_dict()
    return {
        "place": row.get("place", place),
        "sub_district": row.get("sub_district", ""),
        "district": row.get("district", ""),
        "state": row.get("state", ""),
        "source_file": row.get("source_file", ""),
    }
