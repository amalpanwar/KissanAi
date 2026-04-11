from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

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
    return _row_to_result(row, place)


def _row_to_result(row: dict | pd.Series, place_fallback: str) -> dict[str, Any]:
    return {
        "place": row.get("place", place_fallback),
        "sub_district": row.get("sub_district", ""),
        "district": row.get("district", ""),
        "state": row.get("state", ""),
        "source_file": row.get("source_file", ""),
    }


def _iter_norm_candidates(df: pd.DataFrame, cols: Iterable[str]) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for col in cols:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            norm = raw if col == "place_norm" else _normalize_place(raw)
            if not norm or len(norm) < 3:
                continue
            candidates.append((norm, row.to_dict()))
    return candidates


def lookup_place_in_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    df = _load_lookup()
    if df.empty:
        return None
    norm_text = _normalize_place(text)
    if not norm_text:
        return None

    candidates = _iter_norm_candidates(df, ["place_norm"])
    best: tuple[int, dict] | None = None
    for norm, row in candidates:
        if norm in norm_text:
            score = len(norm)
            if best is None or score > best[0]:
                best = (score, row)
    if best:
        return _row_to_result(best[1], text)

    # Fallback: match sub-district or district if explicitly mentioned
    candidates = _iter_norm_candidates(df, ["sub_district", "district"])
    for norm, row in candidates:
        if norm in norm_text:
            return _row_to_result(row, text)

    # Fallback: fuzzy match for minor spelling errors against place_norm
    try:
        import difflib

        norms = df["place_norm"].dropna().astype(str).unique().tolist() if "place_norm" in df.columns else []
        matches = difflib.get_close_matches(norm_text, norms, n=1, cutoff=0.85)
        if matches:
            row = df[df["place_norm"] == matches[0]].iloc[0].to_dict()
            return _row_to_result(row, text)
    except Exception:
        pass

    return None
