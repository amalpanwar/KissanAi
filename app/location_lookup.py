from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _normalize_place(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


@lru_cache(maxsize=4)
def _load_lookup(mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
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


def _get_lookup() -> pd.DataFrame:
    path = Path("data/processed/location_lookup.csv")
    mtime = path.stat().st_mtime_ns if path.exists() else 0
    return _load_lookup(mtime)


def lookup_place(place: str) -> dict[str, Any] | None:
    if not place:
        return None
    df = _get_lookup()
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
        "lat": row.get("lat", ""),
        "lon": row.get("lon", ""),
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
    df = _get_lookup()
    if df.empty:
        return None
    norm_text = _normalize_place(text)
    if not norm_text:
        return None

    # First pass: n-gram token matching for Hinglish queries like "doghat me ..."
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if tokens:
        phrase_norms = set()
        for size in range(min(4, len(tokens)), 0, -1):
            for i in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[i : i + size])
                phrase_norms.add(_normalize_place(phrase))
        if "place_norm" in df.columns:
            place_map = {str(v): row for v, row in zip(df["place_norm"], df.to_dict(orient="records")) if v}
            for norm in phrase_norms:
                if norm in place_map:
                    return _row_to_result(place_map[norm], norm)
            # Prefix match: handle shortened village names like "doghat" -> "doghatrural"
            best_row = None
            best_len = None
            for norm in phrase_norms:
                if len(norm) < 4:
                    continue
                matches = [k for k in place_map.keys() if k.startswith(norm)]
                if not matches:
                    continue
                pick = min(matches, key=len)
                if best_len is None or len(pick) < best_len:
                    best_len = len(pick)
                    best_row = place_map[pick]
            if best_row:
                return _row_to_result(best_row, best_row.get("place", text))
        # Fallback: match sub-district or district if explicitly mentioned
        for col in ["sub_district", "district"]:
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                cand = (row.get(col) or "").strip()
                if not cand:
                    continue
                if _normalize_place(cand) in phrase_norms:
                    row_dict = row.to_dict()
                    row_dict["place"] = cand
                    return _row_to_result(row_dict, cand)

    # Fallback: prefix match for shortened village names (e.g., "doghat" -> "doghatrural")
    try:
        if "place_norm" in df.columns:
            norm_text = _normalize_place(text)
            if len(norm_text) >= 4:
                matches = df[df["place_norm"].astype(str).str.startswith(norm_text)]
                if not matches.empty:
                    # pick the shortest place_norm to avoid overshooting
                    matches = matches.copy()
                    matches["plen"] = matches["place_norm"].astype(str).str.len()
                    row = matches.sort_values("plen").iloc[0].to_dict()
                    return _row_to_result(row, row.get("place", text))
    except Exception:
        pass

    # Fallback: handle v/w swap (Kurava vs Kurawa)
    try:
        if "place_norm" in df.columns:
            norm_text = _normalize_place(text)
            if "v" in norm_text:
                alt = norm_text.replace("v", "w")
                matches = df[df["place_norm"].astype(str) == alt]
                if not matches.empty:
                    row = matches.iloc[0].to_dict()
                    return _row_to_result(row, row.get("place", text))
            if "w" in norm_text:
                alt = norm_text.replace("w", "v")
                matches = df[df["place_norm"].astype(str) == alt]
                if not matches.empty:
                    row = matches.iloc[0].to_dict()
                    return _row_to_result(row, row.get("place", text))
    except Exception:
        pass

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
