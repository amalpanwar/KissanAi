from __future__ import annotations

from pathlib import Path
import pandas as pd


UPAG_CROP_ALIASES = {
    "Paddy": ["Rice"],
    "Rice": ["Rice"],
    "Wheat": ["Wheat"],
    "Mustard": ["Rapeseed & Mustard"],
    "Sugarcane": ["Sugarcane"],
    "Maize": ["Maize"],
    "Black Gram": ["Urad"],
    "Green Gram": ["Moong"],
    "Arhar": ["Tur"],
    "Groundnut": ["Groundnut"],
}


def kg_ha_to_qtl_acre(value_kg_ha: float) -> float:
    return float(value_kg_ha) / 100.0 / 2.47105381


def load_latest_up_yield_qtl_per_acre(
    crop_name: str,
    season: str | None = None,
    csv_path: str | Path = "data/processed/upag_statewise_apy_latest.csv",
) -> dict | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "state" not in df.columns:
        return None
    aliases = UPAG_CROP_ALIASES.get(crop_name, [crop_name])
    subset = df[
        df["state"].astype(str).str.lower().eq("uttar pradesh")
        & df["metric"].astype(str).str.lower().eq("yield")
        & df["crop"].isin(aliases)
    ].copy()
    if subset.empty:
        return None
    if season:
        season_subset = subset[subset["season"].astype(str).str.lower().eq(season.lower())]
        if not season_subset.empty:
            subset = season_subset
    if subset.empty:
        return None
    subset["crop_year_code"] = pd.to_numeric(subset.get("crop_year_code"), errors="coerce")
    subset = subset.sort_values(
        by=["crop_year_code", "latest_estimate", "season"],
        ascending=[False, False, True],
    )
    row = subset.iloc[0]
    unit = str(row.get("unit_of_measure") or "")
    value = pd.to_numeric(row.get("value"), errors="coerce")
    if pd.isna(value):
        return None
    if unit.lower() == "kg/ha":
        qtl_acre = kg_ha_to_qtl_acre(float(value))
    else:
        return None
    return {
        "yield_qtl_per_acre": qtl_acre,
        "source": "UPAG",
        "crop": row.get("crop"),
        "season": row.get("season"),
        "crop_year": row.get("crop_year"),
        "unit": unit,
        "raw_value": float(value),
        "estimation_cycle": row.get("estimation_cycle"),
        "estimate_note": row.get("estimate_note"),
        "general_note": row.get("general_note"),
        "special_production_note": row.get("special_production_note"),
    }
