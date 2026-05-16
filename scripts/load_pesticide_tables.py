from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import init_db


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").replace("\n", " ").split())


def _source_stem(source_file: str) -> str:
    return source_file.split("_table_")[0]


def _header_text_from_table(path: Path) -> str:
    try:
        raw = pd.read_excel(path, header=None, engine="openpyxl", nrows=4)
    except Exception:
        return ""
    cells = [_clean(v) for v in raw.to_numpy().flatten()]
    return " ".join(c for c in cells if c)


def _find_header_source(source_file: str, tables_dir: Path) -> tuple[str, str]:
    source_file = _clean(source_file)
    if not source_file:
        return "", ""
    candidates = []
    exact = tables_dir / source_file
    if exact.exists():
        candidates.append(exact)
    stem = _source_stem(source_file)
    table_1 = tables_dir / f"{stem}_table_1.xlsx"
    if table_1.exists() and table_1 not in candidates:
        candidates.append(table_1)
    for path in candidates:
        text = _header_text_from_table(path)
        if text and re.search(
            r"(dosage|dose|(?<![A-Za-z])a\.?\s*i\.?(?![A-Za-z])|formulation|dilution|waiting)",
            text,
            flags=re.I,
        ):
            return text, path.name
    return "", ""


def _extract_parenthetical_unit(text: str, *keywords: str) -> str:
    for key in keywords:
        if key.lower().replace(" ", "") in {"a.i", "a.i."}:
            key_pattern = r"(?<![A-Za-z])a\.?\s*i\.?(?![A-Za-z])"
        else:
            key_pattern = re.escape(key)
        max_gap = 120 if key.lower() in {"waiting", "period"} else 45
        match = re.search(
            rf"{key_pattern}[^()]{{0,{max_gap}}}\(([^)]{{1,40}})\)",
            text,
            flags=re.I,
        )
        if match:
            return _clean(match.group(1))
    return ""


def _unit_profile(source_file: str, tables_dir: Path, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    source_file = _clean(source_file)
    if source_file in cache:
        return cache[source_file]
    header_text, header_source = _find_header_source(source_file, tables_dir)
    low = header_text.lower()
    dosage_per_ha = bool(re.search(r"(dosage|dose)\s*/?\s*ha|/ha", low))
    ai_unit = _extract_parenthetical_unit(header_text, "a.i", "a. i")
    formulation_unit = _extract_parenthetical_unit(header_text, "formulation")
    dilution_unit = _extract_parenthetical_unit(header_text, "dilution", "water")
    waiting_unit = _extract_parenthetical_unit(header_text, "waiting", "period")

    def suffix_per_ha(unit: str) -> str:
        if not unit:
            return ""
        unit_low = unit.lower()
        if dosage_per_ha and "/ha" not in unit_low and "per ha" not in unit_low:
            return f"{unit}/ha"
        return unit

    profile = {
        "ai_unit": suffix_per_ha(ai_unit),
        "formulation_unit": suffix_per_ha(formulation_unit),
        "dilution_unit": suffix_per_ha(dilution_unit),
        "waiting_period_unit": waiting_unit,
        "unit_source": header_source,
    }
    cache[source_file] = profile
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", default="data/processed/pesticide_recos_usable.xlsx")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument(
        "--tables_dir",
        default="data/processed/pdf_tables",
        help="Directory containing extracted table files used to recover header-level units.",
    )
    args = parser.parse_args()

    cfg = load_config()
    init_db(cfg.paths["sqlite_db"])
    path = Path(args.input_xlsx or args.input_csv)
    if not path.exists():
        print("Normalized pesticide table not found.")
        return

    df = _read_table(path)
    if df.empty:
        print("No rows to load.")
        return
    if "quality_status" in df.columns:
        df = df[df["quality_status"].isin(["valid", "usable"])].copy()
    elif "validation_status" in df.columns:
        df = df[df["validation_status"].isin(["valid", "usable", "pass"])].copy()
    df = df[df["crop_name"].notna() & (df["crop_name"].astype(str).str.strip() != "")]
    tables_dir = Path(args.tables_dir)
    unit_cache: dict[str, dict[str, str]] = {}
    for unit_col in ["ai_unit", "formulation_unit", "dilution_unit", "waiting_period_unit", "unit_source"]:
        if unit_col not in df.columns:
            df[unit_col] = None
        else:
            # Unit metadata must be derived from the source table header during load.
            # Do not trust stale values that may have been produced by an earlier parser.
            df[unit_col] = None
    if "source_file" in df.columns and tables_dir.exists():
        for idx, source_file in df["source_file"].items():
            profile = _unit_profile(source_file, tables_dir, unit_cache)
            for col, value in profile.items():
                if value:
                    df.at[idx, col] = value
    db_columns = [
        "crop_name",
        "disease_name_en",
        "disease_name_hi",
        "pesticide_name",
        "ai_g",
        "formulation",
        "dilution",
        "dose_text",
        "waiting_period_days",
        "ai_unit",
        "formulation_unit",
        "dilution_unit",
        "waiting_period_unit",
        "unit_source",
        "source_file",
        "quality_status",
        "quality_flags",
    ]
    for col in db_columns:
        if col not in df.columns:
            df[col] = None
    df = df[db_columns]
    if df.empty:
        print("No valid rows to load.")
        return

    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS pesticide_recommendations")
        conn.commit()
        init_db(cfg.paths["sqlite_db"])
        df.to_sql("pesticide_recommendations", conn, if_exists="append", index=False)
        print(f"Loaded {len(df)} rows into pesticide_recommendations.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
