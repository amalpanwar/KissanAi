from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_document
from app.config import load_config
from app.db import init_db, insert_research_document


DEFAULT_FILES = [
    "pesticide_recos_usable_with_autofill.xlsx",
    "pesticide_recos_reject_autofilled_still_reject.xlsx",
    "Pesticides1.xlsx",
    "Pesticides3.xlsx",
    "Pesticdes 4.xlsx",
    "Insecticides2.xlsx",
    "insecticides1.xlsx",
]
SKIP_PREFIXES = ("~$",)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").replace("\n", " ")
    return " ".join(text.split())


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in columns}
    for wanted in candidates:
        wanted_norm = "".join(ch for ch in wanted.lower() if ch.isalnum())
        for col, col_norm in normalized.items():
            if wanted_norm and wanted_norm in col_norm:
                return col
    return None


def _source_files(input_dir: Path, explicit_files: list[str] | None) -> list[Path]:
    if explicit_files:
        return [Path(p) if Path(p).is_absolute() else input_dir / p for p in explicit_files]
    paths = []
    for name in DEFAULT_FILES:
        path = input_dir / name
        if path.exists():
            paths.append(path)
    return [p for p in paths if p.exists() and not p.name.startswith(SKIP_PREFIXES)]


def _row_to_text(row: pd.Series, columns: list[str], source_name: str) -> str:
    crop_col = _find_col(columns, ["crop_name", "crop", "nameofcommodity", "nameof crop", "use"])
    pest_col = _find_col(columns, ["disease_name_en", "commonnameof thepest", "targetpest", "nameof insect", "pest", "weed species"])
    pesticide_col = _find_col(columns, ["pesticide_name", "pesticide", "pesticides"])
    ai_col = _find_col(columns, ["ai_g", "a.i.", "a.i", "ai", "dose"])
    formulation_col = _find_col(columns, ["formulation"])
    dilution_col = _find_col(columns, ["dilution", "water", "surface", "exposureperiod"])
    dose_text_col = _find_col(columns, ["dose_text", "dosetext"])
    waiting_col = _find_col(columns, ["waiting_period_days", "waiting period", "waiting_period", "aeration waiting period"])
    source_col = _find_col(columns, ["source_file"])
    status_col = _find_col(columns, ["quality_status", "validation_status"])

    fields = {
        "फसल/उपयोग": _clean(row.get(crop_col)) if crop_col else "",
        "रोग/कीट/खरपतवार": _clean(row.get(pest_col)) if pest_col else "",
        "दवा/पेस्टीसाइड": _clean(row.get(pesticide_col)) if pesticide_col else "",
        "a.i./dose": _clean(row.get(ai_col)) if ai_col else "",
        "formulation": _clean(row.get(formulation_col)) if formulation_col else "",
        "dilution/water": _clean(row.get(dilution_col)) if dilution_col else "",
        "dose_text": _clean(row.get(dose_text_col)) if dose_text_col else "",
        "waiting_period/PHI": _clean(row.get(waiting_col)) if waiting_col else "",
        "quality_status": _clean(row.get(status_col)) if status_col else "",
        "source_table": _clean(row.get(source_col)) if source_col else source_name,
    }
    if not fields["दवा/पेस्टीसाइड"] and not fields["फसल/उपयोग"] and not fields["रोग/कीट/खरपतवार"]:
        return ""

    parts = ["Structured pesticide recommendation from official MUP table."]
    for key, value in fields.items():
        if value:
            parts.append(f"{key}: {value}")
    parts.append(
        "Hindi answer hint: किसान को फसल, रोग/कीट, दवा, डोज, पानी में dilution और waiting period साफ बताएं. "
        "दवा का उपयोग लेबल और स्थानीय कृषि अधिकारी की सलाह के अनुसार ही करें."
    )
    return "\n".join(parts)


def _delete_existing_structured_docs(db_path: str, source_files: list[Path]) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM research_documents WHERE domain = ?", ("pesticide_structured",))
        for path in source_files:
            cur.execute("DELETE FROM research_documents WHERE source_file = ?", (str(path),))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/processed")
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific XLSX file to ingest. Can be passed multiple times. Defaults to final structured pesticide workbooks.",
    )
    parser.add_argument("--domain", default="pesticide_structured")
    parser.add_argument("--district", default="all_india")
    parser.add_argument("--no_delete_existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    input_dir = Path(args.input_dir)
    files = _source_files(input_dir, args.file)
    if not files:
        print("No structured pesticide XLSX files found.")
        return

    if not args.no_delete_existing:
        _delete_existing_structured_docs(cfg.paths["sqlite_db"], files)

    rows_inserted = 0
    files_done = 0
    for path in files:
        if not path.exists() or path.name.startswith(SKIP_PREFIXES):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df = df.dropna(how="all")
        columns = [str(c) for c in df.columns]
        docs = []
        for _, row in df.iterrows():
            text = _row_to_text(row, columns, path.name)
            if text:
                docs.append(text)
        if not docs:
            continue
        text = "\n\n---\n\n".join(docs)
        chunks = chunk_document(
            source_file=str(path),
            text=text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
        )
        for chunk in chunks:
            insert_research_document(
                cfg.paths["sqlite_db"],
                {
                    "source_file": chunk.source_file,
                    "title": path.stem,
                    "publication_year": None,
                    "domain": args.domain,
                    "district": args.district,
                    "text_content": chunk.text,
                },
            )
        rows_inserted += len(chunks)
        files_done += 1
        print(f"Ingested {len(docs)} structured rows from {path.name} as {len(chunks)} chunks.")

    print(f"Structured pesticide ingest complete: {files_done} files, {rows_inserted} chunks.")


if __name__ == "__main__":
    main()
