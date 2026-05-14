from __future__ import annotations

import argparse
from pathlib import Path

import pdfplumber
import pandas as pd


def extract_tables_from_pdf(pdf_path: Path) -> list[pd.DataFrame]:
    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            try:
                extracted = page.extract_tables()
            except Exception:
                extracted = []
            for table in extracted or []:
                if not table or len(table) < 2:
                    continue
                cleaned = [
                    [str(cell).strip() if cell is not None else "" for cell in row]
                    for row in table
                ]
                df = pd.DataFrame(cleaned)
                tables.append(df)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/all_sources")
    parser.add_argument("--output_dir", default="data/processed/pdf_tables")
    parser.add_argument("--match", default="mup")
    parser.add_argument("--xlsx", action="store_true", default=True)
    parser.add_argument("--no_xlsx", action="store_false", dest="xlsx")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in in_dir.glob("*.pdf") if args.match.lower() in p.name.lower()])
    if not pdfs:
        print("No matching PDFs found.")
        return

    for pdf in pdfs:
        tables = extract_tables_from_pdf(pdf)
        if not tables:
            print(f"No tables extracted: {pdf.name}")
            continue
        for i, df in enumerate(tables, start=1):
            out_csv = out_dir / f"{pdf.stem}_table_{i}.csv"
            df.to_csv(out_csv, index=False, header=False)
            if args.xlsx:
                out_xlsx = out_dir / f"{pdf.stem}_table_{i}.xlsx"
                df.to_excel(out_xlsx, index=False, header=False)
        print(f"{pdf.name}: {len(tables)} tables extracted.")


if __name__ == "__main__":
    main()
