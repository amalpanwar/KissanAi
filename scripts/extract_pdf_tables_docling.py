from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


def _load_docling():
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover - dependency is optional
        print(
            "Docling is not installed in the current environment.\n"
            "Install it separately for offline pesticide-table extraction:\n"
            "  pip install -r requirements-docling.txt\n"
            f"Original import error: {exc}"
        )
        raise SystemExit(1)
    return DocumentConverter


def _table_shape(df: pd.DataFrame) -> list[int]:
    return [int(df.shape[0]), int(df.shape[1])]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return df
    df = df.map(
        lambda value: ""
        if pd.isna(value)
        else " ".join(str(value).replace("\xa0", " ").replace("\n", " ").split())
    )
    return df


def extract_tables_from_pdf(pdf_path: Path, save_doc_markdown: bool = True) -> tuple[list[pd.DataFrame], dict]:
    DocumentConverter = _load_docling()
    converter = DocumentConverter()
    started = time.time()
    conv_res = converter.convert(pdf_path)
    elapsed = round(time.time() - started, 2)

    tables: list[pd.DataFrame] = []
    table_meta: list[dict] = []
    for idx, table in enumerate(conv_res.document.tables, start=1):
        try:
            df = table.export_to_dataframe(doc=conv_res.document)
        except Exception:
            continue
        df = _clean_df(df)
        if df.empty or df.shape[0] < 2:
            continue
        tables.append(df)
        table_meta.append(
            {
                "table_index": idx,
                "shape": _table_shape(df),
            }
        )

    meta = {
        "source_pdf": str(pdf_path),
        "table_count": len(tables),
        "elapsed_sec": elapsed,
        "doc_markdown": conv_res.document.export_to_markdown() if save_doc_markdown else "",
        "tables": table_meta,
    }
    return tables, meta


def _write_table_outputs(
    pdf_stem: str,
    tables: list[pd.DataFrame],
    meta: dict,
    out_dir: Path,
    save_xlsx: bool,
    save_csv: bool,
    save_html: bool,
    save_doc_markdown: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, df in enumerate(tables, start=1):
        base = out_dir / f"{pdf_stem}_table_{idx}"
        if save_csv:
            df.to_csv(base.with_suffix(".csv"), index=False, header=False)
        if save_xlsx:
            df.to_excel(base.with_suffix(".xlsx"), index=False, header=False)
        if save_html:
            df.to_html(base.with_suffix(".html"), index=False, header=False)

    meta_path = out_dir / f"{pdf_stem}_docling_meta.json"
    meta_to_save = {k: v for k, v in meta.items() if k != "doc_markdown"}
    meta_path.write_text(json.dumps(meta_to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_doc_markdown and meta.get("doc_markdown"):
        (out_dir / f"{pdf_stem}.md").write_text(str(meta["doc_markdown"]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MUP PDF tables with Docling and save clean XLSX/CSV/HTML outputs."
    )
    parser.add_argument("--input_dir", default="data/raw/all_sources")
    parser.add_argument("--output_dir", default="data/processed/pdf_tables_docling")
    parser.add_argument("--match", default="mup")
    parser.add_argument("--xlsx", action="store_true", default=True)
    parser.add_argument("--no_xlsx", action="store_false", dest="xlsx")
    parser.add_argument("--csv", action="store_true", default=True)
    parser.add_argument("--no_csv", action="store_false", dest="csv")
    parser.add_argument("--html", action="store_true", default=False)
    parser.add_argument("--doc_markdown", action="store_true", default=True)
    parser.add_argument("--no_doc_markdown", action="store_false", dest="doc_markdown")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    pdfs = sorted([p for p in in_dir.glob("*.pdf") if args.match.lower() in p.name.lower()])
    if not pdfs:
        print("No matching PDFs found.")
        return

    done = 0
    for pdf_path in pdfs:
        try:
            tables, meta = extract_tables_from_pdf(pdf_path, save_doc_markdown=args.doc_markdown)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"{pdf_path.name}: Docling extraction failed: {exc}")
            continue
        if not tables:
            print(f"{pdf_path.name}: no usable tables extracted.")
            continue
        _write_table_outputs(
            pdf_stem=pdf_path.stem,
            tables=tables,
            meta=meta,
            out_dir=out_dir,
            save_xlsx=args.xlsx,
            save_csv=args.csv,
            save_html=args.html,
            save_doc_markdown=args.doc_markdown,
        )
        done += 1
        print(f"{pdf_path.name}: extracted {len(tables)} tables with Docling.")

    print(f"Docling extraction complete: {done} PDFs processed.")


if __name__ == "__main__":
    main()
