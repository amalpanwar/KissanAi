from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SUPPORTED = {".txt", ".md", ".csv", ".json", ".pdf"}


def _get_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF document ingestion.") from exc
    return PdfReader


def read_text_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt" or ext == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    if ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(obj, ensure_ascii=False)
    if ext == ".pdf":
        reader = _get_pdf_reader()(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages.append(text)
        return "\n".join(pages)
    raise ValueError(f"Unsupported format: {ext}")


def discover_documents(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            paths.append(p)
    return paths
