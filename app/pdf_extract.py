from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


CACHE_DIR = Path("data/processed/pdf_text_cache")
PDF_CACHE_VERSION = "v3"


def _clean_text(text: str) -> str:
    out = str(text or "").replace("\xa0", " ")
    out = re.sub(r"\s+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _load_pypdf_reader():
    from pypdf import PdfReader

    return PdfReader


@lru_cache(maxsize=1)
def _docling_converter():
    try:
        from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except Exception:
        return None
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.force_backend_text = True
        pipeline_options.do_table_structure = False
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
    except Exception:
        return None


def _cache_file(pdf_path: Path) -> Path:
    stat = pdf_path.stat()
    docling_state = "docling" if _docling_converter() is not None else "pypdf"
    cache_key = hashlib.sha1(
        f"{PDF_CACHE_VERSION}|{docling_state}|{pdf_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"{pdf_path.stem}_{cache_key}.json"


def _load_cache(pdf_path: Path) -> dict[str, Any] | None:
    cache_path = _cache_file(pdf_path)
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(pdf_path: Path, payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(pdf_path)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _page_text_from_doc(doc: Any, page_no: int) -> str:
    for method_name in ("export_to_markdown", "export_to_text", "export_to_html"):
        method = getattr(doc, method_name, None)
        if not callable(method):
            continue
        for kwargs in (
            {"page_no": page_no},
            {"page_number": page_no},
            {"page_no": page_no - 1},
            {"page_number": page_no - 1},
            {"page_numbers": [page_no]},
            {"page_numbers": [page_no - 1]},
        ):
            try:
                value = method(**kwargs)
            except TypeError:
                continue
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return _clean_text(value)
    return ""


def _page_text_from_obj(page_obj: Any) -> str:
    for attr in ("text", "content", "markdown"):
        value = getattr(page_obj, attr, None)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    for method_name in ("export_to_markdown", "export_to_text", "export_to_html"):
        method = getattr(page_obj, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _extract_docling_pages(doc: Any) -> list[str]:
    pages_attr = getattr(doc, "pages", None)
    if pages_attr is None:
        return []
    if isinstance(pages_attr, dict):
        page_numbers = sorted(pages_attr.keys())
        page_objs = [pages_attr[k] for k in page_numbers]
    elif isinstance(pages_attr, (list, tuple)):
        page_objs = list(pages_attr)
        page_numbers = list(range(1, len(page_objs) + 1))
    else:
        return []

    pages: list[str] = []
    for idx, page_obj in zip(page_numbers, page_objs):
        text = _page_text_from_obj(page_obj) or _page_text_from_doc(doc, int(idx))
        if text:
            pages.append(text)
    return pages


def _extract_with_docling(pdf_path: Path) -> dict[str, Any] | None:
    converter = _docling_converter()
    if converter is None:
        return None
    try:
        conv_res = converter.convert(pdf_path)
    except Exception:
        return None
    doc = getattr(conv_res, "document", None)
    if doc is None:
        return None

    text = ""
    for method_name in ("export_to_markdown", "export_to_text", "export_to_html"):
        method = getattr(doc, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            text = _clean_text(value)
            break

    pages = _extract_docling_pages(doc)
    if not text and pages:
        text = "\n\n".join(pages)
    if not text:
        return None
    return {"engine": "docling", "text": text, "pages": pages}


def _extract_with_pypdf(pdf_path: Path) -> dict[str, Any]:
    PdfReader = _load_pypdf_reader()
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(_clean_text(text))
    text = "\n\n".join(page for page in pages if page)
    return {"engine": "pypdf", "text": text, "pages": pages}


def extract_pdf(pdf_path: Path | str, prefer_docling: bool = True) -> dict[str, Any]:
    path = Path(pdf_path)
    cached = _load_cache(path)
    if cached:
        return cached

    result: dict[str, Any] | None = None
    if prefer_docling:
        result = _extract_with_docling(path)
    if result is None:
        result = _extract_with_pypdf(path)
    elif not result.get("pages"):
        pypdf_result = _extract_with_pypdf(path)
        result["pages"] = pypdf_result.get("pages", [])
        result["engine"] = "docling+pypdf-pages"

    _save_cache(path, result)
    return result


def read_pdf_text(pdf_path: Path | str, prefer_docling: bool = True) -> str:
    return str(extract_pdf(pdf_path, prefer_docling=prefer_docling).get("text") or "")


def read_pdf_pages(pdf_path: Path | str, prefer_docling: bool = True) -> list[str]:
    pages = extract_pdf(pdf_path, prefer_docling=prefer_docling).get("pages") or []
    return [str(page or "") for page in pages]


def pdf_engine(pdf_path: Path | str) -> str:
    return str(extract_pdf(pdf_path).get("engine") or "")
