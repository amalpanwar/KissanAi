from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PDF_VECTOR_CACHE_DIR = Path("data/processed/pdf_vector_cache")
PDF_VECTOR_CACHE_VERSION = "v1"


def _cache_prefix(
    pdf_path: Path | str,
    *,
    version: str,
    chunk_size: int,
    overlap: int,
    prefer_docling: bool,
) -> Path:
    path = Path(pdf_path)
    stat = path.stat()
    key = hashlib.sha1(
        (
            f"{PDF_VECTOR_CACHE_VERSION}|{version}|{path.resolve()}|"
            f"{stat.st_mtime_ns}|{stat.st_size}|{chunk_size}|{overlap}|{prefer_docling}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return PDF_VECTOR_CACHE_DIR / f"{path.stem}_{key}"


def load_cached_chunks(
    pdf_path: Path | str,
    *,
    version: str,
    chunk_size: int,
    overlap: int,
    prefer_docling: bool = True,
) -> list[dict[str, Any]] | None:
    path = _cache_prefix(
        pdf_path,
        version=version,
        chunk_size=chunk_size,
        overlap=overlap,
        prefer_docling=prefer_docling,
    ).with_suffix(".chunks.json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append({str(key): value for key, value in item.items()})
    return rows or None


def save_cached_chunks(
    pdf_path: Path | str,
    rows: list[dict[str, Any]],
    *,
    version: str,
    chunk_size: int,
    overlap: int,
    prefer_docling: bool = True,
) -> Path:
    prefix = _cache_prefix(
        pdf_path,
        version=version,
        chunk_size=chunk_size,
        overlap=overlap,
        prefer_docling=prefer_docling,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    chunk_path = prefix.with_suffix(".chunks.json")
    chunk_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return chunk_path


def load_cached_vectors(
    pdf_path: Path | str,
    *,
    version: str,
    chunk_size: int,
    overlap: int,
    prefer_docling: bool = True,
) -> np.ndarray | None:
    path = _cache_prefix(
        pdf_path,
        version=version,
        chunk_size=chunk_size,
        overlap=overlap,
        prefer_docling=prefer_docling,
    ).with_suffix(".vectors.npz")
    if not path.exists():
        return None
    try:
        payload = np.load(path)
        vectors = payload["vectors"]
    except Exception:
        return None
    return np.asarray(vectors, dtype=np.float32)


def save_cached_vectors(
    pdf_path: Path | str,
    vectors: np.ndarray,
    *,
    version: str,
    chunk_size: int,
    overlap: int,
    prefer_docling: bool = True,
) -> Path:
    prefix = _cache_prefix(
        pdf_path,
        version=version,
        chunk_size=chunk_size,
        overlap=overlap,
        prefer_docling=prefer_docling,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    vector_path = prefix.with_suffix(".vectors.npz")
    np.savez_compressed(vector_path, vectors=np.asarray(vectors, dtype=np.float32))
    return vector_path
