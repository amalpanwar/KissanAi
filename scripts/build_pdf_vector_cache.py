from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.crop_guide import GUIDE_PDF, _supplemental_pdf_catalog, _supplemental_pdf_chunks, _supplemental_pdf_vectors
from app.pdf_extract import pdf_engine


def _candidate_pdfs(root_dir: Path, include_guide: bool, match: str) -> list[Path]:
    pdfs: list[Path] = []
    if include_guide and GUIDE_PDF.exists():
        pdfs.append(GUIDE_PDF)
    for pdf_path in _supplemental_pdf_catalog():
        pdfs.append(pdf_path)
    unique: list[Path] = []
    seen: set[str] = set()
    needle = match.strip().lower()
    for pdf_path in pdfs:
        try:
            resolved = str(pdf_path.resolve())
        except Exception:
            resolved = str(pdf_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if needle and needle not in pdf_path.name.lower():
            continue
        if root_dir and pdf_path.parent.resolve() != root_dir.resolve():
            continue
        unique.append(pdf_path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Docling-first overlap chunks and vector caches for non-pesticide PDFs."
    )
    parser.add_argument("--root_dir", default="data/raw")
    parser.add_argument("--match", default="")
    parser.add_argument("--skip_guide", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("KISAANAI_ENABLE_DOCLING_RUNTIME", "1")
    root_dir = Path(args.root_dir)
    pdfs = _candidate_pdfs(root_dir, include_guide=not args.skip_guide, match=args.match)
    if not pdfs:
        print("No candidate PDFs found for vector cache build.")
        return

    built = 0
    for pdf_path in pdfs:
        try:
            chunks = _supplemental_pdf_chunks(str(pdf_path))
            vectors = _supplemental_pdf_vectors(str(pdf_path))
        except Exception as exc:
            print(f"{pdf_path.name}: failed to build cache: {exc}")
            continue
        if not chunks or vectors is None:
            print(f"{pdf_path.name}: no usable chunks generated.")
            continue
        engine = pdf_engine(pdf_path)
        print(
            f"{pdf_path.name}: cached {len(chunks)} chunks, vector_shape={tuple(vectors.shape)}, engine={engine or 'unknown'}"
        )
        built += 1

    print(f"PDF vector cache build complete: {built}/{len(pdfs)} PDFs processed.")


if __name__ == "__main__":
    main()
