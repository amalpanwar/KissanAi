from __future__ import annotations

import argparse
from pathlib import Path

from app.chunking import chunk_document
from app.config import load_config
from app.db import insert_research_document
from app.io_utils import discover_documents, read_text_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()

    cfg = load_config()
    docs = discover_documents(args.input_dir)
    if not docs:
        print("No supported documents found.")
        return

    for p in docs:
        text = read_text_file(Path(p))
        chunks = chunk_document(
            source_file=str(p),
            text=text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
        )
        for ch in chunks:
            insert_research_document(
                cfg.paths["sqlite_db"],
                {
                    "source_file": ch.source_file,
                    "title": Path(p).stem,
                    "publication_year": None,
                    "domain": "agriculture",
                    "district": "western_up",
                    "text_content": ch.text,
                },
            )
    print(f"Ingested {len(docs)} documents into SQLite.")


if __name__ == "__main__":
    main()
