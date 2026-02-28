from __future__ import annotations

import sqlite3

from app.config import load_config
from app.embeddings import Embedder
from app.vector_store import NumpyVectorStore


def main() -> None:
    cfg = load_config()
    conn = sqlite3.connect(cfg.paths["sqlite_db"])
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, source_file, text_content FROM research_documents ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        print("No documents found. Run ingest first.")
        return

    texts = [r["text_content"] for r in rows]
    metadata = [
        {"doc_id": r["id"], "source_file": r["source_file"], "text": r["text_content"]}
        for r in rows
    ]

    embedder = Embedder(cfg.embedding_model)
    vectors = embedder.encode(texts)

    store = NumpyVectorStore(
        index_path=cfg.paths["vector_store"],
        metadata_path=cfg.paths["metadata_store"],
    )
    store.save(vectors, metadata)
    print(f"Saved vector index to {cfg.paths['vector_store']}")


if __name__ == "__main__":
    main()
