from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def chunk_document(source_file: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    parts = split_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(chunk_id=f"{source_file}::chunk_{i}", source_file=source_file, text=part)
        for i, part in enumerate(parts)
    ]
