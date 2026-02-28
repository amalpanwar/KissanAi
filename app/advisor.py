from __future__ import annotations

from dataclasses import dataclass

from app.embeddings import Embedder
from app.generator import LocalGenerator
from app.prompting import build_prompt
from app.retriever import Retriever
from app.vector_store import NumpyVectorStore


@dataclass
class AdvisorConfig:
    embedding_model: str
    generator_model: str
    index_path: str
    metadata_path: str
    top_k: int


class RAGAdvisor:
    def __init__(self, cfg: AdvisorConfig) -> None:
        self.embedder = Embedder(cfg.embedding_model)
        store = NumpyVectorStore(cfg.index_path, cfg.metadata_path)
        vectors, metadata = store.load()
        self.retriever = Retriever(vectors=vectors, metadata=metadata)
        self.generator = LocalGenerator(cfg.generator_model)
        self.top_k = cfg.top_k

    def answer(self, user_query: str) -> dict:
        qvec = self.embedder.encode([user_query])[0]
        retrieved = self.retriever.retrieve(qvec, k=self.top_k)
        prompt = build_prompt(user_query, retrieved)
        response = self.generator.generate(prompt)
        return {
            "answer": response,
            "references": [r.get("source_file") for r in retrieved],
            "retrieved": retrieved,
        }
