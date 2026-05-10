from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        online = os.getenv("KISAANAI_HF_ONLINE", "").strip().lower() in {"1", "true", "yes"}
        try:
            self.model = SentenceTransformer(model_name, local_files_only=not online)
        except Exception:
            if not online:
                raise RuntimeError(
                    f"Embedding model '{model_name}' is not available in the local cache. "
                    "Run once with KISAANAI_HF_ONLINE=1 when internet is available."
                )
            raise

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
