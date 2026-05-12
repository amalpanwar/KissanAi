from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        online = os.getenv("KISAANAI_HF_ONLINE", "").strip().lower() in {"1", "true", "yes"}
        try:
            self.model = SentenceTransformer(model_name, local_files_only=not online)
        except Exception as first_exc:
            if not online:
                try:
                    # Streamlit Cloud does not share the local Hugging Face cache
                    # from the developer machine, so allow a one-time download fallback.
                    self.model = SentenceTransformer(model_name, local_files_only=False)
                    return
                except Exception as second_exc:
                    raise RuntimeError(
                        f"Embedding model '{model_name}' is not available in the local cache and could not be "
                        f"downloaded automatically. Local error: {first_exc}. Download error: {second_exc}"
                    ) from second_exc
            raise

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
