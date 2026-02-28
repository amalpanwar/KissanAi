from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5) -> list[int]:
    if len(doc_vecs) == 0:
        return []
    scores = doc_vecs @ query_vec
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])].tolist()


class Retriever:
    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame) -> None:
        self.vectors = vectors
        self.metadata = metadata

    def retrieve(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        ids = cosine_top_k(query_vec, self.vectors, k=k)
        rows = self.metadata.iloc[ids]
        return rows.to_dict(orient="records")
