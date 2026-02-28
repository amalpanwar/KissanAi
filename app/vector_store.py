from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class NumpyVectorStore:
    def __init__(self, index_path: str, metadata_path: str) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

    def save(self, vectors: np.ndarray, metadata_rows: list[dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_path, vectors=vectors)
        pd.DataFrame(metadata_rows).to_csv(self.metadata_path, index=False)

    def load(self) -> tuple[np.ndarray, pd.DataFrame]:
        obj = np.load(self.index_path)
        vectors = obj["vectors"]
        metadata = pd.read_csv(self.metadata_path)
        return vectors, metadata
