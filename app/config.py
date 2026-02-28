from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    @property
    def region(self) -> str:
        return self.values["region"]

    @property
    def embedding_model(self) -> str:
        return self.values["embedding_model"]

    @property
    def generator_model(self) -> str:
        return self.values["generator_model"]

    @property
    def chunk_size(self) -> int:
        return int(self.values["chunking"]["chunk_size"])

    @property
    def overlap(self) -> int:
        return int(self.values["chunking"]["overlap"])

    @property
    def top_k(self) -> int:
        return int(self.values["retrieval"]["top_k"])

    @property
    def paths(self) -> dict[str, str]:
        return self.values["paths"]


def load_config(path: str | Path = "configs/pipeline.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
