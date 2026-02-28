from __future__ import annotations

import argparse
import json

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="User query")
    args = parser.parse_args()

    cfg = load_config()
    advisor = RAGAdvisor(
        AdvisorConfig(
            embedding_model=cfg.embedding_model,
            generator_model=cfg.generator_model,
            index_path=cfg.paths["vector_store"],
            metadata_path=cfg.paths["metadata_store"],
            top_k=cfg.top_k,
        )
    )
    out = advisor.answer(args.q)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
