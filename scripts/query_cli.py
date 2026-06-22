from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.advisor import RAGAdvisor, build_advisor_config
from app.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="User query")
    args = parser.parse_args()

    cfg = load_config()
    medium_generator_model = os.getenv("KISAANAI_MEDIUM_GENERATOR_MODEL") or cfg.generator_model
    advisor = RAGAdvisor(
        build_advisor_config(
            embedding_model=cfg.embedding_model,
            generator_model=medium_generator_model,
            index_path=cfg.paths["vector_store"],
            metadata_path=cfg.paths["metadata_store"],
            top_k=cfg.top_k,
            db_path=cfg.paths["sqlite_db"],
            complex_generator_model=os.getenv("KISAANAI_COMPLEX_GENERATOR_MODEL") or None,
            response_cache_path=os.getenv("KISAANAI_RESPONSE_CACHE_PATH", "data/processed/query_response_cache.json"),
            query_cache_ttl_sec=int(os.getenv("KISAANAI_QUERY_CACHE_TTL_SEC", str(6 * 60 * 60))),
        )
    )
    out = advisor.answer(args.q)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
