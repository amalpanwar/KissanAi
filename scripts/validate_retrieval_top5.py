from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config


def _load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    if not path.exists():
        return queries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            queries.append(item)
    return queries


def _is_relevant_chunk(chunk: dict, spec: dict) -> bool:
    source = str(chunk.get("source_file") or "").lower()
    text = str(chunk.get("text") or "").lower()
    source_terms = [str(x).lower() for x in spec.get("relevant_source_substrings", [])]
    text_terms = [str(x).lower() for x in spec.get("relevant_text_substrings", [])]

    source_ok = True if not source_terms else any(term in source for term in source_terms)
    text_ok = True if not text_terms else any(term in text for term in text_terms)
    return bool(source_ok and text_ok)
def _gold_doc_ids_from_metadata(advisor: RAGAdvisor, spec: dict) -> set[int]:
    advisor._ensure_rag_components(load_generator=False)
    if advisor.retriever is None:
        return set()
    gold: set[int] = set()
    for idx, row in enumerate(advisor.retriever.metadata.to_dict(orient="records")):
        if _is_relevant_chunk(row, spec):
            gold.add(int(idx))
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/validation/retrieval_top5_queries.jsonl")
    parser.add_argument("--output", default="data/validation/retrieval_top5_report.json")
    args = parser.parse_args()

    cfg = load_config()
    medium_generator_model = os.getenv("KISAANAI_MEDIUM_GENERATOR_MODEL") or cfg.generator_model
    advisor = RAGAdvisor(
        AdvisorConfig(
            embedding_model=cfg.embedding_model,
            generator_model=medium_generator_model,
            index_path=cfg.paths["vector_store"],
            metadata_path=cfg.paths["metadata_store"],
            top_k=cfg.top_k,
            db_path=cfg.paths["sqlite_db"],
            complex_generator_model=os.getenv("KISAANAI_COMPLEX_GENERATOR_MODEL") or None,
        )
    )

    queries = _load_queries(Path(args.queries))
    report_rows: list[dict] = []
    macro_precision = 0.0
    macro_recall = 0.0
    measured = 0

    for spec in queries:
        query = str(spec.get("query") or "").strip()
        context = str(spec.get("context") or "").strip()
        if not query:
            continue
        retrieved = advisor.retrieve_debug(query, context, top_k=5)
        gold_ids = _gold_doc_ids_from_metadata(advisor, spec)
        retrieved_ids = [int(item.get("_doc_id")) for item in retrieved if item.get("_doc_id") is not None][:5]
        hits = [doc_id for doc_id in retrieved_ids if doc_id in gold_ids]
        precision_at_5 = len(hits) / 5.0
        recall_at_5 = (len(hits) / len(gold_ids)) if gold_ids else None
        if recall_at_5 is not None:
            macro_precision += precision_at_5
            macro_recall += recall_at_5
            measured += 1
        report_rows.append(
            {
                "query": query,
                "context": context,
                "query_plan": advisor.query_agent.decide(query, context).to_dict(),
                "gold_relevant_chunks": len(gold_ids),
                "retrieved_top5": [
                    {
                        "doc_id": int(item.get("_doc_id")),
                        "score": float(item.get("_score", item.get("_vector_score", 0.0))),
                        "source_file": item.get("source_file"),
                        "text_preview": str(item.get("text") or "")[:220],
                        "is_relevant": int(item.get("_doc_id")) in gold_ids,
                    }
                    for item in retrieved[:5]
                ],
                "precision_at_5": precision_at_5,
                "recall_at_5": recall_at_5,
            }
        )

    summary = {
        "query_count": len(report_rows),
        "measured_query_count": measured,
        "macro_precision_at_5": (macro_precision / measured) if measured else None,
        "macro_recall_at_5": (macro_recall / measured) if measured else None,
        "rows": report_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
