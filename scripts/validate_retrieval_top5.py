from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config


def _load_query_specs(path: Path) -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = []
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]

    for file_path in files:
        task_type = file_path.stem
        if not file_path.exists():
            continue
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                specs.append((str(item.get("task_type") or task_type), item))
    return specs


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


def _metric_at_k(gold_ids: set[int], retrieved_ids: list[int], k: int) -> tuple[float, float | None, int]:
    top_ids = retrieved_ids[:k]
    hit_count = sum(1 for doc_id in top_ids if doc_id in gold_ids)
    precision = hit_count / float(k)
    recall = (hit_count / float(len(gold_ids))) if gold_ids else None
    return precision, recall, hit_count


def _empty_metric_bucket() -> dict[str, float]:
    return {"count": 0.0, "precision_sum": 0.0, "recall_sum": 0.0, "recall_count": 0.0}


def _update_metric_bucket(bucket: dict[str, float], precision: float, recall: float | None) -> None:
    bucket["count"] += 1.0
    bucket["precision_sum"] += float(precision)
    if recall is not None:
        bucket["recall_sum"] += float(recall)
        bucket["recall_count"] += 1.0


def _finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, float | None]:
    count = bucket["count"] or 0.0
    recall_count = bucket["recall_count"] or 0.0
    return {
        "query_count": int(count),
        "macro_precision": (bucket["precision_sum"] / count) if count else None,
        "macro_recall": (bucket["recall_sum"] / recall_count) if recall_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/validation/retrieval")
    parser.add_argument("--output", default="data/validation/retrieval_report.json")
    parser.add_argument("--ks", default="5,10", help="Comma-separated retrieval cutoffs, e.g. 5,10")
    args = parser.parse_args()

    ks = sorted({max(1, int(part.strip())) for part in str(args.ks).split(",") if part.strip()})
    max_k = max(ks)

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

    specs = _load_query_specs(Path(args.queries))
    report_rows: list[dict] = []
    overall_metrics: dict[int, dict[str, float]] = {k: _empty_metric_bucket() for k in ks}
    per_task_metrics: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: {k: _empty_metric_bucket() for k in ks}
    )

    for task_type, spec in specs:
        query = str(spec.get("query") or "").strip()
        context = str(spec.get("context") or "").strip()
        if not query:
            continue
        retrieved = advisor.retrieve_debug(query, context, top_k=max_k)
        gold_ids = _gold_doc_ids_from_metadata(advisor, spec)
        retrieved_ids = [int(item.get("_doc_id")) for item in retrieved if item.get("_doc_id") is not None][:max_k]

        row_metrics: dict[str, float | int | None] = {}
        for k in ks:
            precision, recall, hit_count = _metric_at_k(gold_ids, retrieved_ids, k)
            row_metrics[f"hits_at_{k}"] = hit_count
            row_metrics[f"precision_at_{k}"] = precision
            row_metrics[f"recall_at_{k}"] = recall
            _update_metric_bucket(overall_metrics[k], precision, recall)
            _update_metric_bucket(per_task_metrics[task_type][k], precision, recall)

        report_rows.append(
            {
                "task_type": task_type,
                "query": query,
                "context": context,
                "query_plan": advisor.query_agent.decide(query, context).to_dict(),
                "gold_relevant_chunks": len(gold_ids),
                "retrieved_top10": [
                    {
                        "doc_id": int(item.get("_doc_id")),
                        "score": float(item.get("_score", item.get("_vector_score", 0.0))),
                        "source_file": item.get("source_file"),
                        "text_preview": str(item.get("text") or "")[:220],
                        "is_relevant": int(item.get("_doc_id")) in gold_ids,
                    }
                    for item in retrieved[:max_k]
                ],
                **row_metrics,
            }
        )

    summary = {
        "query_count": len(report_rows),
        "ks": ks,
        "overall": {f"@{k}": _finalize_metric_bucket(bucket) for k, bucket in overall_metrics.items()},
        "per_task_type": {
            task_type: {f"@{k}": _finalize_metric_bucket(bucket) for k, bucket in metric_map.items()}
            for task_type, metric_map in sorted(per_task_metrics.items())
        },
        "rows": report_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
