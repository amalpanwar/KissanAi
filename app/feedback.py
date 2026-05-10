from __future__ import annotations

import json
import re
from typing import Any


HIGH_RISK_TOPICS = {
    "pesticide",
    "weather",
    "weather_impact",
    "crop_profitability",
    "crop_profitability_followup",
}


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097f]+", (text or "").lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def guardrail_flags(question: str, correction: str, topic: str | None = None) -> list[str]:
    flags: list[str] = []
    text = f"{question}\n{correction}".strip()
    if re.search(r"\b\d{12}\b", text):
        flags.append("possible_aadhaar")
    if re.search(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", text):
        flags.append("possible_phone")
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE):
        flags.append("possible_email")
    if re.search(r"https?://|www\.", text, flags=re.IGNORECASE):
        flags.append("contains_url")
    if len((correction or "").strip()) > 2500:
        flags.append("too_long")
    if topic in HIGH_RISK_TOPICS and re.search(
        r"\b(?:ml|l/ha|kg/ha|g/l|g/kg|dose|dosage|spray|छिड़काव|दवा|chemical|pesticide|fungicide|insecticide)\b",
        correction,
        flags=re.IGNORECASE,
    ):
        flags.append("high_risk_agri_override")
    return flags


def validate_feedback_with_local_sources(
    advisor: Any,
    question: str,
    answer: str,
    correction: str,
    topic: str | None,
    references: list[str] | None = None,
) -> dict[str, Any]:
    references = references or []
    correction = (correction or "").strip()
    flags = guardrail_flags(question, correction, topic)
    if any(f in flags for f in ("possible_aadhaar", "possible_phone", "possible_email", "contains_url")):
        return {
            "status": "rejected_guardrail",
            "method": "guardrail",
            "notes": "Correction contains sensitive or external contact data.",
            "guardrail_flags": flags,
            "evidence": [],
            "training_eligible": False,
        }

    evidence: list[dict[str, Any]] = []
    score = 0.0
    method = "local_rules"
    notes = "Could not validate strongly against local sources."

    try:
        advisor._ensure_rag_components(load_generator=False)
        if advisor.embedder is not None and advisor.retriever is not None and correction:
            qvec = advisor.embedder.encode([f"{question}\n{correction}"])[0]
            retrieved = advisor.retriever.retrieve(qvec, k=3)
            corr_tokens = set(_normalize_tokens(correction))
            best_overlap = 0.0
            for item in retrieved:
                snippet = str(item.get("text") or "")
                overlap = _jaccard(corr_tokens, set(_normalize_tokens(snippet[:800])))
                best_overlap = max(best_overlap, overlap)
                evidence.append(
                    {
                        "source_file": item.get("source_file"),
                        "overlap": round(overlap, 3),
                        "snippet": snippet[:280],
                    }
                )
            score = best_overlap
            method = "vector_index"
            if best_overlap >= 0.18:
                notes = "Correction has meaningful overlap with retrieved local source text."
            else:
                notes = "Local vector retrieval found only weak support."
    except Exception as exc:
        notes = f"Vector validation unavailable: {exc}"

    status = "needs_review"
    training_eligible = False
    if score >= 0.18 and "high_risk_agri_override" not in flags:
        status = "source_matched"
        training_eligible = True
    elif score >= 0.12:
        status = "needs_review"
    if topic in HIGH_RISK_TOPICS:
        training_eligible = False if status != "source_matched" else training_eligible

    if not correction:
        status = "helpful" if answer else "needs_review"
        method = "rating_only"
        notes = "User marked the answer without a correction."
        evidence = [{"references": references}]
        training_eligible = False

    return {
        "status": status,
        "method": method,
        "notes": notes,
        "guardrail_flags": flags,
        "evidence": evidence,
        "training_eligible": training_eligible,
    }


def compact_evidence_text(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""
    lines = []
    for item in evidence[:3]:
        source = item.get("source_file") or "unknown"
        overlap = item.get("overlap")
        snippet = str(item.get("snippet") or "").replace("\n", " ").strip()
        prefix = f"{source}"
        if overlap is not None:
            prefix += f" (overlap {overlap})"
        lines.append(f"- {prefix}: {snippet[:140]}")
    return "\n".join(lines)
