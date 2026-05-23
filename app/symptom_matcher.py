from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path


SYMPTOM_DICTIONARY_PATH = Path("data/processed/symptom_dictionary.json")
SYMPTOM_CANDIDATE_PATH = Path("data/processed/symptom_alias_candidates.json")
TRAINING_FEEDBACK_PATH = Path("data/processed/accepted_feedback.jsonl")


def normalize_symptom_text(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-z\u0900-\u097f\s]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_aliases(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_symptom_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(str(value).strip())
    return out


@lru_cache(maxsize=8)
def load_symptom_dictionary(dict_mtime_ns: int, candidate_mtime_ns: int) -> dict[str, dict[str, object]]:
    _ = dict_mtime_ns
    _ = candidate_mtime_ns
    if not SYMPTOM_DICTIONARY_PATH.exists():
        return {}
    try:
        payload = json.loads(SYMPTOM_DICTIONARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    entries: dict[str, dict[str, object]] = {}
    for canonical, entry in (payload.get("symptoms") or {}).items():
        if not canonical or not isinstance(entry, dict):
            continue
        merged = dict(entry)
        merged["canonical"] = str(canonical).strip()
        merged["aliases"] = _dedupe_aliases([str(a) for a in (entry.get("aliases") or []) if str(a).strip()])
        entries[str(canonical).strip()] = merged

    if SYMPTOM_CANDIDATE_PATH.exists():
        try:
            candidate_payload = json.loads(SYMPTOM_CANDIDATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            candidate_payload = {}
        for canonical, aliases in (candidate_payload.get("aliases") or {}).items():
            if canonical not in entries or not isinstance(aliases, list):
                continue
            merged_aliases = list(entries[canonical].get("aliases") or [])
            merged_aliases.extend(str(a) for a in aliases if str(a).strip())
            entries[canonical]["aliases"] = _dedupe_aliases(merged_aliases)

    return entries


@lru_cache(maxsize=8)
def load_feedback_rows(
    db_path: str,
    db_mtime_ns: int,
    feedback_path: str,
    feedback_mtime_ns: int,
) -> list[dict[str, object]]:
    _ = db_mtime_ns
    _ = feedback_mtime_ns
    rows: list[dict[str, object]] = []
    db_file = Path(db_path) if db_path else None
    if db_file and db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            fetched = conn.execute(
                """
                SELECT
                    q.user_query,
                    q.topic,
                    q.crop_name,
                    q.answer_text,
                    f.correction_text,
                    f.validation_status,
                    f.is_training_eligible
                FROM answer_feedback f
                JOIN query_logs q ON q.id = f.query_log_id
                WHERE f.validation_status = 'accepted' OR f.is_training_eligible = 1
                ORDER BY f.updated_at DESC, f.created_at DESC
                LIMIT 300
                """
            ).fetchall()
            rows.extend(dict(item) for item in fetched)
            conn.close()
        except Exception:
            pass

    fp = Path(feedback_path)
    if fp.exists():
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        except Exception:
            pass

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            normalize_symptom_text(str(row.get("user_query") or "")),
            normalize_symptom_text(str(row.get("correction_text") or "")),
            normalize_symptom_text(str(row.get("topic") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def save_symptom_alias_candidates(payload: dict[str, object]) -> None:
    SYMPTOM_CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMPTOM_CANDIDATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
