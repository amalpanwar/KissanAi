from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


GLOSSARY_PATH = Path("data/processed/agri_glossary.json")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@lru_cache(maxsize=1)
def load_agri_glossary() -> dict[str, dict[str, object]]:
    if not GLOSSARY_PATH.exists():
        return {}
    try:
        payload = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    terms = payload.get("terms") or {}
    out: dict[str, dict[str, object]] = {}
    for key, value in terms.items():
        if isinstance(value, dict):
            out[_normalize(key)] = value
    return out


def match_glossary_entry(question: str) -> dict[str, object] | None:
    q = _normalize(question)
    if not q:
        return None
    glossary = load_agri_glossary()
    if not glossary:
        return None
    cues = (
        "kya hota hai",
        "क्या होता है",
        "kya hai",
        "क्या है",
        "matlab",
        "मतलब",
        "meaning",
        "full form",
        "का फुल फॉर्म",
    )
    asks_for_meaning = any(cue in q for cue in cues)
    matched: tuple[int, dict[str, object]] | None = None
    for term_key, entry in glossary.items():
        aliases = [_normalize(a) for a in (entry.get("aliases") or []) if str(a).strip()]
        aliases.append(term_key)
        for alias in aliases:
            if not alias:
                continue
            if not asks_for_meaning and q != alias:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", q):
                score = len(alias)
                if matched is None or score > matched[0]:
                    matched = (score, entry)
    if matched:
        return matched[1]
    if asks_for_meaning:
        for term_key, entry in glossary.items():
            label = _normalize(str(entry.get("label") or term_key))
            if label and label == q:
                return entry
    return None


def format_glossary_answer(entry: dict[str, object]) -> str:
    label = str(entry.get("label") or "").strip()
    meaning = str(entry.get("meaning") or "").strip()
    answer_hi = str(entry.get("answer_hi") or "").strip()
    lines: list[str] = []
    if answer_hi:
        lines.append(answer_hi)
    elif label and meaning:
        lines.append(f"{label} का मतलब: {meaning}")
    snippet = ""
    for item in (entry.get("snippets") or []):
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            snippet = str(item.get("text")).strip()
            break
    if snippet:
        lines.append(f"संदर्भ: {snippet}")
    return "\n".join(line for line in lines if line).strip()


def glossary_references(entry: dict[str, object]) -> list[str]:
    refs: list[str] = []
    for item in (entry.get("snippets") or []):
        if isinstance(item, dict):
            src = str(item.get("source") or "").strip()
            if src and src not in refs:
                refs.append(src)
    return refs
