"""Prepare reviewed corrections for supervised tuning and held-out evaluation."""
from __future__ import annotations

import hashlib
import re

from app.db import get_training_feedback_examples
from app.feedback import guardrail_flags

# Historical values are unsuitable as timeless model training targets.
DYNAMIC_TOPICS = {"weather", "weather_impact", "price", "market_price", "crop_profitability",
                  "crop_profitability_followup", "crop_choice", "mixed", "news"}


def build_feedback_dataset(db_path):
    splits = {"train": [], "eval": []}
    seen = set()
    for row in get_training_feedback_examples(db_path, limit=10000):
        question = str(row.get("user_query") or "").strip()
        correction = str(row.get("correction_text") or "").strip()
        topic = str(row.get("topic") or "").lower()
        if not question or not correction or topic in DYNAMIC_TOPICS:
            continue
        if guardrail_flags(question, correction):
            continue
        normalized = re.sub(r"\s+", " ", question.casefold())
        if normalized in seen:
            continue
        seen.add(normalized)
        # All duplicates of a question stay in one split, across repeated exports.
        bucket = int(hashlib.sha256(normalized.encode()).hexdigest()[:8], 16) % 5
        split = "eval" if bucket == 0 else "train"
        splits[split].append({"prompt": question, "response": correction})
    return splits
