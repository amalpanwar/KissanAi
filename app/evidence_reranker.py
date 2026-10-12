"""Second-stage query/passage relevance, independent of embedding similarity.

Cosine similarity selects candidates, never proves that they answer a question.
The multilingual cross-encoder runs locally and fails closed if unavailable.
"""
from functools import lru_cache
import re
import numpy as np

MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(MODEL, device='cpu', max_length=512, local_files_only=True)
    except OSError:
        return CrossEncoder(MODEL, device='cpu', max_length=512, local_files_only=False)


def bibliography(text):
    # Reference lists are not agronomic advice, even when crop terms occur.
    years = re.findall(r'\b(?:19|20)\d{2}\b', text)
    citations = re.findall(r'\b\d+\s*(?:\(\d+\))?\s*:\s*\d+\s*[–-]\s*\d+', text)
    return len(years) >= 3 and len(citations) >= 2


def verify(payload, records):
    candidates = []
    seen = set()
    for row in records:
        text = str(row.get('text', '')).strip()
        key = re.sub(r'\s+', '', text).casefold()
        if not text or key in seen or bibliography(text):
            continue
        seen.add(key)
        candidates.append(row)
    if not candidates:
        return [], 'ok'
    variants = payload.get('query_variants') or {}
    pairs = [(row.get('search_query') or variants.get(row.get('language'), payload['question']), row['text']) for row in candidates]
    try:
        # Explicit Identity avoids treating sigmoid probabilities as logit scores.
        import torch
        scores = np.asarray(model().predict(pairs, activation_fct=torch.nn.Identity(), batch_size=8)).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError('Invalid relevance scores')
    except Exception:
        return [], 'unavailable'
    accepted = [dict(row, relevance_score=float(score)) for row, score in zip(candidates, scores) if score >= 0.0]
    accepted.sort(key=lambda row: row['relevance_score'], reverse=True)
    return accepted[:3], 'ok'
