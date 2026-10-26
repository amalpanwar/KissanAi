"""Second-stage query/passage relevance, independent of embedding similarity.

Cosine similarity selects candidates, never proves that they answer a question.
The multilingual cross-encoder ranks locally and fails closed if unavailable.
Its raw logits order passages; they are not calibrated relevance probabilities.
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


def ranking_query(payload, row):
    """Keep search expansion out of the cross-encoder's natural question."""
    variants = payload.get('query_variants') or {}
    search = row.get('search_query') or variants.get(row.get('language'), payload['question'])
    lines = search.splitlines()
    crop = next((line.split(':', 1)[1].strip() for line in lines
                 if line.startswith('Crop context:')), '')
    question = '\n'.join(line for line in lines if not line.startswith(('Crop synonyms:', 'Crop context:'))).strip()
    if crop and crop.casefold() not in question.casefold():
        question += '\nCrop: ' + crop
    return question


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
    pairs = [(ranking_query(payload, row), row['text']) for row in candidates]
    try:
        # Explicit Identity avoids treating sigmoid probabilities as logit scores.
        import torch
        scores = np.asarray(model().predict(pairs, activation_fct=torch.nn.Identity(), batch_size=8)).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError('Invalid relevance scores')
    except Exception:
        return [], 'unavailable'
    # Ranking logits can be negative for useful broad cultivation passages.
    # Do not treat zero as a universal relevance threshold. Synthesis must
    # explicitly abstain when selected evidence does not answer the question.
    accepted = [dict(row, relevance_score=float(score)) for row, score in zip(candidates, scores)]
    accepted.sort(key=lambda row: row['relevance_score'], reverse=True)
    return accepted[:3], 'ok'
