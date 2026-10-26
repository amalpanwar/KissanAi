"""Alias-resolved PDF sections, independent of embedding index recall.

Only source selection is deterministic. Sarvam still writes/translates the answer;
no agricultural advice is supplied by this module.
"""
from functools import lru_cache
from pathlib import Path
import re

from app.agent_system import AgentResult
from app.chunking import split_text
from app.commodity_lookup import load_aliases, normalize, resolve_commodities
from app.pdf_extract import read_pdf_pages

GUIDE = Path(__file__).resolve().parents[1] / 'data/raw/Crop Production guide.pdf'
MAX_SECTION_CHARACTERS = 160000


def overview(payload):
    question = payload.get('original_question') or payload['question']
    return bool(re.search(r'how.*(?:grow|cultivat|farm)|(?:की खेती|ki kheti).*(?:कैसे|kaise)|(?:कैसे|kaise).*(?:खेती|kheti)', question, re.I))


@lru_cache(maxsize=2)
def catalog(path, modified, size):
    pages = read_pdf_pages(Path(path), prefer_docling=False)
    entries = {}
    # Read contents pages themselves; no crop/page-offset map in Python.
    for text in pages[:12]:
        if not re.search(r'CONTENTS|S\.No\.', text):
            continue
        for line in text.splitlines():
            match = re.match(r'^\s*(?:(?:\d+\.?|\([ivx]+\))\s+)?([A-Za-z][A-Za-z ()/–-]+?)\s+(\d{1,3})\s*$', line)
            if match:
                entries[match[1].strip()] = int(match[2])
    printed = {}
    for index, text in enumerate(pages):
        match = re.match(r'^\s*(\d{1,3})(?:\s|$)', text)
        if match:
            printed.setdefault(int(match[1]), index)
    sections = {}
    for name, start in entries.items():
        # A container and its first child share a printed page. Prefer the
        # concrete child, rather than treating a category as a single crop.
        siblings = [n for n, page in entries.items() if page == start]
        if len(siblings) > 1 and name != siblings[-1]:
            continue
        following = sorted({p for p in entries.values() if p > start})
        if start not in printed or not following or following[0] not in printed:
            continue
        first, last = printed[start], printed[following[0]] - 1
        if first <= last:
            sections[name] = (first, last)
    return pages, sections


def _forms(name):
    # PDF headings can also contain a local or botanical name.
    forms = [name, re.split(r'\s*[–-]\s*|\(', name)[0]]
    forms.extend(re.findall(r'\(([^)]+)\)', name))
    return {normalize(form).replace(' ', '') for form in forms if normalize(form)}


def resolve_section(question, sections, aliases):
    matched = resolve_commodities(question, names=sections, aliases=aliases)
    found = set()
    for commodity in matched:
        terms = [commodity]
        for key, values in aliases.items():
            if normalize(key) == normalize(commodity):
                terms.extend([key, *values])
        forms = {_form for term in terms for _form in _forms(term)}
        found.update(name for name in sections if forms & _forms(name))
    return next(iter(found)) if len(found) == 1 else None


def lookup(advisor, payload):
    if not GUIDE.exists():
        return None
    stat = GUIDE.stat()
    try:
        pages, sections = catalog(str(GUIDE), stat.st_mtime_ns, stat.st_size)
        aliases = load_aliases()
        original = payload.get('original_question') or payload.get('display_question') or payload['question']
        section = resolve_section(original, sections, aliases)
        if not section:
            # A translated crop term must not override a recognized original.
            explicit = resolve_commodities(original, names=sections, aliases=aliases)
            if not explicit:
                section = resolve_section(payload['question'], sections, aliases)
                if not section:
                    crop = advisor._extract_preferred_crop_from_context(payload.get('context', ''))
                    section = resolve_section(crop or '', sections, aliases)
        if not section:
            return None
        first, last = sections[section]
        total = sum(len(pages[i]) for i in range(first, last + 1))
        scope = {'section': section, 'pdf_pages': [first + 1, last + 1],
                 'mode': 'overview' if overview(payload) else 'focused',
                 'characters': total}
        if total > MAX_SECTION_CHARACTERS:
            return AgentResult('', 'unavailable', evidence={'records': [], 'guide_scope': scope,
                'method': 'production_guide_section', 'reason': 'guide_section_too_large'})
        query = (payload.get('query_variants') or {}).get('en-IN', payload['question'])
        records = []
        for i in range(first, last + 1):
            for text in split_text(pages[i], chunk_size=1400, overlap=180):
                records.append({'text': text, 'source_file': str(GUIDE), 'page': i + 1,
                    'language': 'en-IN', 'search_query': query, 'guide_section': section,
                    'guide_overview': scope['mode'] == 'overview'})
        return AgentResult('', 'ok' if records else 'unavailable', evidence={
            'records': records, 'guide_scope': scope, 'method': 'production_guide_section'})
    except Exception as exc:
        return AgentResult('', 'unavailable', evidence={'records': [],
            'method': 'production_guide_section', 'reason': 'guide_section_unavailable',
            'error_type': type(exc).__name__})


def with_overview_coverage(ranked, candidates):
    """Retain reranking plus full crop-section coverage for broad guides."""
    if not ranked or not candidates or not all(r.get('guide_overview') for r in candidates):
        return ranked
    from app.evidence_reranker import bibliography
    selected = list(ranked)
    seen = {(row['source_file'], row['text']) for row in selected}
    for row in candidates:
        key = (row['source_file'], row['text'])
        if key not in seen and not bibliography(row['text']):
            selected.append(row)
            seen.add(key)
    return selected
