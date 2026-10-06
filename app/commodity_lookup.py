"""Catalog-driven commodity matching shared by chat and price tools.

Exact token phrases only: a misspelling/unknown commodity asks for clarification
rather than silently selecting a different crop. Hindi combining marks are kept.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIAS_PATH = ROOT / 'data/raw/commodity_aliases.json'
CATALOG_PATH = ROOT / 'data/raw/agmarknet_commodities.csv'


def normalize(value: str) -> str:
    text = unicodedata.normalize('NFKC', str(value)).casefold()
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LNM' else ' ' for c in text).split())


@lru_cache(maxsize=4)
def _read_aliases(path: str, stamp: int) -> dict[str, list[str]]:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: [x for x in v if isinstance(x, str) and x.strip()]
            for k, v in data.items() if isinstance(k, str) and isinstance(v, list)}


def load_aliases():
    try:
        return _read_aliases(str(ALIAS_PATH), ALIAS_PATH.stat().st_mtime_ns)
    except OSError:
        return {}


def catalog_names() -> list[str]:
    import csv
    try:
        with CATALOG_PATH.open(encoding='utf-8-sig', newline='') as stream:
            return [row['commodity_name'] for row in csv.DictReader(stream) if row.get('commodity_name')]
    except (OSError, UnicodeError, csv.Error):
        return []


def resolve_commodities(question: str, names=(), *, aliases=None) -> list[str]:
    """Return all named commodities; longer overlapping phrases win.

    New official catalog/CSV names work without adding crop-specific code.
    Alias translations come from the maintained repository dictionary.
    """
    aliases = load_aliases() if aliases is None else aliases
    labels = {normalize(name): str(name) for name in names if normalize(name)}
    registry = {}
    for name, values in aliases.items():
        key = normalize(name)
        registry[key] = {normalize(v) for v in [name, *values] if normalize(v)}
        labels.setdefault(key, name.title())
    for key in labels:
        registry.setdefault(key, {key})
    query = ' ' + normalize(question) + ' '
    matches = []
    for key, phrases in registry.items():
        for phrase in phrases:
            needle = ' ' + phrase + ' '
            start = query.find(needle)
            while start >= 0:
                matches.append((start, start + len(needle) - 1, key))
                start = query.find(needle, start + 1)
    # For example "green chilli" wins over "chilli" and "pineapple" never
    # matches "apple". Equal-span aliases may intentionally identify variants.
    selected = [(a,b,k) for a,b,k in matches if not any(
        c <= a and b <= d and (d-c) > (b-a) for c,d,_ in matches)]
    return list(dict.fromkeys(labels[k] for a,b,k in sorted(selected)))


def matching_market_names(requested, names, *, aliases=None) -> set[str]:
    aliases = load_aliases() if aliases is None else aliases
    registry = {normalize(k): {normalize(k), *(normalize(v) for v in values)}
                for k, values in aliases.items()}
    allowed = set()
    for name in requested:
        key = normalize(name)
        allowed.update(registry.get(key, {key}))
    return {str(name) for name in names if normalize(name) in allowed}
