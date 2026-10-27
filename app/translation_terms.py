"""Technical names that remain English while surrounding prose becomes Hindi.

Recognizing a name for translation is not a pesticide recommendation or approval.
"""
from functools import lru_cache
import json
from contextlib import closing
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / 'data/raw/translation_technical_terms.json'
DATABASE = ROOT / 'data/processed/kisaanai.db'
# Numeric variety identifiers, optionally preceded by an institutional name.
CODE = re.compile(r'\b(?:[A-Z]{2,8}\s+(?:[A-Z][a-z]+\s+){1,3})?[A-Z]{1,6}(?:\s*\([A-Z]{1,4}\))?\s*-?\s*\d+[A-Z]?\b')


@lru_cache(maxsize=4)
def _glossary(path, modified):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return tuple(term for terms in data.values() if isinstance(terms, list)
                     for term in terms if isinstance(term, str) and term.strip())
    except (OSError, ValueError, AttributeError):
        return ()


@lru_cache(maxsize=4)
def _pesticides(path, modified):
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT DISTINCT pesticide_name FROM pesticide_recommendations WHERE pesticide_name IS NOT NULL').fetchall()
    except sqlite3.Error:
        return ()
    names = set()
    for (name,) in rows:
        # Preserve formulation names and their ingredient names independently.
        # Exclude prose-like dirty cells; never whitelist entire instructions.
        for component in str(name).split('+'):
            ingredient = re.split(r'\d', component, maxsplit=1)[0].strip(' .,;/-')
            if (re.fullmatch(r'[A-Za-z][A-Za-z -]{2,79}', ingredient)
                    and len(ingredient.split()) <= 5
                    and not re.search(r'\b(?:apply|spray|use|mix|with|before|after|should|must|the|and|or|for)\b', ingredient, re.I)):
                names.add(ingredient)
    return tuple(sorted(names, key=len, reverse=True))


def technical_terms(result):
    names = list(result.get('protected_terms') or [])
    for path, reader in [(GLOSSARY, _glossary), (DATABASE, _pesticides)]:
        try:
            names.extend(reader(str(path), path.stat().st_mtime_ns))
        except OSError:
            pass
    # Only source-attested variety identifiers are copied from evidence.
    source = '\n'.join(str(row.get('text', '')) for row in result.get('retrieved', []) if isinstance(row, dict))
    names.extend(match.group().strip() for match in CODE.finditer(source))
    original = str(result.get('answer', ''))
    present = {name.strip() for name in names if isinstance(name, str) and name.strip()
               and re.search(r'(?<!\w)' + re.escape(name.strip()) + r'(?!\w)', original, re.I)}
    return sorted(present, key=len, reverse=True)
