"""Versioned bilingual cache for static production-guide answers only."""
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata
import weakref

from app.hindi_translation import _setting
from app.production_guide_evidence import GUIDE

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 'guide-bilingual-v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=64)
def _digest(path, stamp, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    path = Path(path); stat = path.stat()
    return _digest(str(path), stat.st_mtime_ns, stat.st_size)


def translation_signature():
    from app.hindi_translation import TRANSLATION_VERSION
    from app.translation_terms import DATABASE, _pesticides
    try:
        registry = _pesticides(str(DATABASE), DATABASE.stat().st_mtime_ns)
    except OSError:
        registry = ()
    return _hash([TRANSLATION_VERSION,
        [file_digest(ROOT / path) for path in ('app/hindi_translation.py', 'app/translation_terms.py', 'data/raw/translation_technical_terms.json')],
        sorted(registry), _setting('KISAANAI_SARVAM_CHAT_MODEL', 'sarvam-105b'),
        _setting('KISAANAI_HINDI_TRANSLATION', '1')])


def _normal(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def canonical_question(question, section, mode):
    from app.commodity_lookup import load_aliases, normalize
    from app.production_guide_evidence import _forms
    normalized = ' ' + normalize(question) + ' '
    terms = [section]
    for name, values in load_aliases().items():
        if any(_forms(term) & _forms(section) for term in [name, *values]):
            terms.extend([name, *values])
    for term in sorted({normalize(term) for term in terms}, key=len, reverse=True):
        normalized = normalized.replace(' ' + term + ' ', ' crop ')
    generic = {'how to grow crop', 'how to cultivate crop', 'how to do crop farming',
        'how do i grow crop', 'crop ki kheti kaise kare', 'crop ki kheti kaise karen',
        'crop की खेती कैसे करें', 'crop की खेती कैसे करे', 'crop की खेती कैसे करें बताएं'}
    return 'cultivation-overview' if mode == 'overview' and normalized.strip() in generic else _normal(question)


@dataclass
class CacheRequest:
    key: str
    section: str
    context: str
    hindi_signature: str
    scope: dict


def prepare_request(advisor, query):
    if _setting('KISAANAI_GUIDE_RESPONSE_CACHE', '1').lower() in {'0', 'false', 'no'}:
        return None
    import os
    if os.getenv('KISAANAI_AGENTIC', '1').lower() in {'0', 'false', 'no'}:
        return None
    from app.agent_system import Coordinator
    from app.production_guide_evidence import lookup
    context, question = advisor._split_context_and_question(query)
    from app.commodity_lookup import normalize
    temporal = re.search(r'\b(today|tomorrow|yesterday|now|current|latest|this (?:week|month|year|season)|aaj|abhi|kal)\b', question, re.I)
    padded = ' ' + normalize(question) + ' '
    if temporal or any(' ' + word + ' ' in padded for word in ('आज', 'अभी', 'कल', 'इस सप्ताह', 'इस महीने', 'इस साल', 'इस वर्ष', 'इस मौसम')):
        return None
    if set(Coordinator.domains(question)) - {'agronomy', 'research'}:
        return None
    guide = lookup(advisor, {'question': question, 'original_question': question, 'context': context})
    if guide is None or guide.status != 'ok':
        return None
    scope = guide.evidence['guide_scope']
    source_signature = [CACHE_VERSION, file_digest(GUIDE), file_digest(ROOT / 'data/raw/commodity_aliases.json'),
        [file_digest(ROOT / path) for path in ('app/research.py', 'app/production_guide_evidence.py',
            'app/evidence_reranker.py', 'app/sarvam_generation.py', 'app/pdf_extract.py', 'app/chunking.py')],
        _setting('KISAANAI_SARVAM_CHAT_MODEL', 'sarvam-105b')]
    key = _hash([source_signature, scope['section'],
                 canonical_question(question, scope['section'], scope['mode']), _normal(context)])
    return CacheRequest(key, scope['section'], context, translation_signature(), scope)


class GuideResponseCache:
    def __init__(self, path, max_entries=256):
        self.path = Path(path)
        self.max_entries = max(1, int(max_entries))

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=3)
        self.path.chmod(0o600)
        try:
            db.execute('''CREATE TABLE IF NOT EXISTS guide_answers (
                key TEXT PRIMARY KEY, english TEXT NOT NULL, english_digest TEXT NOT NULL,
                hindi TEXT, hindi_signature TEXT, created REAL NOT NULL, accessed REAL NOT NULL)''')
        except sqlite3.Error:
            db.close()
            raise
        return db

    def get(self, key):
        with closing(self._connect()) as db, db:
            row = db.execute('SELECT english, english_digest, hindi, hindi_signature, created FROM guide_answers WHERE key=?', (key,)).fetchone()
            if not row:
                return None
            db.execute('UPDATE guide_answers SET accessed=? WHERE key=?', (time.time(), key))
        english = json.loads(row[0])
        hindi = json.loads(row[2]) if row[2] else None
        if (not isinstance(english, dict) or not isinstance(english.get('answer'), str)
                or not isinstance(english.get('retrieved'), list)):
            raise ValueError('invalid_guide_cache_entry')
        if hindi is not None and (not isinstance(hindi, dict) or not isinstance(hindi.get('answer'), str)
                                  or not isinstance(hindi.get('translation'), dict)):
            raise ValueError('invalid_guide_cache_translation')
        return {'english': english, 'english_digest': row[1], 'hindi': hindi,
                'hindi_signature': row[3], 'created_at': row[4]}

    def put_english(self, key, english):
        digest = _hash(english)
        now = time.time()
        with closing(self._connect()) as db, db:
            # First writer wins; Hindi can only attach to this exact English text.
            db.execute('INSERT OR IGNORE INTO guide_answers(key,english,english_digest,created,accessed) VALUES(?,?,?,?,?)',
                       (key, json.dumps(english, ensure_ascii=False), digest, now, now))
            db.execute('DELETE FROM guide_answers WHERE key IN (SELECT key FROM guide_answers ORDER BY accessed DESC LIMIT -1 OFFSET ?)', (self.max_entries,))
            stored = db.execute('SELECT english_digest FROM guide_answers WHERE key=?', (key,)).fetchone()
        return digest if stored and stored[0] == digest else None

    def put_hindi(self, key, english_digest, hindi, signature):
        with closing(self._connect()) as db, db:
            cursor = db.execute('UPDATE guide_answers SET hindi=?,hindi_signature=?,accessed=? WHERE key=? AND english_digest=?',
                       (json.dumps(hindi, ensure_ascii=False), signature, time.time(), key, english_digest))
            return cursor.rowcount == 1

    def clear(self):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM guide_answers')


_LOCK_GUARD = threading.Lock()
_LOCKS = weakref.WeakValueDictionary()


def _lock(key):
    with _LOCK_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def source_pages(rows):
    grouped = {}
    for row in rows:
        page = row.get('page')
        if isinstance(page, int) and not isinstance(page, bool) and page > 0:
            grouped.setdefault(row['source_file'], set()).add(page)
    return [{'source_file': source, 'pages': sorted(pages)} for source, pages in grouped.items()]


def finalize_response(result):
    from app.hindi_translation import translate_answer
    translated = translate_answer(result)
    if result.get('requires_hindi'):
        translated = {**translated, 'english_answer': result['answer'],
                      'english_agent_status': result.get('agent_status'),
                      'source_pages': source_pages(result.get('retrieved', []))}
        if translated.get('translation', {}).get('status') not in {'translated', 'not_needed'}:
            # Keep the source-grounded answer visible. translate_answer returns
            # validated Hindi chunks and leaves only failed chunks in English.
            translated = {**translated, 'agent_status': 'partial'}
    return translated


def _english_record(result, request):
    if result.get('english_agent_status') not in {'ok', 'partial'} or not result.get('english_answer'):
        return None
    rows = result.get('retrieved') or []
    refs = result.get('references') or []
    if not rows or not refs or any(ref != str(GUIDE) for ref in refs):
        return None  # Never persist answers mixed with news/web/market evidence.
    if any(row.get('source_file') != str(GUIDE) or row.get('guide_section') != request.section for row in rows):
        return None
    return {'answer': result['english_answer'], 'agent_status': result['english_agent_status'],
        'topic': result.get('topic', 'research'), 'crop': result.get('crop'),
        'retrieved': [{key: row[key] for key in ('text', 'page', 'guide_section', 'guide_overview', 'relevance_score') if key in row} for row in rows]}


def _restore(english, context):
    from app.location_selection import location_from_context
    rows = [{**row, 'source_file': str(GUIDE)} for row in english['retrieved']]
    return {'answer': english['answer'], 'agent_status': english['agent_status'],
            'requires_hindi': True, 'topic': english.get('topic', 'research'),
            'crop': english.get('crop'), 'references': [str(GUIDE)],
            'retrieved': rows, 'location': location_from_context(context)}


def _hindi_record(result):
    from app.hindi_translation import _has_hindi_letters
    if (result.get('agent_status') in {'ok', 'partial'}
            and result.get('translation', {}).get('status') in {'translated', 'not_needed'}
            and _has_hindi_letters(result.get('answer', ''))):
        return {'answer': result['answer'], 'translation': result['translation']}
    return None


def _annotate(result, status, request, created=None):
    return {**result, 'cache_hit': status in {'hit', 'english_hit'}, 'guide_cache': {
        'status': status, 'section': request.section, 'pdf_pages': request.scope['pdf_pages'],
        'created_at': created}}


def answer_with_cache(advisor, query, uncached):
    try:
        request = prepare_request(advisor, query)
    except (OSError, ValueError, KeyError, sqlite3.Error):
        request = None
    if request is None:
        return uncached(query)
    cache = GuideResponseCache(_setting('KISAANAI_GUIDE_CACHE_PATH', str(ROOT / 'data/processed/guide_response_cache.sqlite3')))
    with _lock(request.key):
        try:
            entry = cache.get(request.key)
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return _annotate(uncached(query), 'unavailable', request)
        if entry:
            try:
                raw = _restore(entry['english'], request.context)
            except (ValueError, KeyError, TypeError):
                return _annotate(uncached(query), 'unavailable', request)
            if entry['hindi'] and entry['hindi_signature'] == request.hindi_signature:
                result = {**raw, **entry['hindi'], 'english_answer': raw['answer'],
                          'english_agent_status': raw['agent_status'], 'source_pages': source_pages(raw['retrieved'])}
                return _annotate(result, 'hit', request, entry['created_at'])
            result = finalize_response(raw)
            hindi = _hindi_record(result)
            if hindi:
                try:
                    cache.put_hindi(request.key, entry['english_digest'], hindi, request.hindi_signature)
                except (OSError, ValueError, sqlite3.Error):
                    pass
            return _annotate(result, 'english_hit', request, entry['created_at'])
        result = uncached(query)
        english = _english_record(result, request)
        if english:
            try:
                digest = cache.put_english(request.key, english)
                if digest is None:
                    return _annotate(result, 'not_stored', request)
                hindi = _hindi_record(result)
                if hindi:
                    cache.put_hindi(request.key, digest, hindi, request.hindi_signature)
            except (OSError, ValueError, sqlite3.Error):
                return _annotate(result, 'unavailable', request)
        status = ('stored' if _hindi_record(result) else 'english_stored') if english else 'not_stored'
        return _annotate(result, status, request)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Manage the local bilingual crop-guide response cache')
    parser.add_argument('--clear', action='store_true', help='Remove saved English and Hindi guide answers')
    args = parser.parse_args()
    if args.clear:
        GuideResponseCache(_setting('KISAANAI_GUIDE_CACHE_PATH', str(ROOT / 'data/processed/guide_response_cache.sqlite3'))).clear()
        print('Guide response cache cleared.')
