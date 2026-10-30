"""Final-answer Hindi translation through the local instruction model."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import json
import os
import re
import sys
import time
import unicodedata
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ENDPOINT = 'https://api.sarvam.ai/translate'  # Used only by multilingual query translation.
MODEL = 'local-instruct'
TRANSLATION_VERSION = 'local-hindi-v1'
ANSWER_TIMEOUT = 60.0
FRAGMENT_TIMEOUT = 30.0
REQUEST_TIMEOUT = 15.0
MAX_CHUNKS = 24
TRANSLATION_WORKERS = 1

_STREAM_UPDATE = ContextVar('kisaanai_translation_stream_update', default=None)


@contextmanager
def translation_stream_callback(callback):
    token = _STREAM_UPDATE.set(callback)
    try:
        yield
    finally:
        _STREAM_UPDATE.reset(token)


def _emit_translation_update(text, values):
    callback = _STREAM_UPDATE.get()
    if not callable(callback):
        return
    visible = text
    for token, value in values.items():
        visible = visible.replace(token, value)
    try:
        callback(visible)
    except Exception:
        # Rendering progress must never affect answer generation.
        pass


class TranslationValidationError(ValueError):
    def __init__(self, reason, *, scope=None, source_kind=None, latin_words=(), structure_error=None):
        super().__init__(reason)
        self.scope = scope
        self.source_kind = source_kind
        self.latin_words = latin_words
        self.structure_error = structure_error
        self.structure_error = structure_error


class TranslatedText(str):
    def __new__(cls, text, native_retry=False, chat_retry=False):
        result = super().__new__(cls, text)
        result.native_retry = native_retry
        result.chat_retry = chat_retry
        return result


_SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_ONES_PATTERN = '|'.join(list(_SMALL)[1:10])
_UNDER_HUNDRED = '(?:' + '|'.join(_TENS) + ')(?:[ -]+(?:' + _ONES_PATTERN + '))?|' + '|'.join(_SMALL)
_WORD_NUMBER = re.compile(r'\b(?:(?:' + _ONES_PATTERN + r')[ -]+hundred(?:[ -]+and)?(?:[ -]+(?:' + _UNDER_HUNDRED + r'))?|(?:' + _UNDER_HUNDRED + r')|half|quarter)\b', re.I)


def _numeric_value(text):
    text = ''.join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text).replace('−', '-')
    try:
        return Decimal(text.replace(',', ''))
    except InvalidOperation:
        return None


def _valid_written_numbers(source, output):
    """Only source-spelled quantities may become unprotected numeric digits."""
    expected = []
    for match in _WORD_NUMBER.finditer(TOKEN.sub('', source)):
        words = match.group().lower().replace('-', ' ').split()
        value = Decimal(0)
        for word in words:
            if word == 'hundred': value *= 100
            elif word == 'half': value += Decimal('0.5')
            elif word == 'quarter': value += Decimal('0.25')
            elif word != 'and': value += _SMALL.get(word, _TENS.get(word, 0))
        expected.append(value)
    cursor = 0
    for number in re.findall(r'[+\-−]?\d+(?:[.,]\d+)*', TOKEN.sub('', output)):
        value = _numeric_value(number)
        while cursor < len(expected) and expected[cursor] != value:
            cursor += 1
        if value is None or cursor == len(expected): return False
        cursor += 1
    return True


def _untranslated_error(text, scope):
    words = tuple(re.findall(r'[A-Za-z]+', TOKEN.sub('', text)))
    return TranslationValidationError('untranslated_text', scope=scope,
                                      source_kind='prose', latin_words=words)


def _has_hindi_letters(text):
    return any('\u0900' <= char <= '\u097f' and unicodedata.category(char).startswith('L') for char in text)


_CONNECTOR_SYMBOLS = {'and': {'&', '+'}, 'or': {'/'}, 'per': {'/'},
                      'at': {'@', ':'}, 'to': {'-', '–', '→'}}


def _validate_hindi_content(source, output, scope):
    """Grammar/math fragments need not each contain Hindi sentence letters."""
    if _has_hindi_letters(output):
        return
    prose = TOKEN.sub('', source)
    if not re.search('[A-Za-z]', prose):
        return  # Protected-only chunks never needed linguistic translation.
    source_words = ' '.join(re.findall('[A-Za-z]+', prose)).lower()
    if scope != 'answer' and source_words in _CONNECTOR_SYMBOLS and output.strip() in _CONNECTOR_SYMBOLS[source_words]:
        return
    number_residue = _WORD_NUMBER.sub('', prose)
    numeric_output = output.strip().strip('()[]{} .,:;।॥')
    if (not re.search('[A-Za-z]', number_residue)
            and re.fullmatch(r'[+\-−]?\d+(?:[.,]\d+)*', numeric_output)
            and _valid_written_numbers(source, output)):
        return
    kind = 'connector' if source_words in _CONNECTOR_SYMBOLS else 'prose'
    raise TranslationValidationError('hindi_missing', scope=scope, source_kind=kind)

# Preserve structured links, code, quantities, units and formulae as indivisible tokens.
FORMULA = r'\b(?:ZnSO\s*4|MnSO\s*4|FeSO\s*4|P2O5|K2O|NPK|NAA|FYM|SSP|DAP|DAS|Zn|S|N|P|K|O)\b'
UNIT = r'(?:kg|g|mg|ml|mL|L|litres?|liters?|tonnes?|tons?|t|cm|mm|m|ha|acres?|hectares?|ppm|%|°C)(?:\s*/\s*(?:ha|hectare|acre|kg|g|L|litre|liter))?'
NUMBER = r'\d+(?:[.,:/–-]\d+)*'
PROTECTED = re.compile(
    r'```[\s\S]*?```|`[^`\n]+`|\[[^\]\n]*\]\([^)]+\)|https?://[^\s<>\)]+|\*\*|__|\||'
    + NUMBER + r'(?:\s*(?:%|' + UNIT + r'(?![A-Za-z])))?(?:\s*' + FORMULA + r')?|'
    + FORMULA
)
# Line breaks are formatting, not translation-request boundaries.
NON_HINDI = re.compile(r'[^\u0900-\u097f]+')
TOKEN = re.compile(r'ZXQ[A-Z]+QXZ')


def _setting(name: str, default: str = '') -> str:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    # Streamlit secrets are read only when Streamlit is already in use.
    st = sys.modules.get('streamlit')
    if st is not None:
        try:
            return str(st.secrets.get(name, default)).strip()
        except Exception:
            pass
    return default


def _token(index: int) -> str:
    label = ''
    while True:
        label = chr(65 + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            return 'ZXQ' + label + 'QXZ'


def _protected_names(result: dict) -> list[str]:
    """Preserve known entities/technical terms, never arbitrary capitalized prose."""
    from app.translation_terms import technical_terms
    names = list(result.get('protected_names') or []) + technical_terms(result)
    keys = ('place', 'sub_district', 'district', 'state', 'city', 'name', 'person_name')
    def add_location(location):
        if isinstance(location, dict):
            names.extend(location.get(k) for k in keys)
    add_location(result.get('location'))
    for message in (result.get('agent_trace') or {}).get('messages', []):
        payload = message.get('payload', {})
        add_location((payload.get('evidence') or {}).get('location'))
        # Context is supplied by the app's location controls. It is not sent.
        context = payload.get('context', '')
        names.extend(re.findall(
            r'(?:^|\|)\s*(?:State|District|Subdistrict|Place|City|Name|नाम|जिला|स्थान)\s*:\s*([^|]+)',
            context, re.I,
        ))
    # Preserve the complete resolved weather location, including geocoder names
    # absent from the local lookup table and legacy weather responses.
    names.extend(re.findall(
        r'(?:मौसम|पूर्वानुमान|बारिश)[^\n()]*\(([^()\n]+)\)',
        str(result.get('answer', '')),
    ))
    return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()},
                  key=len, reverse=True)


def _protect(text: str, names: list[str] | None = None) -> tuple[str, dict[str, str]]:
    values = {}
    def replace(match):
        token = _token(len(values))
        values[token] = match.group()
        return token
    # Names are matched before numbers/formulae; longest names win. Restore the
    # exact matched spelling after translation, including case and spaces.
    entity_pattern = '|'.join(r'(?<!\w)(?i:' + re.escape(n) + r')(?!\w)' for n in (names or []))
    pattern = re.compile(entity_pattern + '|' + PROTECTED.pattern) if entity_pattern else PROTECTED
    return pattern.sub(replace, text), values


def _chunks(text: str, limit: int = 1800) -> list[str]:
    chunks = []
    while len(text) > limit:
        boundary = text.rfind(' ', 0, limit + 1)
        if boundary <= 0:
            raise ValueError('Oversized translation token')
        chunks.append(text[:boundary])
        text = text[boundary + 1:]
    if text:
        chunks.append(text)
    return chunks


@lru_cache(maxsize=1)
def _local_translation_generator():
    """Load the app's configured instruction model once; answers stay on-host."""
    from app.config import load_config
    from app.generator import LocalGenerator
    return LocalGenerator(load_config().generator_model)


def _request_translation(text, api_key='', timeout=REQUEST_TIMEOUT, model=MODEL):
    """Translate one English fragment locally without sending answer text out."""
    prompt = (
        "Translate the agricultural advice below from English into clear, natural Hindi "
        "in Devanagari. Preserve every fact and instruction. Keep technical crop, variety, "
        "fertilizer and pesticide names in English when they are already present. "
        "Keep each ZXQ...QXZ placeholder exactly once and in the same order. "
        "Do not add advice, facts, quantities or commentary. Return only the Hindi translation.\n\n"
        f"{text}"
    )
    output = _local_translation_generator().generate(prompt)
    if not isinstance(output, str) or not output.strip():
        raise TranslationValidationError('empty_response')
    return output.strip()


def _native_prose_retry(text, api_key, deadline):
    parts = re.split('(' + TOKEN.pattern + ')', text)
    tasks = []
    for i, part in enumerate(parts):
        if not TOKEN.fullmatch(part) and re.search('[A-Za-z]', part):
            for chunk in _chunks(part.strip(), limit=900):
                tasks.append((i, chunk))
    def translate(task):
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TranslationValidationError('translation_deadline')
        value = _request_translation(task[1], api_key, min(REQUEST_TIMEOUT, remaining), MODEL)
        if TOKEN.search(value): raise TranslationValidationError('protected_values_changed')
        if not _valid_written_numbers(task[1], value): raise TranslationValidationError('new_numbers')
        if re.search('[A-Za-z]', value): raise _untranslated_error(value, 'native_fragment')
        _validate_hindi_content(task[1], value, 'native_fragment')
        return task[0], value.strip()
    translated = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(tasks)))) as pool:
        for index, value in pool.map(translate, tasks):
            translated.setdefault(index, []).append(value)
    for index, values in translated.items():
        prefix = re.match(r'^\s*', parts[index]).group()
        suffix = re.search(r'\s*$', parts[index]).group()
        parts[index] = prefix + ' '.join(values) + suffix
    output = ''.join(parts)
    if TOKEN.findall(output) != TOKEN.findall(text):
        raise TranslationValidationError('protected_values_changed')
    return output


@lru_cache(maxsize=256)
def _translate(text: str, api_key: str, timeout: float) -> str:
    if not re.search('[A-Za-z]', TOKEN.sub('', text)):
        return TranslatedText(text)
    deadline = time.monotonic() + timeout
    output = _request_translation(text, api_key, min(REQUEST_TIMEOUT, timeout))
    native_retry = False
    chat_retry = False
    # Each protected item must occur exactly once and stay in source order.
    if TOKEN.findall(output) != TOKEN.findall(text):
        if not TOKEN.search(text):
            raise TranslationValidationError('protected_values_changed')
        # Retry only original prose concurrently. Values stay local, and the
        # native model avoids another primary call for every protected gap.
        output = _native_prose_retry(text, api_key, deadline)
        chat_retry |= bool(getattr(output, 'chat_retry', False))
        native_retry = True
    residual = TOKEN.sub('', output)
    if not _valid_written_numbers(text, output):
        raise TranslationValidationError('new_numbers')
    if re.search(r'[A-Za-z]', residual):
        output = _native_prose_retry(text, api_key, deadline)
        chat_retry |= bool(getattr(output, 'chat_retry', False))
        residual = TOKEN.sub('', output)
        native_retry = True
    _validate_hindi_content(text, residual, 'chunk')
    return TranslatedText(output.strip(), native_retry, chat_retry)


def translate_answer(result: dict) -> dict:
    """Translate English answer fragments locally; keep source evidence unchanged.

    No farmer answer text is sent to Sarvam or another translation service.
    Numeric/formula checks do not establish semantic correctness.
    """
    if 'translation' in result:
        return result
    original = str(result.get('answer', ''))
    info = {'provider': 'local', 'model': MODEL, 'target_language': 'hi-IN',
            'version': TRANSLATION_VERSION}
    native_retry_used = False
    chat_retry_used = False
    def finish(answer, status, reason=None):
        metadata = {**info, 'status': status}
        if reason:
            metadata['reason'] = reason
        if native_retry_used:
            metadata['retry_model'] = MODEL
        return {**result, 'answer': answer, 'translation': metadata}
    if _setting('KISAANAI_HINDI_TRANSLATION', '1').lower() in {'0', 'false', 'no'}:
        return finish(original, 'disabled')
    if TOKEN.search(original):
        return finish(original, 'fallback', 'reserved_token_in_source')
    protected, values = _protect(original, _protected_names(result))
    segments = list(NON_HINDI.finditer(protected))
    # Formula-only passages need no model; keep approved scientific notation.
    pending = [m for m in segments if re.search(r'[A-Za-z]', TOKEN.sub('', m.group()))]
    if not pending:
        return finish(original, 'not_needed')
    key = ''  # Kept in the helper signature for cache compatibility; no API key is used.
    deadline = time.monotonic() + ANSWER_TIMEOUT
    replacements = []
    try:
        jobs = []
        fragments = []
        for match in pending:
            segment = match.group()
            prefix = re.match(r'^[\s#>*+\-]*', segment).group()
            suffix = re.search(r'\s*$', segment).group()
            core = segment[len(prefix):len(segment) - len(suffix) if suffix else None]
            chunks = _chunks(core, limit=900)
            fragments.append((match, prefix, suffix, len(chunks)))
            jobs.extend(chunks)
        info['request_chunks'] = len(jobs)
        if len(jobs) > MAX_CHUNKS:
            raise TranslationValidationError('request_limit')

        def translate_chunk(chunk):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranslationValidationError('translation_deadline')
            return _translate(chunk, key, min(FRAGMENT_TIMEOUT, remaining))

        # Keep fragment order visible while streaming each individually
        # validated translation as soon as it is ready.
        events = []
        cursor = 0
        job_index = 0
        for match, prefix, suffix, count in fragments:
            events.append(('text', protected[cursor:match.start()] + prefix))
            for part_index in range(count):
                events.append(('job', job_index))
                job_index += 1
                if part_index + 1 < count:
                    events.append(('text', ' '))
            events.append(('text', suffix))
            cursor = match.end()
        events.append(('text', protected[cursor:]))

        translated = [None] * len(jobs)
        failures = []
        chunk_errors = []
        structure_errors = []
        ready_jobs = set()
        output_parts = []
        next_event = 0

        def flush_ready():
            nonlocal next_event
            changed = False
            while next_event < len(events):
                kind, value = events[next_event]
                if kind == 'job' and value not in ready_jobs:
                    break
                output_parts.append(translated[value] if kind == 'job' else value)
                changed = True
                next_event += 1
            if changed:
                _emit_translation_update(''.join(output_parts), values)

        future_indices = {}
        with ThreadPoolExecutor(max_workers=min(TRANSLATION_WORKERS, len(jobs))) as pool:
            for index, chunk in enumerate(jobs):
                future_indices[pool.submit(translate_chunk, chunk)] = index
            for future in as_completed(future_indices):
                index = future_indices[future]
                try:
                    value = future.result()
                    translated[index] = str(value)
                    native_retry_used |= bool(getattr(value, 'native_retry', False))
                    chat_retry_used |= bool(getattr(value, 'chat_retry', False))
                except Exception as exc:
                    # Keep only this original source fragment; neighboring
                    # validated Hindi fragments remain useful to the farmer.
                    translated[index] = jobs[index]
                    reason = (f'http_{exc.code}' if isinstance(exc, HTTPError) else
                              str(exc) if isinstance(exc, TranslationValidationError) else
                              type(exc).__name__)
                    failures.append(reason)
                    chunk_errors.append(exc)
                    if isinstance(exc, TranslationValidationError) and exc.structure_error:
                        structure_errors.append(exc.structure_error)
                ready_jobs.add(index)
                flush_ready()

        output = ''.join(output_parts)
        if Counter(TOKEN.findall(output)) != Counter(values.keys()):
            raise TranslationValidationError('protected_values_lost')
        if not failures:
            _validate_hindi_content(protected, TOKEN.sub('', output), 'answer')
        for token, value in values.items():
            output = output.replace(token, value)
        _emit_translation_update(output, {})
        if failures:
            from collections import Counter as _ReasonCounter
            first_error = chunk_errors[0]
            if isinstance(first_error, TranslationValidationError) and first_error.scope:
                info['failure_scope'] = first_error.scope
                info['source_fragment_kind'] = first_error.source_kind
            if isinstance(first_error, TranslationValidationError) and first_error.latin_words:
                from app.production_guide_evidence import GUIDE
                public_source = ' '.join(str(row.get('text', '')) for row in result.get('retrieved', [])
                                         if isinstance(row, dict) and row.get('source_file') == str(GUIDE))
                vocabulary = {word.casefold() for word in re.findall(r'[A-Za-z]+', public_source)}
                info['remaining_latin_count'] = len(first_error.latin_words)
                info['remaining_latin_words'] = list(dict.fromkeys(word for word in first_error.latin_words
                    if 0 < len(word) <= 30 and word.casefold() in vocabulary))[:12]
            info['translated_chunks'] = len(jobs) - len(failures)
            info['failed_chunks'] = len(failures)
            info['failure_reason_counts'] = dict(_ReasonCounter(failures))
            if structure_errors:
                info['structure_error_counts'] = dict(_ReasonCounter(structure_errors))
                info['failure_scope'] = 'structured_fragment'
                info['source_fragment_kind'] = 'prose'
                info['retry_model'] = MODEL
            if len(failures) == len(jobs):
                return finish(output, 'fallback', failures[0])
            return finish(output, 'partial', 'partial_translation')
        return finish(output, 'translated')

    except Exception as exc:
        # Never expose provider bodies, API keys, or farmer text in diagnostics.
        if isinstance(exc, TranslationValidationError) and exc.scope:
            info['failure_scope'] = exc.scope
            info['source_fragment_kind'] = exc.source_kind
        if isinstance(exc, TranslationValidationError) and getattr(exc, 'structure_error', None):
            info['structure_error'] = exc.structure_error
            info['retry_model'] = _setting('KISAANAI_SARVAM_CHAT_MODEL', 'sarvam-105b')
        if isinstance(exc, TranslationValidationError) and exc.latin_words:
            from app.production_guide_evidence import GUIDE
            public_source = ' '.join(str(row.get('text', '')) for row in result.get('retrieved', [])
                                     if isinstance(row, dict) and row.get('source_file') == str(GUIDE))
            vocabulary = {word.casefold() for word in re.findall(r'[A-Za-z]+', public_source)}
            info['remaining_latin_count'] = len(exc.latin_words)
            info['remaining_latin_words'] = list(dict.fromkeys(word for word in exc.latin_words
                if 0 < len(word) <= 30 and word.casefold() in vocabulary))[:12]
        reason = (f'http_{exc.code}' if isinstance(exc, HTTPError) else
                  str(exc) if isinstance(exc, TranslationValidationError) else type(exc).__name__)
        return finish(original, 'fallback', reason)
