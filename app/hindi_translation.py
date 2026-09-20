"""Final-answer Hindi translation through Sarvam; never changes source evidence."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import os
import re
import sys
import time
from urllib.request import Request, urlopen

ENDPOINT = 'https://api.sarvam.ai/translate'
MODEL = 'sarvam-translate:v1'
# Preserve structured links, code, quantities, units and formulae as indivisible tokens.
FORMULA = r'\b(?:ZnSO\s*4|MnSO\s*4|FeSO\s*4|P2O5|K2O|NPK|NAA|FYM|SSP|DAP|DAS|Zn|S|N|P|K)\b'
UNIT = r'(?:kg|g|mg|ml|mL|L|litres?|liters?|tonnes?|tons?|t|cm|mm|m|ha|acres?|hectares?|ppm|%|°C)(?:\s*/\s*(?:ha|hectare|acre|L|litre|liter))?'
NUMBER = r'\d+(?:[.,:/–-]\d+)*'
PROTECTED = re.compile(
    r'```[\s\S]*?```|`[^`\n]+`|\[[^\]\n]*\]\([^)]+\)|https?://[^\s<>\)]+|\*\*|__|\||'
    + NUMBER + r'(?:\s*' + UNIT + r'\b)?(?:\s*' + FORMULA + r')?|'
    + FORMULA
)
NON_HINDI = re.compile(r'[^\u0900-\u097f\n]+')
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


def _protect(text: str) -> tuple[str, dict[str, str]]:
    values = {}
    def replace(match):
        token = _token(len(values))
        values[token] = match.group()
        return token
    return PROTECTED.sub(replace, text), values


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


@lru_cache(maxsize=256)
def _translate(text: str, api_key: str, timeout: float) -> str:
    request = Request(ENDPOINT, data=json.dumps({
        'input': text, 'source_language_code': 'en-IN',
        'target_language_code': 'hi-IN', 'model': MODEL,
        'mode': 'formal', 'numerals_format': 'international',
    }).encode('utf-8'), headers={
        'Content-Type': 'application/json', 'api-subscription-key': api_key,
    }, method='POST')
    with urlopen(request, timeout=timeout) as response:
        output = json.loads(response.read().decode('utf-8')).get('translated_text')
    if not isinstance(output, str) or not output.strip():
        raise ValueError('Empty translation')
    # Each protected item must occur exactly once and stay in source order.
    if TOKEN.findall(output) != TOKEN.findall(text):
        raise ValueError('Protected values changed')
    residual = TOKEN.sub('', output)
    if re.search(r'\d', residual) or re.search(r'[A-Za-z]', residual):
        raise ValueError('Translation contains new numbers or untranslated English')
    if not re.search(r'[\u0900-\u097f]', residual):
        raise ValueError('Hindi translation missing')
    return output.strip()


def translate_answer(result: dict) -> dict:
    """Translate English prose only; fall back atomically on any failure.

    Only final answer fragments leave the app. Questions, identity, evidence,
    references, and agent messages are not sent to Sarvam. Numeric/formula checks
    do not establish semantic correctness; translation still needs evaluation.
    """
    if 'translation' in result:
        return result
    original = str(result.get('answer', ''))
    info = {'provider': 'sarvam', 'model': MODEL, 'target_language': 'hi-IN'}
    def finish(answer, status, reason=None):
        metadata = {**info, 'status': status}
        if reason:
            metadata['reason'] = reason
        return {**result, 'answer': answer, 'translation': metadata}
    if _setting('KISAANAI_HINDI_TRANSLATION', '1').lower() in {'0', 'false', 'no'}:
        return finish(original, 'disabled')
    if TOKEN.search(original):
        return finish(original, 'fallback', 'reserved_token_in_source')
    protected, values = _protect(original)
    segments = list(NON_HINDI.finditer(protected))
    # Formula-only passages need no model; keep approved scientific notation.
    pending = [m for m in segments if re.search(r'[A-Za-z]', TOKEN.sub('', m.group()))]
    if not pending:
        return finish(original, 'not_needed')
    key = _setting('SARVAM_API_KEY')
    if not key:
        return finish(original, 'not_configured')
    deadline = time.monotonic() + 20
    calls = 0
    replacements = []
    try:
        for match in pending:
            segment = match.group()
            # Keep Markdown list/header markers and surrounding whitespace local.
            prefix = re.match(r'^[\s#>*+\-]*', segment).group()
            suffix = re.search(r'\s*$', segment).group()
            core = segment[len(prefix):len(segment) - len(suffix) if suffix else None]
            translated = []
            for chunk in _chunks(core):
                calls += 1
                remaining = deadline - time.monotonic()
                if calls > 24 or remaining <= 0:
                    raise TimeoutError('Translation budget exceeded')
                translated.append(_translate(chunk, key, min(8.0, remaining)))
            replacements.append((match.start(), match.end(), prefix + ' '.join(translated) + suffix))
        output = protected
        for start, end, replacement in reversed(replacements):
            output = output[:start] + replacement + output[end:]
        if Counter(TOKEN.findall(output)) != Counter(values.keys()):
            raise ValueError('Protected values lost')
        for token, value in values.items():
            output = output.replace(token, value)
        return finish(output, 'translated')
    except Exception as exc:
        # Never expose provider bodies, API keys, or farmer text in diagnostics.
        return finish(original, 'fallback', type(exc).__name__)
