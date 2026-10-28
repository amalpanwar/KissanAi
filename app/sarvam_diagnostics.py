"""Opt-in Cloud-side checks using fixed public text, never farmer messages."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.hindi_translation import _has_hindi_letters, _setting

SAMPLE = 'This is a translation test. The sample label mentions Urea and 50 kg.'
_LOCK = threading.Lock()


def _probe(key, endpoint, model):
    started = time.monotonic()
    if endpoint == 'translate':
        url = 'https://api.sarvam.ai/translate'
        payload = {'input': SAMPLE, 'source_language_code': 'en-IN',
                   'target_language_code': 'hi-IN', 'model': model, 'mode': 'formal'}
    else:
        url = 'https://api.sarvam.ai/v1/chat/completions'
        payload = {'model': model, 'messages': [{'role': 'user', 'content':
            'Translate this fixed sample into Hindi. Keep Urea and 50 kg unchanged. '
            'Return only the translation.\n' + SAMPLE}],
            'temperature': 0, 'reasoning_effort': None, 'max_tokens': 300}
    result = {'model': model.replace(key, '[redacted]')[:120]}
    try:
        request = Request(url, data=json.dumps(payload).encode(), headers={
            'Content-Type': 'application/json', 'api-subscription-key': key}, method='POST')
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode())
        text = data.get('translated_text') if endpoint == 'translate' else data['choices'][0]['message']['content']
        if not isinstance(text, str):
            raise ValueError('invalid_response')
        # Only the fixed sample output is shown. Never return provider error
        # bodies, headers, request credentials, or actual farmer answer text.
        text = text.replace(key, '[redacted]')[:4000]
        result.update(status='responded', output=text, hindi_present=_has_hindi_letters(text))
    except Exception as exc:
        result.update(status='error', reason=f'http_{exc.code}' if isinstance(exc, HTTPError) else type(exc).__name__)
    result['elapsed_ms'] = round((time.monotonic() - started) * 1000)
    return result


@lru_cache(maxsize=4)
def _cached_probe(key, model, bucket):
    with ThreadPoolExecutor(max_workers=2) as pool:
        translate = pool.submit(_probe, key, 'translate', 'sarvam-translate:v1')
        chat = pool.submit(_probe, key, 'chat', model)
        return {'sample': SAMPLE, 'checked_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
                'translate': translate.result(), 'chat': chat.result()}


def run_diagnostics():
    key = _setting('SARVAM_API_KEY')
    if not key:
        return {'status': 'not_configured', 'sample': SAMPLE}
    model = _setting('KISAANAI_SARVAM_CHAT_MODEL', 'sarvam-105b')
    # Prevent concurrent clicks from multiplying identical billable probes.
    with _LOCK:
        return deepcopy(_cached_probe(key, model, int(time.time() // 300)))
