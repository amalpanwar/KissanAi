"""Remote synthesis from reranked document evidence; Hindi is translated separately."""
import json
from urllib.request import Request, urlopen
from app.hindi_translation import _setting

ENDPOINT = 'https://api.sarvam.ai/v1/chat/completions'


def generate_text(prompt):
    """Synthesize reranked evidence remotely; final Hindi comes from translation."""
    key = _setting('SARVAM_API_KEY')
    if not key:
        raise RuntimeError('sarvam_not_configured')
    if _setting('KISAANAI_SARVAM_GENERATION', '1').lower() in {'0', 'false', 'no'}:
        raise RuntimeError('sarvam_generation_disabled')
    request = Request(ENDPOINT, data=json.dumps({
        'model': _setting('KISAANAI_SARVAM_CHAT_MODEL', 'sarvam-105b'),
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'reasoning_effort': None, 'max_tokens': 1800,
    }).encode(), headers={'Content-Type': 'application/json', 'api-subscription-key': key}, method='POST')
    with urlopen(request, timeout=25) as response:
        choice = json.loads(response.read().decode())['choices'][0]
    answer = choice['message']['content']
    if choice.get('finish_reason') != 'stop' or not isinstance(answer, str) or not answer.strip():
        raise ValueError('incomplete_generation')
    return answer
