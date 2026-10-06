"""Bounded, dated metadata from Down To Earth Hindi's public agriculture feed."""
from __future__ import annotations

import html
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urljoin, quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

SOURCE_URL = 'https://hindi.downtoearth.org.in/agriculture'
FEED_URL = 'https://hindi.downtoearth.org.in/api/v1/collections/agriculture?limit=30'
CACHE_PATH = Path(__file__).resolve().parents[1] / 'data/processed/agriculture_news_cache.json'
TTL = 900
MAX_STALE = 86400
_lock = threading.Lock()
_memo = None
_last_attempt = 0.0


def clean_text(value, limit=200):
    text = re.sub(r'<[^>]*>', ' ', html.unescape(str(value or '')))
    return ' '.join(text.split())[:limit].strip()


def safe_article_url(value):
    url = urljoin(SOURCE_URL + '/', str(value or ''))
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if (parsed.scheme != 'https' or parsed.hostname != 'hindi.downtoearth.org.in'
            or parsed.username or parsed.password or port not in (None, 443)
            or not parsed.path.startswith('/agriculture/') or parsed.query or parsed.fragment):
        return None
    return "https://hindi.downtoearth.org.in" + quote(parsed.path, safe="/%-._~")


def parse_feed(payload, now=None):
    now = time.time() if now is None else now
    articles, seen = [], set()
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        return []
    for item in payload['items'][:60]:
        if not isinstance(item, dict) or not isinstance(item.get('story'), dict):
            continue
        story = item['story']
        url = safe_article_url(story.get('url') or '/' + str(story.get('slug') or ''))
        title = clean_text(story.get('headline'), 200)
        try:
            published = float(story['first-published-at']) / 1000
        except (TypeError, ValueError, KeyError):
            continue
        if not url or not title or url in seen or not 0 < published <= now + 300:
            continue
        seen.add(url)
        articles.append({'title': title, 'summary': clean_text(story.get('subheadline'), 220),
                         'url': url, 'published_at': datetime.fromtimestamp(published, timezone.utc).isoformat()})
    return sorted(articles, key=lambda a: a['published_at'], reverse=True)[:30]


def _fetch():
    request = Request(FEED_URL, headers={'User-Agent': 'KisaanAI/1.0 (agriculture news links)', 'Accept': 'application/json'})
    with urlopen(request, timeout=6) as response:
        if urlsplit(response.geturl()).hostname != 'hindi.downtoearth.org.in':
            raise ValueError('Unexpected feed host')
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('News feed too large')
    return json.loads(raw)


def _load_cache():
    try:
        cached = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
        if not isinstance(cached.get('articles'), list) or not isinstance(cached.get('fetched_at'), (float, int)):
            return None
        # Revalidate disk metadata; never treat cached content as instructions.
        rows = []
        for item in cached['articles'][:30]:
            url = safe_article_url(item.get('url'))
            dt = datetime.fromisoformat(item['published_at'])
            if not url or dt.tzinfo is None:
                continue
            rows.append(dict(title=clean_text(item.get('title')), summary=clean_text(item.get('summary'), 220),
                             url=url, published_at=dt.astimezone(timezone.utc).isoformat()))
        return {'articles': rows, 'fetched_at': cached['fetched_at']}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def get_news(*, cache_only=False, now=None):
    global _memo, _last_attempt
    now = time.time() if now is None else now
    with _lock:
        if _memo is None:
            _memo = _load_cache()
        fresh = _memo and 0 <= now - _memo['fetched_at'] < TTL
        if not fresh and not cache_only and now - _last_attempt >= 60:
            _last_attempt = now
            try:
                articles = parse_feed(_fetch(), now)
                if not articles:
                    raise ValueError('No dated agriculture articles')
                _memo = {'articles': articles, 'fetched_at': now}
                try:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    temp = CACHE_PATH.with_suffix('.tmp')
                    temp.write_text(json.dumps(_memo, ensure_ascii=False), encoding='utf-8')
                    temp.replace(CACHE_PATH)
                except OSError:
                    pass  # Memory cache still works on read-only hosting.
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
        if not _memo or not 0 <= now - _memo['fetched_at'] <= MAX_STALE:
            return {'articles': [], 'status': 'unavailable', 'fetched_at': None}
        return {**_memo, 'status': 'ok' if now - _memo['fetched_at'] < TTL else 'stale'}


NEWS_INTENT = re.compile(r'\b(news|headlines?|khabar|khabre|samachar|latest updates?)\b|खबर|ख़बर|समाचार|ताज़ा जानकारी|ताजा जानकारी', re.I)
CROPS = {
    'weather': ('weather', 'rain', 'monsoon', 'mausam', 'barish', 'मौसम', 'बारिश', 'वर्षा', 'मानसून'),
    'msp': ('msp', 'समर्थन मूल्य'),
    'policy': ('policy', 'scheme', 'yojana', 'योजना', 'नीति'),
    'wheat': ('wheat', 'gehu', 'gehun', 'gehoon', 'गेहूं', 'गेहूँ'),
    'sugarcane': ('sugarcane', 'ganna', 'गन्ना', 'गन्ने'),
    'onion': ('onion', 'pyaj', 'pyaz', 'प्याज', 'प्याज़'),
    'rice': ('rice', 'paddy', 'dhan', 'धान', 'चावल'),
    'mustard': ('mustard', 'sarso', 'sarson', 'सरसों'),
    'potato': ('potato', 'aloo', 'आलू'),
    'pesticide': ('pesticide', 'pesticides', 'कीटनाशक'),
    'fertilizer': ('fertilizer', 'urea', 'उर्वरक', 'यूरिया'),
}


def _contains(text, term):
    return re.search(r'(?<![\w\u0900-\u097f])' + re.escape(term) + r'(?![\w\u0900-\u097f])', text, re.I) is not None


def relevant_articles(articles, question, *, related_only=False, now=None):
    now = time.time() if now is None else now
    groups = [terms for terms in CROPS.values() if any(_contains(question, word) for word in terms)]
    if not groups:
        if related_only:
            return []
        # General agriculture-news requests show the whole feed; unsupported specific
        # terms must not silently receive unrelated stories.
        generic = {'news','latest','updates','update','agriculture','agricultural','farming','farmer','farmers','headlines',
                   'khabar','khabre','samachar','aaj','ki','ka','ke','kheti','krishi','kisan','kisaan','taza','taaza','batao','bataiye',
                   'please','show','me','the','what','is','are','any','tell','give','about','on','in','of','and','today','hindi','hai','hain','kya',
                   'खबर','खबरें','खबरों','समाचार','कृषि','खेती','किसान','किसानों','की','का','के','में','आज','ताजा','ताज़ा','नई','नयी','जानकारी','बताएं','बताओ','बताइए','दिखाएं','है','हैं','क्या','हिंदी'}
        terms = [t for t in re.findall(r'[\w\u0900-\u097f]+', question.lower()) if t not in generic and len(t) > 2]
        if terms:
            groups = [tuple(terms)]
    selected = []
    for item in articles:
        try:
            age = now - datetime.fromisoformat(item['published_at']).timestamp()
        except (ValueError, KeyError):
            continue
        if age < -300 or age > 30 * 86400:
            continue
        text = item['title'] + ' ' + item.get('summary', '')
        if groups and not any(any(_contains(text, term) for term in group) for group in groups):
            continue
        selected.append(item)
    return selected[:2 if related_only else 5]


def format_news(snapshot, articles, *, related_only=False):
    if snapshot['status'] == 'unavailable':
        return 'कृषि समाचार अभी उपलब्ध नहीं हैं। कृपया थोड़ी देर बाद फिर पूछें।'
    if not articles:
        return 'इस विषय पर पिछले 30 दिनों की कोई संबंधित खबर इस स्रोत की उपलब्ध सूची में नहीं मिली।'
    heading = 'संबंधित हालिया समाचार' if related_only else 'हालिया कृषि समाचार'
    checked = datetime.fromtimestamp(snapshot['fetched_at'], ZoneInfo('Asia/Kolkata'))
    lines = [f'**{heading} — डाउन टू अर्थ हिंदी**', f'सूची जाँची गई: {checked:%d-%m-%Y %H:%M} IST']
    if snapshot['status'] == 'stale':
        lines.append('लाइव अपडेट नहीं मिल पाया; पिछली सुरक्षित समाचार सूची दिखाई जा रही है।')
    for item in articles:
        date = datetime.fromisoformat(item['published_at']).astimezone(ZoneInfo('Asia/Kolkata')).strftime('%d-%m-%Y')
        title = re.sub(r'([\\\[\]*_`<>])', r'\\\1', item['title'])
        lines.append(f"- {date}: [{title}]({item['url']})")
    lines.append('ये समाचार रिपोर्ट हैं; इन्हें स्थानीय मौसम, मंडी भाव या कीटनाशक की आधिकारिक सिफारिश न मानें।')
    return '\n\n'.join(lines)
