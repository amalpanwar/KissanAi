"""Session-scoped Hindi query previews; API keys stay on the server."""
import json
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from app.hindi_translation import ENDPOINT, _setting, _token, TOKEN


def translate_query(text, names=(), target="hi-IN"):
    text = str(text).strip()
    if target not in {'hi-IN', 'en-IN'}:
        raise ValueError('Unsupported query language')
    needs_translation = re.search('[A-Za-z]', text) if target == 'hi-IN' else re.search('[\u0900-\u097f]', text)
    if not text or not needs_translation:
        return {'status':'unchanged', 'text':text}
    if len(text) > 1000:
        return {'status':'too_long', 'text':text}
    key = _setting('SARVAM_API_KEY')
    if not key:
        return {'status':'not_configured', 'text':text}
    protected = []
    def mask(match):
        protected.append(match.group())
        return _token(len(protected)-1)
    name_patterns = [re.escape(n) for n in sorted({n for n in names if isinstance(n,str) and n.strip()},key=len,reverse=True)]
    pattern = '|'.join(name_patterns + [r'\d+(?:[.,:/-]\d+)*'])
    masked = re.sub(pattern,mask,text,flags=re.I)
    if len(masked) > 1000:
        return {'status':'too_long', 'text':text}
    request = Request(ENDPOINT,data=json.dumps({
        'input':masked, 'source_language_code':'auto', 'target_language_code':target,
        'model':'mayura:v1', 'mode':'formal', 'output_script':'fully-native' if target == 'hi-IN' else None,
        'numerals_format':'international',
    }).encode(),headers={'Content-Type':'application/json','api-subscription-key':key},method='POST')
    try:
        with urlopen(request,timeout=8) as response:
            output=json.loads(response.read().decode()).get('translated_text')
        if not isinstance(output,str) or not re.search('[\u0900-\u097f]' if target == 'hi-IN' else '[A-Za-z]', output):
            return {'status':'invalid_output','text':text}
        if TOKEN.findall(output) != TOKEN.findall(masked):
            return {'status':'protected_values_changed','text':text}
        for i,value in enumerate(protected):
            output=output.replace(_token(i),value)
        return {'status':'ok','text':output.strip()}
    except HTTPError as exc:
        # Never return upstream bodies, request headers or credentials to the UI.
        status = {400:'invalid_request', 401:'authentication_failed', 403:'authentication_failed',
                  422:'invalid_request', 429:'rate_limited'}.get(exc.code, 'service_error')
        return {'status':status, 'text':text}
    except TimeoutError:
        return {'status':'timeout', 'text':text}
    except URLError as exc:
        return {'status':'timeout' if isinstance(exc.reason, TimeoutError) else 'network_error', 'text':text}
    except (ValueError, TypeError, AttributeError):
        return {'status':'invalid_output', 'text':text}
    except Exception:
        return {'status':'unavailable','text':text}
