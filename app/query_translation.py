"""Session-scoped Hindi query previews; API keys stay on the server."""
import json
import re
from urllib.request import Request, urlopen
from app.hindi_translation import ENDPOINT, _setting, _token, TOKEN


def translate_query(text, names=()):
    text = str(text).strip()
    if not text or not re.search('[A-Za-z]', text):
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
        'input':masked, 'source_language_code':'auto', 'target_language_code':'hi-IN',
        'model':'mayura:v1', 'mode':'formal', 'output_script':'native',
        'numerals_format':'international',
    }).encode(),headers={'Content-Type':'application/json','api-subscription-key':key},method='POST')
    try:
        with urlopen(request,timeout=8) as response:
            output=json.loads(response.read().decode()).get('translated_text')
        if not isinstance(output,str) or not re.search('[\u0900-\u097f]',output):
            raise ValueError('No Hindi output')
        if TOKEN.findall(output) != TOKEN.findall(masked):
            raise ValueError('Protected values changed')
        for i,value in enumerate(protected):
            output=output.replace(_token(i),value)
        return {'status':'ok','text':output.strip()}
    except Exception:
        return {'status':'unavailable','text':text}
