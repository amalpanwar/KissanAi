"""Bounded document and web evidence lookup without downloading an LLM.

Relevance uses bilingual concept coverage, not proof that a snippet answers
all parts of a question. Output is labelled excerpts, never invented synthesis.
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse

from app.agent_system import AgentResult
from app.commodity_lookup import normalize, resolve_commodities

STOP = set('how what is are the a an to of for in and or please tell me can do i my ka ki ke ko mein me hai hain kare karen kaise kese kya batao bataye btaye mujhe करें करे कैसे क्या की का के में है हैं मुझे बताएं'.split())
STOP.update('k दें karvaye karwaye karvana karwana karaye bare baare barey jankari jaankari de den दीजिए जानकारी बारे करवाएं करवाये करवाएँ कराएं बताइए give information about explain describe'.split())
GROUPS = {
    'soil': 'soil mitti matti मृदा मिट्टी',
    'variety': 'variety varieties किस्म किस्में किस्मों प्रजाति kism kisam',
    'fertility': 'fertility fertile urvarta urvarata उर्वरता उपजाऊपन उपजाऊ',
    'organic': 'organic जैविक कार्बनिक',
    'test': 'test tests testing check checking जांच जाँच jaanch janch parikshan परीक्षण',
    'sample': 'sample samples sampling नमूना नमूने नमूना लेना',
    'fertilizer': 'fertilizer fertiliser fertilizers खाद उर्वरक',
    'irrigation': 'irrigation sinchai सिंचाई',
    'seed': 'seed seeds beej बीज',
}
SYNONYMS = {word: key for key, words in GROUPS.items() for word in words.split()}
DOMAINS = ['icar.gov.in', 'soilhealth.dac.gov.in', 'agriwelfare.gov.in', 'tnau.ac.in']


def terms(text):
    return {SYNONYMS.get(word, word) for word in normalize(text).split() if word not in STOP and len(word)>1}


def search_question(question):
    words = normalize(question).split()
    normalized = ' '.join(SYNONYMS.get(word, word) for word in words if word not in STOP)
    crops = resolve_commodities(question)
    return normalized + (' ' + ' '.join(crops) if crops else '')


def rank(question, records, query_terms=None):
    query = query_terms if query_terms is not None else terms(search_question(question))
    ranked = []
    seen = set()
    for row in records:
        text = str(row.get('text') or '').strip()
        source = str(row.get('source_file') or '').strip()
        if not text or not source or (source,text) in seen:
            continue
        seen.add((source,text))
        # Source filenames alone cannot make an unrelated paragraph relevant.
        overlap = query & terms(text)
        # Require the question's concepts together, not a single common word
        # such as soil in a pesticide registration table.
        if len(query) < 2 or len(overlap) / len(query) < 0.75:
            continue
        parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
        parts = sorted(parts, key=lambda part: len(query & terms(part)), reverse=True)
        relevant = [part.strip() for part in parts
                    if len(query & terms(part)) / len(query) >= 0.75][:3]
        if not relevant:
            continue
        snippet = " ".join(relevant)[:1800]
        item = dict(row, text=snippet, coverage=len(query & terms(snippet))/max(1,len(query)))
        if any(r['source_file'] == source for r in ranked):
            continue
        ranked.append(item)
    return sorted(ranked,key=lambda row:row['coverage'],reverse=True)[:5]


def documents(advisor, payload):
    path = Path(advisor.cfg.metadata_path)
    records = []
    query_terms = terms(search_question(payload["question"]))
    started = monotonic()
    truncated = False
    try:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'text','source_file'}.issubset(reader.fieldnames or []):
                raise ValueError('Document index schema unavailable')
            for i,row in enumerate(reader):
                if i >= 100000 or monotonic()-started > 5:
                    truncated=True; break
                # Keep memory bounded while checking the indexed shared docs.
                found=rank(payload['question'],[{'text':str(row.get('text') or '')[:12000],
                                               'source_file':row.get('source_file','')}], query_terms)
                if found:
                    records=rank(payload['question'],records+found,query_terms)
    except (OSError,ValueError,csv.Error):
        return AgentResult('', 'unavailable', evidence={'records':[], 'reason':'document_index_unavailable'})
    return AgentResult('', 'partial' if truncated else 'ok', evidence={'records':records,'truncated':truncated})


def web(payload):
    from app.web_search import search_with_status
    response=search_with_status(search_question(payload['question']),num=5,include_domains=DOMAINS)
    records=[]
    for hit in response['results']:
        url=urlparse(hit.link)
        host=(url.hostname or '').lower()
        if url.scheme=='https' and not url.username and any(host==d or host.endswith('.'+d) for d in DOMAINS):
            records.append({'text':hit.snippet[:1800],'source_file':hit.link,'title':hit.title})
    return AgentResult('',response['status'], evidence={'records':rank(payload['question'],records),
                       'provider':response['provider'],'reason':response.get('reason',''),
                       'query':search_question(payload['question']), 'results_received':len(response['results']),
                       'relevant_results':len(rank(payload['question'],records))})


def excerpt(row):
    # Display data as quoted text; the research route never executes/generates
    # instructions from retrieved documents or web snippets.
    text=re.sub(r'\s+',' ',str(row['text'])).strip()
    if len(text)>650:
        text=text[:650].rsplit(' ',1)[0]+'…'
    return text


def evidence_answer(rows, heading):
    parts=[heading]
    for i,row in enumerate(rows[:3],1):
        parts += [f"{i}. {excerpt(row)}"]
    return '\n\n'.join(parts)


class ResearchAgent:
    def run(self, goal, payload, bus):
        local=bus.ask('research','documents','Check indexed shared documents',payload)
        remote=bus.ask('research','web_search','Check current authoritative web sources',payload)
        rows=rank(payload['question'],local.evidence.get('records',[])+remote.evidence.get('records',[]))
        direct=[r for r in rows if r['coverage']>=0.75 and len(terms(search_question(payload['question'])))>=2]
        checks={'documents':local.status,'web_search':remote.status,'web_provider':remote.evidence.get('provider'),
                'web_reason':remote.evidence.get('reason'),'document_reason':local.evidence.get('reason'),
                'web_query':remote.evidence.get('query'),
                'web_results_received':remote.evidence.get('results_received',0),
                'web_relevant_results':remote.evidence.get('relevant_results',0),
                'document_relevant_results':len(local.evidence.get('records',[]))}
        limitations=[]
        if local.status!='ok': limitations.append('साझा दस्तावेज़ों की खोज पूरी नहीं हो सकी।')
        if remote.status!='ok': limitations.append('वेब खोज उपलब्ध नहीं हुई; उसकी पुष्टि नहीं हो सकी।')
        if direct:
            answer=evidence_answer(direct,'आपके सवाल से मेल खाते स्रोतों के अनुसार:\n\nनीचे उपलब्ध जानकारी के अंश हैं; इन्हें पूरे उत्तर या व्यक्तिगत सलाह के रूप में सत्यापित नहीं किया गया है:')
            if limitations: answer+='\n\n'+' '.join(limitations)
            return AgentResult(answer,'partial' if limitations else 'ok',[r['source_file'] for r in direct[:3]],
                               {'checks':checks}, {'topic':'research'})
        answer='इस सवाल का उत्तर देने के लिए पर्याप्त संबंधित जानकारी नहीं मिली। कृपया विषय/फसल स्पष्ट करें या संबंधित दस्तावेज़ साझा करें।'
        if limitations: answer+='\n\n'+' '.join(limitations)
        return AgentResult(answer,'unavailable',evidence={'checks':checks},metadata={'topic':'research'})


def answer_with_research_followup(advisor, query, state):
    _,question=advisor._split_context_and_question(query)
    offer=state.pop('pending_research_offer',None)
    if offer and normalize(question) in {'yes','haan','han','हाँ','हां','जी हाँ','दिखाएं','proceed','yes please'}:
        rows=offer['records']
        return {'answer':evidence_answer(rows,'ये पहले मिले संबंधित स्रोत-अंश हैं; आपके मूल सवाल का पूरा उत्तर इनमें सत्यापित नहीं हुआ है:'),
                'references':[r['source_file'] for r in rows], 'topic':'research_related','agent_status':'partial'}
    result=advisor.answer(query)
    if result.get('research_offer'):
        state['pending_research_offer']=result['research_offer']
    return result
