"""Evidence-first research agents using language-aware semantic retrieval."""
from urllib.parse import urlparse
from app.agent_system import AgentResult

DOMAINS = ['icar.gov.in', 'soilhealth.dac.gov.in', 'agriwelfare.gov.in', 'tnau.ac.in']


def documents(advisor,payload):
    from app.multilingual_retrieval import retrieve_documents
    return retrieve_documents(advisor,payload)


def web(payload,advisor=None):
    from app.web_search import search_with_status
    from app.multilingual_retrieval import rank_web
    query=(payload.get('query_variants') or {}).get('en-IN',payload['question'])
    response=search_with_status(query,num=5,include_domains=DOMAINS)
    candidates=[]
    for hit in response['results']:
        url=urlparse(hit.link);host=(url.hostname or '').lower()
        if url.scheme=='https' and not url.username and any(host==d or host.endswith('.'+d) for d in DOMAINS):
            candidates.append({'text':hit.snippet[:4000],'source_file':hit.link,'title':hit.title})
    rows=rank_web(advisor,payload,candidates) if candidates and advisor else []
    ranking_unavailable=bool(candidates) and (advisor is None or getattr(advisor,'embedder',None) is None)
    return AgentResult('', 'unavailable' if ranking_unavailable else response['status'],
        evidence={'records':rows,'provider':response['provider'],
                  'reason':'semantic_encoder_unavailable' if ranking_unavailable else response.get('reason',''),
                  'query':query,'results_received':len(response['results']),'relevant_results':len(rows)})


def evidence_answer(rows,heading):
    parts=[heading]
    for i,row in enumerate(rows[:3],1):
        text=str(row['text']).strip()
        if len(text)>1800:
            text=text[:1800].rsplit(' ',1)[0]+'…'
        parts.append(f'{i}. {text}')
    return '\n\n'.join(parts)


class ResearchAgent:
    def run(self,goal,payload,bus):
        local=bus.ask('research','documents','Search document embeddings in each document language',payload)
        rows=local.evidence.get('records',[])
        remote=None
        if not rows:
            remote=bus.ask('research','web_search','Search official web sources after document lookup',payload)
            rows=remote.evidence.get('records',[])
        checks={'documents':local.status,'document_search':local.evidence,
                'web_search':remote.status if remote else 'not_needed'}
        if remote:
            checks.update({'web_provider':remote.evidence.get('provider'),'web_reason':remote.evidence.get('reason')})
        if rows:
            answer=evidence_answer(rows,'आपके सवाल से अर्थ के आधार पर मिले स्रोत-अंश:')
            answer+='\n\nये स्रोत में दी गई जानकारी है; स्थानीय उपयुक्तता या “सबसे बेहतर” विकल्प की पुष्टि के लिए स्रोत की शर्तें भी देखें।'
            return AgentResult(answer,'partial' if local.status!='ok' else 'ok',
                               list(dict.fromkeys(r['source_file'] for r in rows[:3])),{'checks':checks}, {'topic':'research'})
        answer='इस सवाल का उत्तर देने के लिए पर्याप्त संबंधित जानकारी नहीं मिली।'
        if local.status!='ok':
            answer+=' दस्तावेज़ की semantic खोज या आवश्यक अनुवाद पूरा नहीं हो सका।'
        if remote and remote.status!='ok':
            answer+=' वेब खोज या उसके परिणामों की जाँच भी उपलब्ध नहीं हुई।'
        answer+=' कृपया सवाल स्पष्ट करें या संबंधित दस्तावेज़ की उपलब्धता जाँचें।'
        return AgentResult(answer,'unavailable',evidence={'checks':checks},metadata={'topic':'research'})


def answer_with_research_followup(advisor,query,state):
    # Older weak-match offers must never override the user's current question.
    state.pop('pending_research_offer',None)
    return advisor.answer(query)
