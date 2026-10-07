"""One footer for fresh and replayed replies; tool activity follows the answer."""
import re
from urllib.parse import urlparse


def clean_references(values):
    if not isinstance(values, (list, tuple)):
        values = [values] if isinstance(values, str) else []
    return list(dict.fromkeys(str(value).strip() for value in values
                             if isinstance(value, str) and re.search(r'\w', value)))


def render_response_footer(st, item):
    from app.translation_status import render_translation_status
    if item.get('translation'):
        render_translation_status(st, item['translation'])
    trace = item.get('agent_trace') or {}
    if trace:
        with st.expander('Plan and agent activity'):
            st.write(trace.get('goal', ''))
            st.write(' → '.join(trace.get('plan', [])))
            # Tool outcomes are observable evidence, not invented reasoning.
            for message in trace.get('messages', []):
                if message.get('kind') != 'result':
                    continue
                sender = message.get('sender')
                payload = message.get('payload') or {}
                evidence = payload.get('evidence') or {}
                if sender == 'web_search':
                    provider = evidence.get('provider') or 'none'
                    status = payload.get('status', 'unavailable')
                    st.write(f"Web search: {provider} — {status}")
                    if evidence.get('query'):
                        st.write('Search query: ' + evidence['query'])
                    st.write(f"Results received: {evidence.get('results_received', 0)}; relevant: {evidence.get('relevant_results', 0)}")
                    if evidence.get('reason'):
                        st.write('Search outcome: ' + evidence['reason'])
                elif sender == 'documents':
                    st.write(f"Documents: {payload.get('status', 'unavailable')}; relevant passages: {len(evidence.get('records', []))}")
            st.json(trace.get('decisions', []))
    refs = clean_references(item.get('references', []))
    if refs:
        with st.expander('Sources Used'):
            for i, source in enumerate(refs, 1):
                parsed = urlparse(source)
                if parsed.scheme in {'https', 'http'} and parsed.hostname:
                    st.link_button(f'Source {i}: {parsed.hostname}', source)
                else:
                    # Plain text protects filenames from Markdown list/escape parsing.
                    st.text(f'{i}. {source}')
