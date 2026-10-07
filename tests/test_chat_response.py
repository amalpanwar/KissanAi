import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock
from app.chat_response import clean_references, render_response_footer


class ResponseFooterTests(unittest.TestCase):
    def test_sources_are_not_empty_markdown_bullets(self):
        self.assertEqual(clean_references(['-', '*', '', None, 'guide.pdf', 'guide.pdf']), ['guide.pdf'])
        st=MagicMock()
        render_response_footer(st, {'references':['*','guide_name.pdf','https://icar.gov.in/soil']})
        st.text.assert_called_once_with('1. guide_name.pdf')
        st.link_button.assert_called_once_with('Source 2: icar.gov.in','https://icar.gov.in/soil')

    def test_actual_web_failure_and_results_visible(self):
        st=MagicMock()
        render_response_footer(st, {'agent_trace':{'plan':['research'], 'messages':[
            {'kind':'result','sender':'web_search','payload':{'status':'unavailable',
             'evidence':{'provider':'tavily','reason':'search_request_failed','query':'soil fertility',
                         'results_received':0,'relevant_results':0}}}]}})
        displayed=[c.args[0] for c in st.write.call_args_list]
        self.assertIn('Web search: tavily — unavailable',displayed)
        self.assertIn('Search outcome: search_request_failed',displayed)

    def test_history_and_live_use_same_footer_after_answer(self):
        tree=ast.parse(Path('streamlit_app.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='render_response_footer']
        self.assertEqual(len(calls),2)
        history=next(n for n in tree.body if isinstance(n,ast.For)
                     and ast.unparse(n.target)=='item' and 'chat_history' in ast.unparse(n.iter))
        history_source=ast.unparse(history)
        self.assertLess(history_source.index("st.write(item['text'])"),history_source.index('render_response_footer'))
        source=Path('streamlit_app.py').read_text()
        self.assertLess(source.index('st.write(final_answer)',source.index('"agent_trace": (None if intent_price')),
                        source.index('render_response_footer(st, st.session_state.chat_history[-1])'))

if __name__=='__main__': unittest.main()
