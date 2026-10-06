import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app import agriculture_news as news
from app.agent_system import Coordinator, AgentResult, AgronomyAgent, ToolAgent
from app.agent_tools import AdvisorTools


def story(title='प्याज की नई किस्म', age=60, url=None):
    return {'story': {'headline': title, 'subheadline': '<b>' + title + '</b>',
        'url': url or 'https://hindi.downtoearth.org.in/agriculture/onion-update',
        'first-published-at': (time.time() - age) * 1000}}


class NewsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target, value in [('_memo', None), ('_last_attempt', 0.0), ('CACHE_PATH', Path(self.tmp.name) / 'news.json')]:
            p = patch.object(news, target, value); p.start(); self.addCleanup(p.stop)

    def test_parse_dates_sort_dedupe_and_ignore_unsafe_or_future_links(self):
        payload = {'items': [story(age=100), story(age=10), story(url='https://evil.example/agriculture/a'),
                             story(age=-3600, url='https://hindi.downtoearth.org.in/agriculture/future'),
                             story(url='https://hindi.downtoearth.org.in:bad/agriculture/a'), None]}
        rows = news.parse_feed(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['summary'], 'प्याज की नई किस्म')
        self.assertTrue(rows[0]['published_at'].endswith('+00:00'))
        self.assertEqual(news.parse_feed({'items': 'bad'}), [])

    def test_cache_ttl_failure_stale_expiry_and_read_only(self):
        now = time.time()
        with patch.object(news, '_fetch', return_value={'items': [story()]}) as fetch:
            self.assertEqual(news.get_news(now=now)['status'], 'ok')
            self.assertEqual(news.get_news(now=now + 10)['status'], 'ok')
            self.assertEqual(fetch.call_count, 1)
        with patch.object(news, '_fetch', side_effect=TimeoutError):
            stale = news.get_news(now=now + news.TTL + 1)
            self.assertEqual(stale['status'], 'stale')
            self.assertIn('पिछली सुरक्षित', news.format_news(stale, stale['articles']))
            self.assertEqual(news.get_news(now=now + news.MAX_STALE + 1)['status'], 'unavailable')
        news._memo = None
        with patch.object(news, '_fetch', side_effect=AssertionError('cache only must not fetch')):
            self.assertEqual(len(news.get_news(cache_only=True, now=now + 10)['articles']), 1)

    def test_relevance_and_age(self):
        rows = news.parse_feed({'items': [story(), story('गेहूं की खबर', url='https://hindi.downtoearth.org.in/agriculture/wheat'),
                                               story('धान की खबर', age=40*86400, url='https://hindi.downtoearth.org.in/agriculture/rice')]})
        self.assertEqual(len(news.relevant_articles(rows, 'pyaz ki kheti kaise kare', related_only=True)), 1)
        self.assertEqual(len(news.relevant_articles(rows, 'gehu ki taza khabar')), 1)
        self.assertEqual(news.relevant_articles(rows, 'sugarcane news'), [])
        self.assertEqual(news.relevant_articles(rows, 'dhan news'), [])
        self.assertEqual(news.relevant_articles(rows, 'general farming', related_only=True), [])
        for question in ['आज की कृषि खबरें', 'What is the latest agriculture news?', 'aaj ki krishi news batao']:
            self.assertEqual(len(news.relevant_articles(rows, question)), 2)

    def test_news_agent_uses_feed_without_model_and_agronomy_requests_related_news(self):
        snap = {'status': 'ok', 'articles': news.parse_feed({'items': [story()]}), 'fetched_at': time.time()}
        handlers = AdvisorTools(None)
        agents = {'news': ToolAgent(handlers.news), 'agronomy': AgronomyAgent(lambda g,p: AgentResult('Cultivation guide'))}
        with patch.object(news, 'get_news', return_value=snap) as fetch:
            result = Coordinator(agents, planner=lambda _: self.fail('No model needed')).answer('latest agriculture news')
            self.assertEqual(result['topic'], 'news')
            self.assertIn('onion-update', result['answer'])
            result = Coordinator(agents).answer('pyaz ki kheti kaise kare')
            self.assertIn('Cultivation guide', result['answer'])
            self.assertIn('संबंधित हालिया समाचार', result['answer'])
            self.assertIn(('agronomy', 'news'), [(m['sender'], m['recipient']) for m in result['agent_trace']['messages']])
            self.assertTrue(fetch.call_args.kwargs['cache_only'])

    def test_sidebar_displays_dates_links_and_failure(self):
        from streamlit.testing.v1 import AppTest
        snap = {'status': 'stale', 'articles': news.parse_feed({'items': [story()]}), 'fetched_at': time.time()}
        with patch('app.news_panel.get_news', return_value=snap):
            at = AppTest.from_string('import streamlit as st\nfrom app.news_panel import render_news_panel\nwith st.sidebar:\n    render_news_panel()').run()
            self.assertFalse(at.exception)
            self.assertTrue(at.sidebar.warning)
            self.assertTrue(any('प्रकाशित:' in item.value for item in at.sidebar.caption))
        with patch('app.news_panel.get_news', return_value={'status':'unavailable','articles':[],'fetched_at':None}):
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(at.sidebar.info)


if __name__ == '__main__':
    unittest.main()
