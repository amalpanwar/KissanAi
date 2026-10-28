import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.guide_response_cache import (GuideResponseCache, answer_with_cache,
    finalize_response, prepare_request, translation_signature)
from app.production_guide_evidence import GUIDE


class GuideCacheTests(unittest.TestCase):
    def setUp(self):
        if not GUIDE.exists():
            self.skipTest('Production guide absent')
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'cache.sqlite3'
        self.advisor = SimpleNamespace(_split_context_and_question=lambda q: ('', q),
            _extract_preferred_crop_from_context=lambda c: None)
        settings = patch('app.guide_response_cache._setting', side_effect=lambda k, d='': str(self.path) if k == 'KISAANAI_GUIDE_CACHE_PATH' else d)
        settings.start(); self.addCleanup(settings.stop)
        self.raw = {'answer': 'Prepare the field according to the guide.', 'agent_status': 'ok',
            'requires_hindi': True, 'topic': 'research', 'references': [str(GUIDE)],
            'retrieved': [{'source_file': str(GUIDE), 'text': 'FIELD PREPARATION',
                           'page': 135, 'guide_section': 'WHEAT', 'guide_overview': True}],
            'agent_trace': {'private_context': 'PRIVATE PERSON'}}
        self.finalize = patch('app.hindi_translation.translate_answer',
            side_effect=lambda result: {**result, 'answer': 'स्रोत के अनुसार खेत तैयार करें।',
                                       'translation': {'status': 'translated', 'version': 'test'}})
        self.translate = self.finalize.start(); self.addCleanup(self.finalize.stop)
        self.uncached = MagicMock(side_effect=lambda q: finalize_response(copy.deepcopy(self.raw)))

    def test_alias_equivalent_overviews_reuse_both_languages_and_pages(self):
        first = answer_with_cache(self.advisor, 'गेहूं की खेती कैसे करें?', self.uncached)
        second = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(first['guide_cache']['status'], 'stored')
        self.assertEqual(second['guide_cache']['status'], 'hit')
        self.assertEqual(second['english_answer'], self.raw['answer'])
        self.assertEqual(second['answer'], first['answer'])
        self.assertEqual(second['source_pages'], [{'source_file': str(GUIDE), 'pages': [135]}])
        self.uncached.assert_called_once()
        self.translate.assert_called_once()
        self.assertNotIn('agent_trace', second)
        # A new cache object reads the same durable on-disk entry.
        request = prepare_request(self.advisor, 'How to grow wheat?')
        self.assertIsNotNone(GuideResponseCache(self.path).get(request.key)['hindi'])

    def test_advisor_entry_point_reuses_cache_before_coordinator_work(self):
        from app.advisor import RAGAdvisor
        advisor = object.__new__(RAGAdvisor)
        advisor._extract_preferred_crop_from_context = lambda c: None
        advisor._answer_without_guide_cache = self.uncached
        first = advisor.answer('How to grow wheat?')
        second = advisor.answer('गेहूं की खेती कैसे करें?')
        self.assertEqual(first['guide_cache']['status'], 'stored')
        self.assertEqual(second['guide_cache']['status'], 'hit')
        self.uncached.assert_called_once()
        self.translate.assert_called_once()

    def test_generation_model_change_invalidates_english_too(self):
        answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        with patch('app.guide_response_cache._setting', side_effect=lambda k, d='': str(self.path) if k == 'KISAANAI_GUIDE_CACHE_PATH' else 'other-model' if k == 'KISAANAI_SARVAM_CHAT_MODEL' else d):
            result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(result['guide_cache']['status'], 'stored')
        self.assertEqual(self.uncached.call_count, 2)

    def test_cached_followup_preserves_crop_and_topic_metadata(self):
        self.raw.update(topic='crop_guide_followup', crop='Wheat')
        answer_with_cache(self.advisor, 'When should I irrigate wheat?', self.uncached)
        result = answer_with_cache(self.advisor, 'When should I irrigate wheat?', self.uncached)
        self.assertEqual(result['guide_cache']['status'], 'hit')
        self.assertEqual(result['topic'], 'crop_guide_followup')
        self.assertEqual(result['crop'], 'Wheat')

    def test_bajra_and_pearl_millet_share_the_resolved_section(self):
        left = prepare_request(self.advisor, 'Bajra ki kheti kaise kare')
        right = prepare_request(self.advisor, 'How to grow pearl millet?')
        self.assertEqual(left.key, right.key)
        self.assertEqual(left.section, 'Cumbu')
        self.assertNotEqual(left.key, prepare_request(self.advisor, 'How to grow wheat?').key)

    def test_focus_context_and_extra_qualifiers_do_not_collide(self):
        general = prepare_request(self.advisor, 'How to grow wheat?')
        for query in ['How to grow wheat organically?', 'When should I irrigate wheat?', 'How deep should I sow wheat?']:
            self.assertNotEqual(general.key, prepare_request(self.advisor, query).key)
        self.advisor._split_context_and_question = lambda q: ('State: Uttar Pradesh | District: Meerut', q)
        self.assertNotEqual(general.key, prepare_request(self.advisor, 'How to grow wheat?').key)

    def test_translation_failure_saves_english_and_retries_only_translation(self):
        self.translate.side_effect = [
            {**self.raw, 'translation': {'status': 'fallback', 'reason': 'TimeoutError'}},
            {**self.raw, 'answer': 'खेत तैयार करें।', 'translation': {'status': 'translated'}}]
        first = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(first['guide_cache']['status'], 'english_stored')
        request = prepare_request(self.advisor, 'How to grow wheat?')
        self.assertIsNone(GuideResponseCache(self.path).get(request.key)['hindi'])
        second = answer_with_cache(self.advisor, 'गेहूं की खेती कैसे करें?', self.uncached)
        third = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(second['guide_cache']['status'], 'english_hit')
        self.assertEqual(third['guide_cache']['status'], 'hit')
        self.assertEqual(third['answer'], 'खेत तैयार करें।')
        self.assertEqual(self.uncached.call_count, 1)
        self.assertEqual(self.translate.call_count, 2)

    def test_language_policy_change_reuses_english_but_rebuilds_hindi(self):
        answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        with patch('app.hindi_translation.TRANSLATION_VERSION', 'changed-policy'):
            result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(result['guide_cache']['status'], 'english_hit')
        self.assertEqual(self.uncached.call_count, 1)
        self.assertEqual(self.translate.call_count, 2)

    def test_pdf_content_change_invalidates_the_whole_entry(self):
        from app.guide_response_cache import file_digest
        first = prepare_request(self.advisor, 'How to grow wheat?')
        with patch('app.guide_response_cache.file_digest', side_effect=lambda p: 'changed-pdf' if Path(p) == GUIDE else file_digest(p)):
            second = prepare_request(self.advisor, 'How to grow wheat?')
        self.assertNotEqual(first.key, second.key)

    def test_generation_failures_and_dynamic_answers_are_not_cached(self):
        self.uncached.side_effect = None
        self.uncached.return_value = {'answer': 'उत्तर उपलब्ध नहीं है।', 'agent_status': 'unavailable'}
        for _ in range(2):
            result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
            self.assertEqual(result['guide_cache']['status'], 'not_stored')
        self.assertEqual(self.uncached.call_count, 2)
        for query in ['Wheat price today?', 'Wheat weather forecast', 'Wheat pesticide spray', 'Latest wheat news', 'Can I sow wheat today?', 'क्या आज गेहूं की बुवाई कर सकते हैं?']:
            self.assertIsNone(prepare_request(self.advisor, query))

    def test_mixed_news_or_web_sources_are_not_saved(self):
        self.raw['references'].append('https://example.org/news')
        for _ in range(2):
            result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
            self.assertEqual(result['guide_cache']['status'], 'not_stored')
        self.assertEqual(self.uncached.call_count, 2)

    def test_cache_storage_failure_does_not_block_the_answer(self):
        self.path.mkdir()
        result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(result['answer'], 'स्रोत के अनुसार खेत तैयार करें।')
        self.assertEqual(result['guide_cache']['status'], 'unavailable')

    def test_runtime_record_does_not_persist_questions_context_or_agent_traces(self):
        self.advisor._split_context_and_question = lambda q: ('Name: PRIVATE FARMER', q)
        answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT key,english,hindi FROM guide_answers').fetchone()
        serialized = str(row)
        for private in ['PRIVATE FARMER', 'PRIVATE PERSON', 'How to grow wheat?', 'agent_trace']:
            self.assertNotIn(private, serialized)

    def test_disable_flag_bypasses_cache(self):
        with patch('app.guide_response_cache._setting', side_effect=lambda k, d='': '0' if k == 'KISAANAI_GUIDE_RESPONSE_CACHE' else d):
            self.assertIsNone(prepare_request(self.advisor, 'How to grow wheat?'))

    def test_first_writer_cannot_pair_hindi_with_different_english(self):
        cache = GuideResponseCache(self.path)
        english = {'answer': 'first', 'retrieved': []}
        digest = cache.put_english('key', english)
        self.assertIsNone(cache.put_english('key', {'answer': 'second', 'retrieved': []}))
        self.assertFalse(cache.put_hindi('key', 'wrong-digest', {'answer': 'गलत', 'translation': {}}, 'sig'))
        self.assertTrue(cache.put_hindi('key', digest, {'answer': 'सही', 'translation': {}}, 'sig'))
        self.assertEqual(cache.get('key')['english']['answer'], 'first')
        self.assertEqual(cache.get('key')['hindi']['answer'], 'सही')

    def test_concurrent_repeated_requests_generate_and_translate_once(self):
        prepare_request(self.advisor, 'How to grow wheat?')
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached), range(4)))
        self.assertEqual(self.uncached.call_count, 1)
        self.assertEqual(self.translate.call_count, 1)
        self.assertEqual(sum(result['guide_cache']['status'] == 'hit' for result in results), 3)

    def test_corrupt_sqlite_cache_falls_back_to_live_answer(self):
        self.path.write_text('not a sqlite database')
        result = answer_with_cache(self.advisor, 'How to grow wheat?', self.uncached)
        self.assertEqual(result['answer'], 'स्रोत के अनुसार खेत तैयार करें।')
        self.assertEqual(result['guide_cache']['status'], 'unavailable')

    def test_concurrent_writes_keep_entries_and_bound_size(self):
        cache = GuideResponseCache(self.path, max_entries=20)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: cache.put_english(str(i), {'answer': str(i), 'retrieved': []}), range(12)))
        for i in range(12):
            self.assertEqual(cache.get(str(i))['english']['answer'], str(i))
        small = GuideResponseCache(self.path, max_entries=2)
        small.put_english('new', {'answer': 'new', 'retrieved': []})
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM guide_answers').fetchone()[0], 2)
        small.clear()
        self.assertIsNone(small.get('new'))
