"""Actual-PDF routing and coverage; provider responses are mocked."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from app.production_guide_evidence import lookup, GUIDE, catalog, resolve_section, with_overview_coverage
from app.commodity_lookup import load_aliases
from app.evidence_reranker import verify as real_verify
from app.advisor import RAGAdvisor
from app.hindi_translation import TOKEN
import test_research as fixtures


class GuideSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GUIDE.exists():
            raise unittest.SkipTest('Production PDF absent')
        stat = GUIDE.stat()
        cls.pages, cls.sections = catalog(str(GUIDE), stat.st_mtime_ns, stat.st_size)

    def test_aliases_resolve_actual_sections_and_boundaries(self):
        cases = [('गेहूं की खेती कैसे करें?', 'WHEAT', (133, 135)),
                 ('बाजरे की खेती कैसे करें?', 'Cumbu', (92, 102)),
                 ('Bajra ki kheti kaise kare', 'Cumbu', (92, 102)),
                 ('How to grow pearl millet?', 'Cumbu', (92, 102)),
                 ('Rice ki kheti kaise kare', 'RICE', (12, 75)),
                 ('How to grow sugarcane?', 'SUGARCANE', (296, 318)),
                 ('How to grow maize?', 'Maize', (114, 127)),
                 ('cotton', 'Cotton', (263, 294))]
        for question, expected, bounds in cases:
            with self.subTest(question=question):
                name = resolve_section(question, self.sections, load_aliases())
                self.assertEqual(name, expected)
                self.assertEqual(self.sections[name], bounds)

    def test_multi_crop_does_not_silently_select_one(self):
        self.assertIsNone(resolve_section('Compare wheat and pearl millet', self.sections, load_aliases()))

    def test_alias_json_controls_new_mappings_without_code_changes(self):
        aliases = {'test crop': ['farmer nickname', 'Cumbu']}
        self.assertEqual(resolve_section('farmer nickname farming', self.sections, aliases), 'Cumbu')

    def test_source_section_does_not_require_vector_index_or_models(self):
        advisor = SimpleNamespace(_extract_preferred_crop_from_context=lambda c: None)
        for question in ['गेहूं की खेती कैसे करें?', 'बाजरे की खेती कैसे करें?']:
            result = lookup(advisor, {'question': question, 'original_question': question})
            self.assertEqual(result.status, 'ok')
            scope = result.evidence['guide_scope']
            for row in result.evidence['records']:
                self.assertEqual(row['source_file'], str(GUIDE))
                self.assertTrue(scope['pdf_pages'][0] <= row['page'] <= scope['pdf_pages'][1])
            coverage = with_overview_coverage(result.evidence['records'][:3], result.evidence['records'])
            self.assertEqual(len(coverage), len(result.evidence['records']))
            self.assertTrue(any('HARVEST' in row['text'].upper() for row in coverage))

    def test_explicit_crop_wins_over_old_context(self):
        advisor = SimpleNamespace(_extract_preferred_crop_from_context=lambda c: 'Wheat')
        result = lookup(advisor, {'question': 'Bajra farming?', 'original_question': 'Bajra farming?', 'context': 'Wheat'})
        self.assertEqual(result.evidence['guide_scope']['section'], 'Cumbu')

    def test_missing_or_unmatched_guide_uses_general_retrieval(self):
        advisor = SimpleNamespace(_extract_preferred_crop_from_context=lambda c: None)
        self.assertIsNone(lookup(advisor, {'question': 'unknown crop'}))
        with patch('app.production_guide_evidence.GUIDE', GUIDE.with_name('missing.pdf')):
            self.assertIsNone(lookup(advisor, {'question': 'wheat'}))

    def test_focused_question_keeps_focused_ranking(self):
        advisor = SimpleNamespace(_extract_preferred_crop_from_context=lambda c: None)
        result = lookup(advisor, {'question': 'When should I irrigate wheat?'})
        self.assertEqual(result.evidence['guide_scope']['mode'], 'focused')
        ranked = result.evidence['records'][:3]
        self.assertEqual(with_overview_coverage(ranked, result.evidence['records']), ranked)

    def test_oversized_section_is_explicit_not_silently_truncated(self):
        advisor = SimpleNamespace(_extract_preferred_crop_from_context=lambda c: None)
        with patch('app.production_guide_evidence.MAX_SECTION_CHARACTERS', 1):
            result = lookup(advisor, {'question': 'How to grow wheat?'})
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.evidence['reason'], 'guide_section_too_large')
        self.assertEqual(result.evidence['records'], [])

    def test_complete_pipeline_sends_whole_resolved_section_then_translates(self):
        fixture = fixtures.ResearchTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.guide_lookup.side_effect = lookup
        advisor = RAGAdvisor.__new__(RAGAdvisor)
        advisor.__dict__.update(fixture.advisor.__dict__)
        for question, section in [('गेहूं की खेती कैसे करें?', 'WHEAT'), ('बाजरे की खेती कैसे करें?', 'Cumbu')]:
            with self.subTest(question=question):
                fixture.generator.generate.return_value = 'Cultivation follows the source guide. [1]'
                model = MagicMock()
                model.predict.side_effect = lambda pairs, **kw: np.linspace(-1, -5, len(pairs))
                def query_translation(text, names=(), target='hi-IN'):
                    return {'status': 'ok', 'text': 'How to grow ' + section + '?' if target == 'en-IN' else text}
                with patch('app.evidence_reranker.verify', side_effect=real_verify), patch(
                        'app.evidence_reranker.model', return_value=model), patch.dict('sys.modules', {'torch': MagicMock()}), patch(
                        'app.multilingual_retrieval.translate_query', side_effect=query_translation), patch(
                        'app.hindi_translation._setting', side_effect=lambda name, default='': 'test-key' if name == 'SARVAM_API_KEY' else default), patch(
                        'app.hindi_translation._translate', side_effect=lambda text, key, timeout: 'फसल की खेती स्रोत के अनुसार करें। ' + ' '.join(TOKEN.findall(text))), patch(
                        'app.web_search.search_with_status') as web:
                    result = advisor.answer(question)
                self.assertEqual(result['translation']['status'], 'translated')
                self.assertEqual(result['agent_status'], 'ok')
                self.assertIn('फसल', result['answer'])
                self.assertTrue(all(r['guide_section'] == section for r in result['retrieved']))
                self.assertGreater(len(result['retrieved']), 3)
                self.assertTrue(any('HARVEST' in r['text'].upper() for r in result['retrieved']))
                self.assertEqual(fixture.generator.generate.call_args.kwargs['timeout'], 45)
                advisor._ensure_rag_components.assert_not_called()
                web.assert_not_called()
