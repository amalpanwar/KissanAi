"""Cultivation uses English document evidence and Sarvam, never template replies."""
import os
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_research as fixtures
from app.advisor import RAGAdvisor
from app.crop_guide import GUIDE_PDF, _section_page_range, _extract_section_text
from app.hindi_translation import TOKEN


class DocumentGuidePipelineTests(unittest.TestCase):
    def setUp(self):
        if not GUIDE_PDF.exists():
            if os.getenv('CI'): self.fail('Production guide must be deployed')
            self.skipTest('Guide absent locally')
        self.fixture = fixtures.ResearchTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.metadata.loc[0, 'text'] = _extract_section_text('Cumbu')[:3000]
        self.fixture.metadata.loc[0, 'source_file'] = str(GUIDE_PDF)
        self.fixture.metadata.to_csv(self.fixture.path, index=False)
        self.fixture.generator.generate.return_value = 'Pearl millet cultivation follows the source practices. [1]'
        self.advisor = RAGAdvisor.__new__(RAGAdvisor)
        self.advisor.__dict__.update(self.fixture.advisor.__dict__)

    def test_bajra_aliases_retrieve_rerank_and_translate_instead_of_rendering_templates(self):
        def query_translation(text, names=(), target='hi-IN'):
            return {'status':'ok', 'text': 'How to cultivate Bajra?' if target=='en-IN' else text}
        for question in ['Bajra ki kheti kaise kare', 'बाजरा की खेती कैसे करें?', 'How to grow pearl millet?']:
            with self.subTest(question=question), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=query_translation), patch(
                    'app.advisor.build_crop_production_guide', side_effect=AssertionError('No template answers')), patch.object(
                    self.advisor, '_get_generator_for_model', side_effect=AssertionError('No local generation')), patch(
                    'app.hindi_translation._setting', side_effect=lambda name, default='': 'test-key' if name=='SARVAM_API_KEY' else default), patch(
                    'app.hindi_translation._translate', side_effect=lambda text, key, timeout: 'बाजरा की खेती स्रोत के अनुसार करें। ' + ' '.join(TOKEN.findall(text))):
                result = self.advisor.answer(question)
            self.assertEqual(result['agent_status'], 'ok')
            self.assertIn('बाजरा', result['answer'])
            self.assertIn('pearl millet', result['agent_trace']['backend_question'].lower())
            self.assertEqual(result['agent_trace']['planner'], 'rules')
            self.assertIn(str(GUIDE_PDF), result['references'])
            self.assertEqual(result['translation']['status'], 'translated')
            self.assertEqual(result['retrieved'][0]['text'], self.fixture.metadata.loc[0, 'text'].strip())
            prompt = self.fixture.generator.generate.call_args.args[0]
            evidence = json.loads(prompt[prompt.index('{"question"'):])['evidence']
            self.assertEqual(evidence[0]['text'], self.fixture.metadata.loc[0, 'text'].strip())
            self.assertNotIn('Insecticide registration', prompt)

    def test_sugarcane_section_stops_before_sweet_sorghum(self):
        self.assertEqual(_section_page_range('Sugarcane'), (296, 318))
        text = _extract_section_text('Sugarcane')
        self.assertIn('7. SUGARCANE', text)
        self.assertNotIn('8. SWEET SORGHUM', text)

    def test_missing_documents_never_fall_back_to_crop_templates(self):
        self.advisor.retriever = None
        with patch('app.multilingual_retrieval.translate_query', side_effect=fixtures.translation), patch(
                'app.web_search.search_with_status', return_value={'status':'ok','provider':'test','results':[]}), patch(
                'app.advisor.build_crop_production_guide', side_effect=AssertionError('No template answers')):
            result = self.advisor.answer('बाजरा की खेती कैसे करें?')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertNotIn('स्टेप-बाय-स्टेप', result['answer'])

if __name__ == '__main__': unittest.main()
