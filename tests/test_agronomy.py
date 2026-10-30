"""Regression coverage for crop guide retrieval and generation fallback."""
import os
import unittest
from unittest.mock import patch

from app.advisor import RAGAdvisor, AdvisorConfig
from app import crop_guide


class AgronomyTests(unittest.TestCase):
    def setUp(self):
        self.advisor = RAGAdvisor(AdvisorConfig(
            'unused-embedding', 'unused-generator', '/tmp/missing-index.npz',
            '/tmp/missing-metadata.csv', 3))
        self.model = patch.object(self.advisor, '_get_generator_for_model',
                                 side_effect=AssertionError('No model needed'))
        self.semantic = patch.object(self.advisor, '_ensure_agri_intent_semantic_index',
                                    side_effect=AssertionError('No embeddings needed'))
        self.model_mock = self.model.start()
        self.semantic_mock = self.semantic.start()
        self.addCleanup(self.model.stop)
        self.addCleanup(self.semantic.stop)
        self.local_generator = patch('app.advisor.LocalGenerator',
                                     side_effect=RuntimeError('model unavailable'))
        self.local_generator_mock = self.local_generator.start()
        self.addCleanup(self.local_generator.stop)

    def test_cultivation_routes_without_models(self):
        for question in ['gehu ki kheti kese kare', 'gehun ki kheti kaise karein',
                         'गेहूं की खेती कैसे करें', 'how to grow wheat']:
            chunks = [{"text": "Wheat guide evidence: apply fertilizer 40 kg.", "page": 1,
                       "source_file": "guide.pdf"}]
            with self.subTest(question=question), patch(
                    'app.advisor.build_crop_production_source_chunks', return_value=chunks):
                result = self.advisor.answer(question)
                self.assertEqual(result['topic'], 'crop_guide')
                self.assertEqual(result['agent_status'], 'ok')
                self.assertEqual(result['agent_trace']['planner'], 'rules')
                self.assertEqual(result['references'], ['guide.pdf'])
                self.assertIn('Wheat guide evidence', result['answer'])
                self.assertIn('PDF page 1', result['answer'])
                self.assertEqual(result['answer_generation']['status'], 'source_excerpt_fallback')
        self.model_mock.assert_not_called()
        self.semantic_mock.assert_not_called()

    def test_missing_guide_is_unavailable(self):
        with patch('app.advisor.build_crop_production_source_chunks', return_value=[]):
            result = self.advisor.answer('gehu ki kheti kese kare')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertNotIn('मॉडल', result['answer'])

    def test_unreadable_guide_is_unavailable(self):
        with patch('app.advisor.build_crop_production_source_chunks', side_effect=OSError('unreadable')), \
                self.assertLogs('app.advisor', level='ERROR'):
            result = self.advisor.answer('gehu ki kheti kese kare')
        self.assertEqual(result['agent_status'], 'unavailable')

    def test_repository_wheat_guide_without_models(self):
        if not crop_guide.GUIDE_PDF.exists():
            if os.getenv('CI'):
                self.fail('Deployment must include the crop production guide PDF')
            self.skipTest('Guide PDF absent from this local checkout; required in CI')
        result = self.advisor.answer('gehu ki kheti kese kare')
        self.assertEqual(result['agent_status'], 'ok', result['answer'])
        self.assertEqual(result['topic'], 'crop_guide')
        self.assertIn('PDF page', result['answer'])
        self.assertGreater(len(result['answer']), 200)
        self.assertTrue(result['references'])
        self.assertEqual(result['answer_generation']['status'], 'source_excerpt_fallback')
        self.model_mock.assert_not_called()
        self.semantic_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
