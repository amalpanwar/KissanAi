"""Regression coverage for cultivation questions without model downloads."""
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

    def test_cultivation_routes_without_models(self):
        for question in ['gehu ki kheti kese kare', 'gehun ki kheti kaise karein',
                         'गेहूं की खेती कैसे करें', 'how to grow wheat']:
            with self.subTest(question=question), patch('app.advisor.build_crop_production_guide',
                    return_value=('गेहूं की खेती: स्टेप-बाय-स्टेप गाइड', ['guide.pdf'])):
                result = self.advisor.answer(question)
                self.assertEqual(result['topic'], 'crop_guide')
                self.assertEqual(result['agent_status'], 'ok')
                self.assertEqual(result['agent_trace']['planner'], 'rules')
                self.assertEqual(result['references'], ['guide.pdf'])
        self.model_mock.assert_not_called()
        self.semantic_mock.assert_not_called()

    def test_missing_guide_is_unavailable(self):
        with patch('app.advisor.build_crop_production_guide', return_value=(None, [])):
            result = self.advisor.answer('gehu ki kheti kese kare')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertNotIn('मॉडल', result['answer'])

    def test_unreadable_guide_is_unavailable(self):
        with patch('app.advisor.build_crop_production_guide', side_effect=OSError('unreadable')), \
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
        self.assertIn('गेहूं', result['answer'])
        self.assertGreater(len(result['answer']), 200)
        self.assertTrue(result['references'])
        self.model_mock.assert_not_called()
        self.semantic_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
