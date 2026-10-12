"""Cultivation uses translated retrieval and reranking, never crop templates."""
import unittest
from unittest.mock import patch
import test_research as fixtures
from app.agent_system import build_coordinator


class AgronomyTests(unittest.TestCase):
    def test_hindi_sugarcane_uses_english_retrieval_and_synthesis(self):
        for english in ['How is sugarcane farming done?', 'How to cultivate sugarcane?']:
            with self.subTest(english=english):
                fixture = fixtures.ResearchTests(); fixture.setUp()
                try:
                    def translate(q, **kw):
                        return {'status': 'ok', 'text': english if kw.get('target') == 'en-IN' else q}
                    fixture.metadata.loc[0, 'text'] = 'Plant sugarcane in trenches 30 cm deep and 90 cm apart.'
                    fixture.metadata.to_csv(fixture.path, index=False)
                    fixture.generator.generate.return_value = 'Plant sugarcane in trenches 30 cm deep and 90 cm apart. [1]'
                    with patch('app.multilingual_retrieval.translate_query', side_effect=translate), patch(
                            'app.advisor.build_crop_production_guide', side_effect=AssertionError('No templates')):
                        result = build_coordinator(fixture.advisor).answer('गन्ने की खेती कैसे करें?')
                    self.assertEqual(result['agent_trace']['plan'], ['agronomy'])
                    self.assertEqual(result['agent_status'], 'ok')
                    self.assertIn('crop.pdf', result['references'])
                    self.assertTrue(result['requires_hindi'])
                    self.assertTrue(any(english in c.args[0][0] for c in fixture.embedder.encode.call_args_list))
                    prompt = fixture.generator.generate.call_args.args[0]
                    self.assertIn(english, prompt)
                    self.assertIn('30 cm', prompt)
                    self.assertNotIn('Insecticide', prompt)
                    self.assertIn('relevance', str(result['agent_trace']))
                finally:
                    fixture.doCleanups()


if __name__ == '__main__': unittest.main()
