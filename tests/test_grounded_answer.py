"""Evidence synthesis and final Hindi output, without external model downloads."""
import unittest
from unittest.mock import patch
import test_research as fixtures
from app.research import synthesize
from app.advisor import RAGAdvisor
from app.agent_system import build_coordinator


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ResearchTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.rows = [{'text': 'Plant cane in trenches 30 cm deep.', 'source_file': 'guide.pdf', 'page': 2}]

    def test_synthesis_keeps_source_text_and_page_in_prompt(self):
        self.fixture.generator.generate.return_value = 'Plant in trenches 30 cm deep. [1]'
        answer = synthesize(self.fixture.advisor, {'question': 'How to grow cane?'}, self.rows)
        self.assertIn('30 cm', answer)
        prompt = self.fixture.generator.generate.call_args.args[0]
        self.assertIn('guide.pdf', prompt)
        self.assertIn('"page": 2', prompt)

    def test_unsupported_quantities_or_citations_are_rejected(self):
        for answer in ['', 'Plant in trenches.', 'Plant 80 cm deep. [1]', 'Plant 30 cm deep. [4]']:
            with self.subTest(answer=answer):
                self.fixture.generator.generate.return_value = answer
                with self.assertRaises(ValueError):
                    synthesize(self.fixture.advisor, {'question': 'cane?'}, self.rows)

    def test_missing_generator_does_not_dump_excerpts(self):
        self.fixture.generator.generate.side_effect = RuntimeError('remote generation unavailable')
        with patch('app.multilingual_retrieval.translate_query', side_effect=fixtures.translation):
            result = build_coordinator(self.fixture.advisor).answer('किस्म बताएं', 'Rice')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertNotIn('Cultivar A', result['answer'])
        self.assertIn('उत्तर तैयार नहीं', result['answer'])

    def test_final_answer_is_translated_after_english_synthesis(self):
        advisor = RAGAdvisor.__new__(RAGAdvisor)
        advisor.__dict__.update(self.fixture.advisor.__dict__)
        hindi = 'यह किस्म जल्दी पकती है। [1]'
        def finish(result):
            self.assertIn('Cultivar A', result['answer'])
            self.assertIn('crop.pdf', result['references'])
            self.assertTrue(result['retrieved'])
            return {**result, 'answer': hindi, 'translation': {'status': 'translated'}}
        with patch('app.multilingual_retrieval.translate_query', side_effect=fixtures.translation), patch(
                'app.hindi_translation.translate_answer', side_effect=finish) as final:
            result = advisor.answer('किस प्रकार का चावल बेहतर है?')
        final.assert_called_once()
        self.assertEqual(result['answer'], hindi)

    def test_translation_failure_never_displays_english_answer(self):
        advisor = RAGAdvisor.__new__(RAGAdvisor)
        advisor.__dict__.update(self.fixture.advisor.__dict__)
        for status in ['not_configured', 'disabled', 'fallback']:
            with self.subTest(status=status), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=fixtures.translation), patch(
                    'app.hindi_translation.translate_answer',
                    side_effect=lambda result: {**result, 'translation': {'status': status}}):
                result = advisor.answer('किस प्रकार का चावल बेहतर है?')
            self.assertEqual(result['agent_status'], 'unavailable')
            self.assertNotIn('Cultivar A', result['answer'])
            self.assertIn('हिंदी अनुवाद', result['answer'])


if __name__ == '__main__': unittest.main()
