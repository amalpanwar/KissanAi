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

    def test_one_corrective_request_uses_same_evidence_and_preserves_validation(self):
        for bad, reason in [('Plant 30 cm deep.', 'missing_evidence_citations'),
                            ('Plant 30 cm deep. [4]', 'invalid_evidence_citations'),
                            ('Plant 80 cm deep. [1]', 'unsupported_quantities')]:
            with self.subTest(reason=reason):
                self.fixture.generator.generate.reset_mock()
                self.fixture.generator.generate.side_effect = [bad, 'Plant 30 cm deep. [1]']
                diagnostics = {}
                answer = synthesize(self.fixture.advisor, {'question': 'cane?'}, self.rows, diagnostics)
                self.assertEqual(answer, 'Plant 30 cm deep. [1]')
                self.assertEqual(diagnostics, {'attempts': 2, 'validation_failures': [reason]})
                prompts = [call.args[0] for call in self.fixture.generator.generate.call_args_list]
                self.assertTrue(prompts[1].startswith(prompts[0]))
                self.assertIn(reason, prompts[1])
                self.assertIn('guide.pdf', prompts[1])
                self.assertNotIn(bad, prompts[1])

    def test_repeated_validation_failure_keeps_evidence_and_specific_reason(self):
        self.fixture.generator.generate.return_value = 'Invented 999 kg. [1]'
        with patch('app.multilingual_retrieval.translate_query', side_effect=fixtures.translation):
            result = build_coordinator(self.fixture.advisor).answer('किस्म बताएं', 'Rice')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertEqual(self.fixture.generator.generate.call_count, 2)
        self.assertTrue(result['retrieved'])
        self.assertIn('crop.pdf', result['references'])
        message = next(m for m in result['agent_trace']['messages'] if m['sender'] == 'agronomy' and m['kind'] == 'result')
        generation = message['payload']['evidence']['checks']['generation']
        self.assertEqual(generation['reason'], 'unsupported_quantities')
        self.assertEqual(generation['validation_failures'], ['unsupported_quantities'] * 2)
        self.assertEqual(generation['attempts'], 2)
        self.assertNotIn('999', str(generation))
        self.assertNotIn('999', result['answer'])

    def test_transport_errors_are_not_retried_as_validation_errors(self):
        self.fixture.generator.generate.side_effect = TimeoutError('private error')
        diagnostics = {}
        with self.assertRaises(TimeoutError):
            synthesize(self.fixture.advisor, {'question': 'cane?'}, self.rows, diagnostics)
        self.assertEqual(self.fixture.generator.generate.call_count, 1)
        self.assertEqual(diagnostics, {'attempts': 1, 'validation_failures': []})

    def test_truncated_generation_has_one_recovery_attempt(self):
        self.fixture.generator.generate.side_effect = [ValueError('incomplete_generation'), 'Plant 30 cm deep. [1]']
        diagnostics = {}
        self.assertEqual(synthesize(self.fixture.advisor, {'question': 'cane?'}, self.rows, diagnostics),
                         'Plant 30 cm deep. [1]')
        self.assertEqual(diagnostics['validation_failures'], ['incomplete_generation'])

    def test_insufficient_evidence_abstains_without_corrective_retry(self):
        self.fixture.generator.generate.return_value = 'INSUFFICIENT_EVIDENCE'
        with self.assertRaisesRegex(ValueError, 'insufficient_evidence'):
            synthesize(self.fixture.advisor, {'question': 'unrelated question?'}, self.rows)
        self.assertEqual(self.fixture.generator.generate.call_count, 1)

    def test_insufficient_documents_fall_back_to_web_once(self):
        from app.research import ResearchAgent
        from app.agent_system import AgentResult
        from unittest.mock import MagicMock
        bus = MagicMock()
        web_rows = [{'text': 'Plant wheat in prepared fields.', 'source_file': 'https://tnau.ac.in/guide'}]
        bus.ask.side_effect = [AgentResult('', 'ok', evidence={'records': self.rows}),
                               AgentResult('', 'ok', evidence={'records': web_rows})]
        self.fixture.generator.generate.side_effect = ['INSUFFICIENT_EVIDENCE', 'Plant wheat in prepared fields. [1]']
        with patch('app.evidence_reranker.verify', side_effect=[(self.rows, 'ok'), (web_rows, 'ok')]):
            result = ResearchAgent().run(None, {'question': 'wheat?'}, bus)
        self.assertEqual(bus.ask.call_count, 2)
        self.assertEqual(result.status, 'partial')
        self.assertEqual(result.references, ['https://tnau.ac.in/guide'])
        self.assertEqual(result.evidence['checks']['document_generation']['reason'], 'insufficient_evidence')

    def test_insufficient_web_evidence_never_publishes_advice(self):
        from app.research import ResearchAgent
        from app.agent_system import AgentResult
        from unittest.mock import MagicMock
        bus = MagicMock()
        bus.ask.side_effect = [AgentResult('', 'ok', evidence={'records': self.rows}),
                               AgentResult('', 'ok', evidence={'records': self.rows})]
        self.fixture.generator.generate.return_value = 'INSUFFICIENT_EVIDENCE'
        with patch('app.evidence_reranker.verify', return_value=(self.rows, 'ok')):
            result = ResearchAgent().run(None, {'question': 'unrelated?'}, bus)
        self.assertEqual(bus.ask.call_count, 2)
        self.assertEqual(self.fixture.generator.generate.call_count, 2)
        self.assertEqual(result.status, 'unavailable')
        self.assertNotIn('INSUFFICIENT_EVIDENCE', result.answer)
        self.assertNotIn('Plant', result.answer)

    def test_single_source_guide_without_inline_citations_still_validates_quantities(self):
        rows = [{**self.rows[0], 'guide_overview': True, 'guide_section': 'Sugarcane'}]
        self.fixture.generator.generate.return_value = 'Plant cane in trenches 30 cm deep.'
        diagnostics = {}
        self.assertEqual(synthesize(self.fixture.advisor, {'question': 'cane?'}, rows, diagnostics),
                         'Plant cane in trenches 30 cm deep.')
        self.assertEqual(diagnostics['attempts'], 1)
        self.assertEqual(diagnostics['reference_mode'], 'app_attached_pdf_section')
        for invalid in ['Plant cane 999 cm deep.', 'Plant cane 30 cm deep. [4]']:
            with self.subTest(invalid=invalid):
                self.fixture.generator.generate.return_value = invalid
                with self.assertRaises(ValueError):
                    synthesize(self.fixture.advisor, {'question': 'cane?'}, rows)

    def test_multiple_pdf_sources_still_require_citations(self):
        rows = [{**self.rows[0], 'guide_overview': True, 'guide_section': 'Sugarcane'},
                {**self.rows[0], 'source_file': 'other.pdf', 'guide_overview': True, 'guide_section': 'Sugarcane'}]
        self.fixture.generator.generate.return_value = 'Plant cane 30 cm deep.'
        with self.assertRaisesRegex(ValueError, 'missing_evidence_citations'):
            synthesize(self.fixture.advisor, {'question': 'cane?'}, rows)

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

    def test_translation_failure_keeps_english_answer_visible(self):
        advisor = RAGAdvisor.__new__(RAGAdvisor)
        advisor.__dict__.update(self.fixture.advisor.__dict__)
        for status in ['not_configured', 'disabled', 'fallback']:
            with self.subTest(status=status), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=fixtures.translation), patch(
                    'app.hindi_translation.translate_answer',
                    side_effect=lambda result: {**result, 'translation': {'status': status}}):
                result = advisor.answer('किस प्रकार का चावल बेहतर है?')
            self.assertEqual(result['agent_status'], 'partial')
            self.assertIn('Cultivar A', result['answer'])
            self.assertEqual(result['translation']['status'], status)


if __name__ == '__main__': unittest.main()
