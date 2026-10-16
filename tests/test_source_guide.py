"""Real-document sugarcane regression: no models or translation services."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from app.advisor import RAGAdvisor, AdvisorConfig
from app.hindi_translation import TOKEN
from app.crop_guide import GUIDE_PDF, _section_page_range, _extract_section_text


class SourceGuideTests(unittest.TestCase):
    def setUp(self):
        if not GUIDE_PDF.exists() or not Path('data/raw/Trench_Planting_Sugarcane.pdf').exists():
            if os.getenv('CI'):
                self.fail('Both sugarcane source PDFs must be deployed')
            self.skipTest('Repository source PDFs absent')
        settings = patch('app.hindi_translation._setting', side_effect=lambda name, default='': 'test-key' if name == 'SARVAM_API_KEY' else default)
        settings.start(); self.addCleanup(settings.stop)
        translator = patch('app.hindi_translation._translate', side_effect=lambda text, key, timeout: 'हिंदी विवरण ' + ' '.join(TOKEN.findall(text)))
        translator.start(); self.addCleanup(translator.stop)
        generator = patch('app.sarvam_generation.generate_guide', side_effect=lambda result, question: result)
        generator.start(); self.addCleanup(generator.stop)
        self.advisor = RAGAdvisor(AdvisorConfig('unused', 'unused', 'missing', 'missing', 3))

    def test_hindi_hinglish_and_english_guide_without_local_models(self):
        for question in ['Ganne ki kheti kaise kare', 'गन्ने की खेती कैसे करें?', 'How to grow sugarcane?']:
            with self.subTest(question=question), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=AssertionError('No query translation')), patch.object(
                    self.advisor, '_ensure_rag_components', side_effect=AssertionError('No model loading')):
                result = self.advisor.answer(question)
            self.assertEqual(result['agent_status'], 'ok')
            self.assertEqual(result['topic'], 'crop_guide')
            self.assertIn('ट्रेंच विधि', result['answer'])
            self.assertIn('सिंचाई', result['answer'])
            self.assertIn('30 सेमी', result['answer'])
            self.assertIn('data/raw/Crop Production guide.pdf', result['references'])
            self.assertIn('data/raw/Trench_Planting_Sugarcane.pdf', result['references'])
            self.assertNotIn('बीज की गहराई लगभग 2 सेमी', result['answer'])
            self.assertNotIn('20-35', result['answer'])

    def test_bajra_uses_final_sarvam_translation(self):
        result = self.advisor.answer('Bajra ki kheti kaise kare')
        self.assertEqual(result['translation']['provider'], 'sarvam')
        self.assertEqual(result['translation']['status'], 'translated')
        self.assertNotIn('Apply', result['answer'])

    def test_failed_guide_translation_does_not_show_english(self):
        with patch('app.hindi_translation._translate', side_effect=TimeoutError):
            result = self.advisor.answer('Bajra ki kheti kaise kare')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertNotIn('Apply', result['answer'])
        self.assertEqual(result['translation']['status'], 'fallback')

    def test_sugarcane_section_stops_before_sweet_sorghum(self):
        self.assertEqual(_section_page_range('Sugarcane'), (296, 318))
        text = _extract_section_text('Sugarcane')
        self.assertIn('7. SUGARCANE', text)
        self.assertNotIn('8. SWEET SORGHUM', text)

    def test_bajra_aliases_use_cumbu_guide_without_models(self):
        for question in ['Bajra ki kheti kaise kare', 'बाजरा की खेती कैसे करें?',
                         'How to grow pearl millet?', 'How to grow cumbu?']:
            with self.subTest(question=question), patch.object(
                    self.advisor, '_ensure_rag_components', side_effect=AssertionError('No model loading')), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=AssertionError('No query translation')):
                result = self.advisor.answer(question)
            self.assertEqual(result['agent_status'], 'ok')
            self.assertIn('बाजरा', result['answer'])
            self.assertEqual(result['agent_trace']['backend_question'], 'Cumbu')
            self.assertEqual(result['references'], ['data/raw/Crop Production guide.pdf'])

    def test_all_guide_aliases_resolve_without_commodity_dictionary(self):
        from app.source_guide import guide_crop
        from app.crop_guide import CROP_ALIASES
        for crop, aliases in CROP_ALIASES.items():
            for alias in [crop, *aliases]:
                with self.subTest(crop=crop, alias=alias):
                    self.assertEqual(guide_crop(self.advisor, alias, ''), crop)

    def test_rice_and_wheat_guides_without_local_models(self):
        for question, label in [('Rice ki kheti kaise kare', 'धान'),
                                ('धान की खेती कैसे करें?', 'धान'),
                                ('How to grow rice?', 'धान'),
                                ('gehu ki kheti kaise kare', 'गेहूं')]:
            with self.subTest(question=question), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=AssertionError('No query translation')), patch.object(
                    self.advisor, '_ensure_rag_components', side_effect=AssertionError('No model loading')):
                result = self.advisor.answer(question)
            self.assertEqual(result['agent_status'], 'ok')
            self.assertIn(label, result['answer'])
            self.assertEqual(result['references'], ['data/raw/Crop Production guide.pdf'])
            self.assertEqual(result['agent_trace']['planner'], 'source_guide')
            self.assertNotIn('ट्रेंच विधि', result['answer'])

    def test_ssnm_values_come_from_supplied_trench_pdf(self):
        with patch.object(self.advisor, '_ensure_rag_components', side_effect=AssertionError('No model loading')):
            result = self.advisor.answer('Ganne ki trench SSNM formula batao')
        self.assertEqual(result['agent_status'], 'ok')
        self.assertIn('1000 क्विंटल/हेक्टेयर', result['answer'])
        self.assertIn('नाइट्रोजन: 207.5', result['answer'])
        self.assertIn('बोरॉन: 0.0625', result['answer'])
        self.assertIn('सीधे उर्वरक की मात्रा न मानें', result['answer'])

    def test_mixed_weather_query_still_uses_coordinator(self):
        from app.source_guide import sugarcane_guide
        self.assertIsNone(sugarcane_guide(self.advisor, 'Ganne ki kheti aur mausam batao'))

    def test_missing_sources_reports_unavailable_without_model_fallback(self):
        with patch('app.crop_guide.build_crop_production_guide', return_value=(None, [])), patch(
                'app.source_guide.TRENCH', Path('/missing-trench.json')), patch.object(
                self.advisor, '_ensure_rag_components', side_effect=AssertionError('No model loading')):
            result = self.advisor.answer('Ganne ki kheti kaise kare')
        self.assertEqual(result['agent_status'], 'unavailable')
        self.assertFalse(result['references'])

if __name__ == '__main__': unittest.main()
