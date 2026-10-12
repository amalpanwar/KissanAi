"""Real-document sugarcane regression: no models or translation services."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from app.advisor import RAGAdvisor, AdvisorConfig
from app.crop_guide import GUIDE_PDF, _section_page_range, _extract_section_text


class SourceGuideTests(unittest.TestCase):
    def setUp(self):
        if not GUIDE_PDF.exists() or not Path('data/raw/Trench_Planting_Sugarcane.pdf').exists():
            if os.getenv('CI'):
                self.fail('Both sugarcane source PDFs must be deployed')
            self.skipTest('Repository source PDFs absent')
        self.advisor = RAGAdvisor(AdvisorConfig('unused', 'unused', 'missing', 'missing', 3))

    def test_hindi_hinglish_and_english_guide_without_models_or_translation(self):
        for question in ['Ganne ki kheti kaise kare', 'गन्ने की खेती कैसे करें?', 'How to grow sugarcane?']:
            with self.subTest(question=question), patch(
                    'app.multilingual_retrieval.translate_query', side_effect=AssertionError('No query translation')), patch(
                    'app.hindi_translation.translate_answer', side_effect=AssertionError('Already rendered Hindi')), patch.object(
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

    def test_sugarcane_section_stops_before_sweet_sorghum(self):
        self.assertEqual(_section_page_range('Sugarcane'), (296, 318))
        text = _extract_section_text('Sugarcane')
        self.assertIn('7. SUGARCANE', text)
        self.assertNotIn('8. SWEET SORGHUM', text)

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
