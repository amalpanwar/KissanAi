import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from app.advisor import AdvisorConfig, RAGAdvisor


class ProfitabilityRouteTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'test.db'
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE crop_economics (district TEXT, season TEXT, crop_name TEXT, cost_min_inr_per_acre REAL, cost_max_inr_per_acre REAL, market_price_inr_per_qtl REAL, avg_yield_qtl_per_acre REAL)')
            conn.executemany('INSERT INTO crop_economics VALUES (?,?,?,?,?,?,?)', [
                ('Baghpat', 'rabi', 'Wheat', 10000, 15000, 2000, 20),
                ('Baghpat', 'rabi', 'Mustard', 8000, 10000, 3000, 10),
                ('Meerut', 'rabi', 'OtherDistrict', 100, 200, 9000, 100),
            ])
        self.advisor = RAGAdvisor(AdvisorConfig('unused', 'unused', 'missing.npz', 'missing.csv', 3, db_path=str(path)))
        self.model = self.start(patch.object(self.advisor, '_get_generator_for_model', side_effect=AssertionError('Model must not load')))
        self.semantic = self.start(patch.object(self.advisor, '_ensure_agri_intent_semantic_index', side_effect=AssertionError('Embeddings must not load')))
        self.start(patch.object(self.advisor, '_load_agmarknet_prices', return_value={}))
        self.start(patch.object(self.advisor, '_official_cost_range_for_profitability', return_value=None))
        self.start(patch('app.advisor.load_latest_up_yield_qtl_per_acre', return_value=None))
        self.start(patch('app.advisor.get_latest_sugarcane_frp', return_value=None))
        self.start(patch('app.hindi_translation._setting', side_effect=lambda n, default='': '0' if n == 'KISAANAI_HINDI_TRANSLATION' else default))

    def start(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def ask(self, question):
        return self.advisor.answer('State: Uttar Pradesh | District: Baghpat | Season: rabi | किसान का प्रश्न: ' + question)

    def test_profit_comparisons_use_database_without_models(self):
        for question in ['konsi fasal jyada labhdayak', 'kaunsi fasal zyada laabhdayak hai',
                         'कौन सी फसल ज्यादा लाभदायक है', 'which crop is most profitable']:
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertEqual(result['agent_status'], 'ok', result['answer'])
                self.assertEqual(result['topic'], 'crop_profitability')
                self.assertEqual(result['agent_trace']['planner'], 'rules')
                self.assertIn('1) Wheat', result['answer'])
                self.assertIn('₹25000-₹30000', result['answer'])
                self.assertNotIn('OtherDistrict', result['answer'])
                self.assertIn('crop_economics (SQLite)', result['references'])
        self.model.assert_not_called()
        self.semantic.assert_not_called()

    def test_sarvam_timeout_keeps_profit_answer(self):
        with patch('app.hindi_translation._setting', side_effect=lambda n, default='': 'test-key' if n == 'SARVAM_API_KEY' else default), patch('app.hindi_translation.urlopen', side_effect=TimeoutError):
            result = self.ask('konsi fasal jyada labhdayak')
        self.assertEqual(result['agent_status'], 'ok')
        self.assertIn('1) Wheat', result['answer'])
        self.assertEqual(result['translation']['status'], 'fallback')
        self.assertEqual(result['translation']['reason'], 'TimeoutError')

    def test_partial_district_database_keeps_annual_estimate(self):
        result = self.ask('konsi fasal jyada labhdayak')
        self.assertIn('Wheat', result['answer'])
        self.assertIn('Sugarcane', result['answer'])
        self.assertIn('अतिरिक्त वार्षिक फसल तुलना', result['answer'])
        self.assertIn('crop_profit_baselines', result['references'])

    def test_explicit_season_does_not_add_annual_estimate(self):
        result = self.ask('rabi me konsi fasal jyada labhdayak')
        self.assertIn('Wheat', result['answer'])
        self.assertNotIn('Sugarcane', result['answer'])

    def test_annual_supplement_respects_budget(self):
        text, _ = self.advisor._structured_crop_recommendation(
            'District: Baghpat | Season: Rabi', 'कौन सी फसल ज्यादा लाभदायक है बजट 20000')
        self.assertNotIn('Sugarcane', text)

    def test_hinglish_normalizes_profit_words(self):
        self.assertEqual(self.advisor._normalize_hinglish('konsi fasal jyada labhdayak'), 'कौन सी फसल ज्यादा लाभदायक')


if __name__ == '__main__':
    unittest.main()
