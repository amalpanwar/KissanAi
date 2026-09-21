import unittest
from unittest.mock import patch
from app.advisor import AdvisorConfig, RAGAdvisor, WESTERN_UP_CROP_BASELINES
import test_profitability_route as profitability_tests


class AnnualBaselineTests(unittest.TestCase):
    def setUp(self):
        self.advisor = RAGAdvisor(AdvisorConfig('unused', 'unused', 'missing', 'missing', 3))
        for name, value in [('_official_cost_range_for_profitability', None), ('_estimate_pesticide_pressure', 'अज्ञात')]:
            p = patch.object(self.advisor, name, return_value=value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(self.advisor, '_market_price_for_crop', side_effect=lambda crop, base, prices: {'price':base['fallback_price'], 'source':'test baseline', 'date':''})
        p.start(); self.addCleanup(p.stop)
        p = patch('app.advisor.load_latest_up_yield_qtl_per_acre', return_value=None)
        self.yields = p.start(); self.addCleanup(p.stop)

    def rank(self, season='Rabi', question='कौन सी फसल ज्यादा लाभदायक है?'):
        return self.advisor._rank_from_profit_baselines('Baghpat', season, question, {})

    def test_generic_question_includes_annual_crop_with_rabi(self):
        text = self.rank()
        self.assertIn('Sugarcane', text)
        self.assertIn('Wheat', text)
        self.assertIn('वार्षिक/लंबी अवधि', text)
        self.assertIn('प्रति फसल चक्र', text)
        self.yields.assert_any_call('Sugarcane', season='Annual')

    def test_explicit_rabi_excludes_annual_crops(self):
        text = self.rank(question='रबी में कौन सी फसल ज्यादा लाभदायक है?')
        self.assertNotIn('Sugarcane', text)
        self.assertIn('Wheat', text)

    def test_query_season_overrides_selected_season(self):
        text = self.rank(question='खरीफ में कौन सी फसल ज्यादा लाभदायक है?')
        self.assertIn('Paddy', text)
        self.assertNotIn('Wheat', text)
        self.assertNotIn('Sugarcane', text)

    def test_no_top_five_truncation_for_all_seasons(self):
        text = self.rank(season='')
        for crop in WESTERN_UP_CROP_BASELINES:
            self.assertIn(crop, text)

    def test_season_crop_lists_remain_strict(self):
        self.assertFalse(self.advisor._season_matches('Rabi', 'Annual'))


class AnnualDatabaseTests(unittest.TestCase):
    setUp = profitability_tests.ProfitabilityRouteTests.setUp
    start = profitability_tests.ProfitabilityRouteTests.start
    ask = profitability_tests.ProfitabilityRouteTests.ask

    def test_annual_database_row_included_for_general_comparison(self):
        import sqlite3
        with sqlite3.connect(self.advisor.cfg.db_path) as conn:
            conn.execute('INSERT INTO crop_economics VALUES (?,?,?,?,?,?,?)', ('Baghpat','Annual','Sugarcane',85000,115000,380,320))
        result = self.ask('konsi fasal jyada labhdayak')
        self.assertIn('Sugarcane', result['answer'])
        self.assertIn('वार्षिक/लंबी अवधि', result['answer'])

    def test_explicit_rabi_excludes_annual_database_row(self):
        import sqlite3
        with sqlite3.connect(self.advisor.cfg.db_path) as conn:
            conn.execute('INSERT INTO crop_economics VALUES (?,?,?,?,?,?,?)', ('Baghpat','Annual','Sugarcane',85000,115000,380,320))
        result = self.ask('rabi me konsi fasal jyada labhdayak')
        self.assertNotIn('Sugarcane', result['answer'])
        self.assertIn('Wheat', result['answer'])


if __name__ == '__main__':
    unittest.main()
