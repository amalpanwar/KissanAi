import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from app.advisor import RAGAdvisor
from app.agent_tools import AdvisorTools
from app.sugarcane_prices import sugarcane_price_result


class CanePricesTests(unittest.TestCase):
    def setUp(self):
        self.tools = AdvisorTools(RAGAdvisor.__new__(RAGAdvisor))
        self.location = {'state': 'Uttar Pradesh', 'district': 'Baghpat', 'place': 'Asara'}

    def test_hinglish_hindi_english_cane_with_missing_market_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch('app.agent_tools.MARKET_CSV', Path(tmp)/'missing.csv'), \
             patch.object(self.tools, 'location', return_value=self.location):
            for question in ['ganne ka price btaye', 'ganna ka bhav', 'गन्ने का भाव बताएं', 'sugarcane price', 'ganne ka msp']:
                with self.subTest(question=question):
                    result = self.tools.prices('', {'question': question})
                    self.assertIn('₹400/क्विंटल', result.answer)
                    self.assertIn('₹365/क्विंटल', result.answer)
                    self.assertIn('Asara', result.answer)
                    self.assertEqual(len(result.references), 2)
                    self.assertNotIn('Agmarknet में उपलब्ध मंडी भाव', result.answer)

    def test_seasons_and_state_scope(self):
        result = sugarcane_price_result(self.location, date(2026,10,6))
        self.assertEqual(result.status, 'partial')
        self.assertIn('पुराने सत्र का रिकॉर्ड', result.answer)
        self.assertIn('2026-27 की उत्तर प्रदेश SAP अधिसूचना', result.answer)
        future = sugarcane_price_result(self.location, date(2028,10,1))
        self.assertEqual(future.status, 'stale')
        self.assertNotIn('इस सत्र की दर', future.answer)
        other = sugarcane_price_result({'state':'Haryana'}, date(2026,10,6))
        self.assertNotIn('₹400', other.answer)
        self.assertEqual(len(other.references), 1)
        upcoming = sugarcane_price_result(self.location, date(2026,6,1))
        self.assertIn('आगामी सत्र की घोषित दर', upcoming.answer)

    def test_price_word_does_not_mean_rice(self):
        for question, expected in [('price btaye', None), ('agriculture', None),
                                   ('rice price', 'Rice'), ('potato price', 'Potato'),
                                   ('sarso price', 'Mustard'), ('ganne price', 'Sugarcane')]:
            with self.subTest(question=question):
                self.assertEqual(self.tools.advisor._extract_crop_from_query(question), expected)

    def test_missing_market_name_does_not_drop_valid_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'market.csv'
            pd.DataFrame([{'state_name':'Uttar Pradesh', 'district_name':'Baghpat',
                'cmdt_name':'Wheat', 'market_name':None, 'rep_date':'01-01-2025',
                'model_price_wt':2500, 'unit_name_price':'Rs./Quintal'}]).to_csv(path,index=False)
            with patch('app.agent_tools.MARKET_CSV', path), patch.object(self.tools, 'location', return_value=self.location):
                result = self.tools.prices('', {'question':'gehu ka price'})
            self.assertEqual(len(result.evidence['records']), 1)
            self.assertIn('2,500.00', result.answer)
            self.assertIn('मंडी का नाम उपलब्ध नहीं', result.answer)
            self.assertEqual(result.status, 'stale')


if __name__ == '__main__':
    unittest.main()
