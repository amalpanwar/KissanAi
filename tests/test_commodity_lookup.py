import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from app.advisor import RAGAdvisor
from app.agent_system import Coordinator, PriceAgent, ToolAgent, AgentResult
from app.agent_tools import AdvisorTools
from app.commodity_lookup import resolve_commodities, matching_market_names

ALIASES = {'tomato': ['tomato', 'tamatar', 'टमाटर'], 'onion':['onion','pyaz','प्याज'],
           'rice':['rice','chawal','चावल'], 'wheat':['wheat','gehu','गेहूं'],
           'chili red':['chilli','lal mirch'], 'green chilli':['green chilli','hari mirch','हरी मिर्च'],
           'apple':['apple','seb'], 'pineapple':['pineapple','ananas'],
           'sugarcane':['sugarcane','ganne','गन्ने']}


class CommodityLookupTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('app.commodity_lookup.load_aliases', return_value=ALIASES)
        patcher.start(); self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'prices.csv'
        self.tools = AdvisorTools(RAGAdvisor.__new__(RAGAdvisor))
        self.location = {'state':'Uttar Pradesh','district':'Baghpat','place':'Asara'}
        for patcher in [patch('app.agent_tools.MARKET_CSV',self.path),
                        patch('app.commodity_lookup.catalog_names', return_value=list(ALIASES)),
                        patch.object(self.tools,'location',return_value=self.location)]:
            patcher.start(); self.addCleanup(patcher.stop)
        today = datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()
        self.rows = [{'state_name':'Uttar Pradesh','district_name':'Baghpat','cmdt_name':name,
                      'market_name':None,'rep_date':today,'model_price_wt':price,'unit_name_price':'Rs./Quintal'}
                     for name,price in [('Tomato',1800),('Onion',2100),('Rice',3000),('Dragon Fruit',9000)]]
        pd.DataFrame(self.rows).to_csv(self.path,index=False)

    def answer(self, question):
        return self.tools.prices('', {'question':question})

    def test_tomato_variants_through_coordinator_without_model(self):
        agents = {'prices':PriceAgent(self.tools.prices),
                  'market_data':ToolAgent(lambda g,p:AgentResult('checked'))}
        for q in ['tomato price','tamatar ka bhav','टमाटर का भाव बताएं']:
            result = Coordinator(agents, planner=lambda _: self.fail('No model needed')).answer(q)
            self.assertEqual(result['agent_status'],'ok',result)
            self.assertIn('1,800.00',result['answer'])
            self.assertNotIn('3,000.00',result['answer'])

    def test_new_catalog_names_and_word_boundaries(self):
        self.assertIn('9,000.00',self.answer('dragon fruit price').answer)
        self.assertEqual(resolve_commodities('price agriculture', aliases=ALIASES),[])
        self.assertEqual(resolve_commodities('pineapple price', aliases=ALIASES),['Pineapple'])
        self.assertEqual(resolve_commodities('hari mirch price', aliases=ALIASES),['Green Chilli'])
        self.assertEqual(matching_market_names(['Rice'],['Rice','Rice Bran'],aliases=ALIASES),{'Rice'})

    def test_multiple_crops_and_missing_crop(self):
        result = self.answer('tomato and onion price')
        self.assertEqual({r['commodity'] for r in result.evidence['records']},{'Tomato','Onion'})
        result = self.answer('tomato and wheat price')
        self.assertEqual(result.status,'partial')
        self.assertIn('Wheat',result.answer)
        result = self.answer('latest kiwi price')
        self.assertEqual(result.status,'needs_input')
        self.assertNotIn('1,800',result.answer)
        self.assertEqual(self.answer('gehu ka price').status,'unavailable')

    def test_bad_files_and_values_are_explicit_not_exceptions(self):
        for content in ['', 'wrong,columns\na,b', 'a,b\n"unterminated']:
            self.path.write_text(content)
            self.assertEqual(self.answer('tomato price').status,'unavailable')
        self.path.unlink()
        self.assertEqual(self.answer('tomato price').status,'unavailable')
        for price, day in [('inf','2026-01-01'),(-10,'2026-01-01'),(300,'2099-01-01'),(300,'bad')]:
            pd.DataFrame([{**self.rows[0],'model_price_wt':price,'rep_date':day}]).to_csv(self.path,index=False)
            self.assertEqual(self.answer('tomato price').status,'unavailable')

    def test_stale_requested_crop_not_hidden_by_fresh_other_crop(self):
        self.rows[0]['rep_date']='2020-01-01'
        pd.DataFrame(self.rows).to_csv(self.path,index=False)
        result = self.answer('tomato and onion price')
        self.assertEqual(len(result.evidence['records']),2)
        self.assertIn('पुराना रिकॉर्ड',result.answer)

    def test_unknown_or_missing_dictionary_falls_back_to_catalog_name(self):
        with patch('app.commodity_lookup.load_aliases',return_value={}):
            self.assertIn('1,800.00',self.answer('tomato price').answer)
            self.assertEqual(self.answer('tamatar price').status,'needs_input')


class ProductionDictionaryTests(unittest.TestCase):
    def test_every_catalogued_commodity_is_recognized_by_name(self):
        from app.commodity_lookup import load_aliases, normalize
        aliases = load_aliases()
        if not aliases:
            import os
            if os.getenv('CI'):
                self.fail('Deployment must include the commodity alias dictionary')
            self.skipTest('Production dictionary absent from local checkout')
        for name in aliases:
            with self.subTest(commodity=name):
                found = resolve_commodities(name + ' price', aliases=aliases)
                self.assertIn(normalize(name), {normalize(item) for item in found})


if __name__ == '__main__':
    unittest.main()
