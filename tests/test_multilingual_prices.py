import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from app.advisor import RAGAdvisor
from app.agent_system import AgentResult

class MultilingualPriceTests(unittest.TestCase):
    def test_hindi_price_query_routes_english_to_market_then_hindi_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta=Path(tmp)/'meta.csv'
            pd.DataFrame([{'text':'Agricultural crop research.','source_file':'guide.pdf'}]).to_csv(meta,index=False)
            market=Path(tmp)/'market.csv'
            pd.DataFrame([{'state_name':'Uttar Pradesh','district_name':'Baghpat','cmdt_name':'Rice',
                'market_name':'Baraut','rep_date':'01-10-2026','model_price_wt':3100,'unit_name_price':'Rs./Quintal'}]).to_csv(market,index=False)
            advisor=RAGAdvisor.__new__(RAGAdvisor)
            advisor.cfg=SimpleNamespace(metadata_path=str(meta))
            def translate(q,**kwargs):
                return {'status':'ok','text':'What is the price of rice?' if kwargs.get('target')=='en-IN' else q}
            with patch('app.multilingual_retrieval.translate_query',side_effect=translate), \
                 patch('app.agent_tools.MARKET_CSV',market), \
                 patch('app.agent_tools.AdvisorTools.location',return_value={'state':'Uttar Pradesh','district':'Baghpat'}), \
                 patch('app.agent_tools.AdvisorTools.refresh_market',return_value=AgentResult('checked')), \
                 patch('app.commodity_lookup.load_aliases',return_value={'rice':['rice','चावल']}), \
                 patch('app.commodity_lookup.catalog_names',return_value=['Rice']), \
                 patch('app.hindi_translation.translate_answer',side_effect=lambda r:{**r,'answer':'चावल का उपलब्ध भाव: 3100 रुपये प्रति क्विंटल।'}) as output:
                result=advisor.answer('चावल का मूल्य क्या आप बता सकते हैं?')
            self.assertEqual(result['agent_trace']['plan'],['prices'])
            self.assertEqual(result['agent_trace']['backend_question'],'What is the price of rice?')
            self.assertIn('3,100.00',output.call_args.args[0]['answer'])
            self.assertIn('चावल',result['answer'])
            self.assertEqual(result['references'],['https://agmarknet.gov.in/'])
