import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd
from app.agent_system import build_coordinator
from app.location_query import explicit_named_place
from app.research import answer_with_research_followup
from app.web_search import WebSearchResult, search_with_status


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'metadata.csv'
        self.advisor=SimpleNamespace(cfg=SimpleNamespace(metadata_path=str(self.path)),
            _get_generator_for_model=MagicMock(side_effect=AssertionError('No model download')))

    def docs(self, text):
        with self.path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['text','source_file']);writer.writeheader()
            if text: writer.writerow({'text':text,'source_file':'soil-guide.pdf'})

    def run_question(self, remote, question='mitti ki jaanch kaise kare?'):
        with patch('app.web_search.search_with_status',return_value=remote):
            return build_coordinator(self.advisor).answer(question,'State: Uttar Pradesh | District: Baghpat |')

    def test_soil_question_checks_both_agents_and_uses_documents(self):
        self.docs('Soil test: collect soil samples and take them to the laboratory for testing.')
        result=self.run_question({'status':'ok','provider':'tavily','results':[]})
        self.assertEqual(result['agent_status'],'ok')
        self.assertIn('soil-guide.pdf',result['references'])
        self.assertIn('laboratory',result['answer'])
        pairs=[(m['sender'],m['recipient']) for m in result['agent_trace']['messages']]
        self.assertIn(('research','documents'),pairs)
        self.assertIn(('research','web_search'),pairs)
        self.advisor._get_generator_for_model.assert_not_called()

    def test_missing_documents_still_checks_tavily(self):
        result=self.run_question({'status':'ok','provider':'tavily','results':[
            WebSearchResult('Soil testing','Soil test samples are collected for laboratory testing.',
                            'https://icar.gov.in/soil-test','icar.gov.in')]})
        self.assertEqual(result['agent_status'],'partial')
        self.assertEqual(result['references'],['https://icar.gov.in/soil-test'])
        self.assertIn('दस्तावेज़ों की खोज पूरी नहीं',result['answer'])

    def test_both_sources_empty_or_unavailable_are_honest(self):
        self.docs('')
        result=self.run_question({'status':'ok','provider':'tavily','results':[]})
        self.assertEqual(result['agent_status'],'unavailable')
        self.assertFalse(result['references'])
        self.assertNotIn('मॉडल',result['answer'])
        self.path.unlink()
        result=self.run_question({'status':'unavailable','provider':'none','results':[],'reason':'search_not_configured'})
        self.assertIn('वेब खोज उपलब्ध नहीं',result['answer'])

    def test_weak_matches_fall_back_without_switching_topics(self):
        self.docs('Soil application of insecticide controls pests. Approved bio pesticide formulations.')
        for question in ['mitti ki urvarta k bare me jankari de', 'mitti ki jaanch kaise karvaye?']:
            result=self.run_question({'status':'ok','provider':'tavily','results':[]},question)
            self.assertEqual(result['agent_status'],'unavailable')
            self.assertFalse(result['references'])
            self.assertNotIn('research_offer',result)
            self.assertNotIn('soil-guide.pdf',result['answer'])

    def test_fertility_hinglish_and_hindi_search_same_concepts(self):
        from app.research import search_question, rank
        for question in ['mitti ki urvarta k bare me jankari de', 'मिट्टी की उर्वरता के बारे में जानकारी दें']:
            self.docs('Soil fertility depends on organic matter and nutrient availability.')
            result=self.run_question({'status':'ok','provider':'tavily','results':[]},question)
            self.assertEqual(result['agent_status'],'ok')
            self.assertIn('organic matter',result['answer'])
        self.assertEqual(search_question('mitti ki jaanch kaise karvaye?'),'soil test')
        self.assertEqual(search_question('mitti ki urvarta k bare me jankari de'),'soil fertility')
        self.assertEqual(rank('soil fertility',[{'source_file':'soil fertility.pdf',
            'text':'Soil application of pesticide. Fertility is discussed elsewhere.'}]),[])

    def test_new_question_discards_previous_offer(self):
        advisor=SimpleNamespace(_split_context_and_question=lambda q:('',q),answer=MagicMock(return_value={'answer':'new question'}))
        state={'pending_research_offer':{'records':[]}}
        self.assertEqual(answer_with_research_followup(advisor,'another topic',state)['answer'],'new question')
        self.assertNotIn('pending_research_offer',state)

    def test_ordinary_prose_is_not_a_village(self):
        lookup=pd.DataFrame([{'place':'Kareempur','district':'Meerut','sub_district':'Sardhana'},
                             {'place':'Doghat Rural','district':'Baghpat','sub_district':'Baraut'}])
        for q in ['mitti ki jaanch kaise kare?', 'gehu ki kheti kaise kare', 'how do I test soil?']:
            self.assertIsNone(explicit_named_place(q,lookup))
        self.assertEqual(explicit_named_place('Doghat mein mitti ki jaanch kaise kare?',lookup),'Doghat Rural')
        self.assertEqual(explicit_named_place('Kareempur weather',lookup),'Kareempur')
        self.assertIsNone(explicit_named_place('kare kaise',lookup))

    def test_tavily_timeout_and_missing_key_have_distinct_status(self):
        with patch('app.web_search._setting',return_value=''):
            self.assertEqual(search_with_status('soil test')['reason'],'search_not_configured')
        with patch('app.web_search._setting',return_value='key'),patch('app.web_search.urlopen',side_effect=TimeoutError()):
            self.assertEqual(search_with_status('soil test')['reason'],'search_request_failed')
        with patch.dict(os.environ,{'TAVILY_API_KEY':''}),patch.dict('sys.modules',{'streamlit':SimpleNamespace(secrets={'TAVILY_API_KEY':'secret'})}):
            from app.web_search import is_tavily_search_configured
            self.assertTrue(is_tavily_search_configured())


if __name__=='__main__': unittest.main()
