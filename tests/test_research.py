import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app.agent_system import build_coordinator
from app.retriever import Retriever
from app.multilingual_retrieval import prepare_query, retrieve_documents
from app.research import answer_with_research_followup


def translation(text,names=(),target='hi-IN'):
    return {'status':'ok','text':'What cultivars suit this crop?' if target=='en-IN' else 'इस फसल की कौन सी किस्में उपयुक्त हैं?'}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'metadata.csv'
        self.metadata=pd.DataFrame([
            {'text':'Cultivar A matures early; cultivar B needs a longer season.','source_file':'crop.pdf'},
            {'text':'Insecticide registration restrictions.','source_file':'pesticides.pdf'},
            {'text':'यह किस्म जल्दी पकती है। दूसरी किस्म देर से पकती है।','source_file':'hindi.pdf'}])
        self.metadata.to_csv(self.path,index=False)
        self.embedder=MagicMock();self.embedder.encode.side_effect=lambda texts:np.array([[1.,0.] for _ in texts])
        self.advisor=SimpleNamespace(cfg=SimpleNamespace(metadata_path=str(self.path)),embedder=self.embedder,
            retriever=Retriever(np.array([[1.,0.],[0.,1.],[.99,.01]]),self.metadata),
            _ensure_rag_components=MagicMock(),_extract_preferred_crop_from_context=lambda c:'Rice' if c else None,
            _extract_crop_from_query=lambda q:None,_has_profitability_terms=lambda q:False,_has_crop_guide_terms=lambda q:False)

    def test_language_specific_semantic_queries_not_word_overlap(self):
        with patch('app.multilingual_retrieval.translate_query',side_effect=translation):
            q,info=prepare_query(self.advisor,'kaunsi achhi hai','Rice')
            result=retrieve_documents(self.advisor,dict(question=q,context='Rice',**info))
        sources={r['source_file'] for r in result.evidence['records']}
        self.assertEqual(sources,{'crop.pdf','hindi.pdf'})
        calls=[c.args[0][0] for c in self.embedder.encode.call_args_list]
        self.assertTrue(any('What cultivars' in c for c in calls))
        self.assertTrue(any('कौन सी किस्में' in c for c in calls))
        self.assertTrue(all('Rice' in c for c in calls))

    def test_hindi_only_corpus_skips_english_translation(self):
        self.metadata.iloc[[2]].to_csv(self.path,index=False)
        with patch('app.multilingual_retrieval.translate_query',side_effect=translation) as translate:
            _,info=prepare_query(self.advisor,'इसकी किस्म बताएं','')
        self.assertEqual(translate.call_count,1)
        self.assertNotIn('en-IN',info['query_variants'])

    def test_documents_answer_before_web_and_keep_citations(self):
        with patch('app.multilingual_retrieval.translate_query',side_effect=translation), patch('app.web_search.search_with_status') as search:
            result=build_coordinator(self.advisor).answer('किस प्रकार का चावल बेहतर है?','Rice')
        self.assertIn('Cultivar A',result['answer'])
        self.assertNotIn('Insecticide',result['answer'])
        self.assertIn('crop.pdf',result['references'])
        search.assert_not_called()

    def test_missing_index_reports_failure_then_checks_web(self):
        self.advisor.retriever=None
        with patch('app.multilingual_retrieval.translate_query',side_effect=translation), patch('app.web_search.search_with_status',return_value={'status':'ok','provider':'tavily','results':[]}) as search:
            result=build_coordinator(self.advisor).answer('किस्म बताएं','Rice')
        self.assertEqual(result['agent_status'],'unavailable')
        self.assertIn('खोज या आवश्यक अनुवाद पूरा नहीं',result['answer'])
        search.assert_called_once()

    def test_failed_english_translation_does_not_search_wrong_partition(self):
        self.metadata.iloc[:2].to_csv(self.path,index=False)
        self.advisor.retriever=Retriever(np.array([[1.,0.],[0.,1.]]),self.metadata.iloc[:2])
        def failed(q,**kw):
            return {'status':'unavailable','text':q} if kw.get('target')=='en-IN' else {'status':'unchanged','text':q}
        with patch('app.multilingual_retrieval.translate_query',side_effect=failed):
            q,info=prepare_query(self.advisor,'किस्म बताएं','')
            result=retrieve_documents(self.advisor,dict(question=q,**info))
        self.assertEqual(result.status,'partial')
        self.assertEqual(result.evidence['records'],[])
        self.embedder.encode.assert_not_called()

    def test_old_offer_cannot_override_new_query(self):
        advisor=SimpleNamespace(answer=MagicMock(return_value={'answer':'new'}))
        state={'pending_research_offer':{'records':[]}}
        self.assertEqual(answer_with_research_followup(advisor,'new question',state),{'answer':'new'})
        self.assertNotIn('pending_research_offer',state)
