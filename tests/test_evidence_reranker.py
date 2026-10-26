import unittest
from unittest.mock import patch, MagicMock
from app.evidence_reranker import verify, bibliography


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self):
        # These unit tests mock inference; CI intentionally omits PyTorch.
        # Keep activation construction mocked too, so missing dependencies do
        # not masquerade as a model failure before predict() is exercised.
        torch = MagicMock()
        self.identity = torch.nn.Identity.return_value
        patcher = patch.dict('sys.modules', {'torch': torch})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bibliography_and_duplicate_text_never_reach_answers(self):
        ref='Jones. 2009. Journal 10(3): 311–20. Singh. 2005. Crop Protection 24:1042–53. Cohen. 2005. Nature.'
        self.assertTrue(bibliography(ref))
        model=MagicMock(); model.predict.return_value=[2.0, -3.0]
        rows=[{'text':ref}, {'text':'cultivar description'}, {'text':'cultivar description'}, {'text':'herbicide dosage'}]
        with patch('app.evidence_reranker.model',return_value=model):
            accepted,status=verify({'question':'Which rice variety is better?'},rows)
        self.assertEqual(status,'ok')
        self.assertEqual([r['text'] for r in accepted],['cultivar description', 'herbicide dosage'])
        self.assertEqual(len(model.predict.call_args.args[0]),2)
        self.assertIs(model.predict.call_args.kwargs['activation_fct'], self.identity)

    def test_negative_logits_are_ranked_not_discarded(self):
        model = MagicMock()
        model.predict.return_value = [-3.44, -8.77, -3.47, -5.51]
        rows = [{'text': text} for text in ['wheat field preparation', 'onion prices', 'wheat irrigation', 'rice nursery']]
        with patch('app.evidence_reranker.model', return_value=model):
            accepted, status = verify({'question': 'How to do wheat farming?'}, rows)
        self.assertEqual(status, 'ok')
        self.assertEqual([r['text'] for r in accepted], ['wheat field preparation', 'wheat irrigation', 'rice nursery'])
        self.assertEqual(accepted[0]['relevance_score'], -3.44)

    def test_ranking_excludes_alias_annotations_but_preserves_followup_crop(self):
        model = MagicMock(); model.predict.return_value = [-1.0]
        rows = [{'text': 'wheat field preparation', 'search_query': 'How to do wheat farming?\nCrop synonyms: Wheat; gehoo; gehu; gehun\nCrop context: Wheat'}]
        with patch('app.evidence_reranker.model', return_value=model):
            verify({'question': 'गेहूं की खेती कैसे करें?'}, rows)
        self.assertEqual(model.predict.call_args.args[0][0][0], 'How to do wheat farming?')
        rows[0]['search_query'] = 'When should I irrigate?\nCrop context: Wheat'
        with patch('app.evidence_reranker.model', return_value=model):
            verify({'question': 'When should I irrigate?'}, rows)
        self.assertEqual(model.predict.call_args.args[0][0][0], 'When should I irrigate?\nCrop: Wheat')

    def test_unavailable_or_invalid_model_fails_closed(self):
        for value in [None, [float('nan')], []]:
            model=MagicMock()
            if value is None:
                model.predict.side_effect=RuntimeError('unavailable')
            else:
                model.predict.return_value=value
            with patch('app.evidence_reranker.model',return_value=model):
                rows,status=verify({'question':'rice?'},[{'text':'potentially unrelated'}])
            self.assertEqual(rows,[])
            self.assertEqual(status,'unavailable')
            model.predict.assert_called_once()

    def test_rejected_documents_trigger_web_search(self):
        from app.research import ResearchAgent
        from app.agent_system import AgentResult
        bus=MagicMock();bus.ask.side_effect=[AgentResult('', 'ok', evidence={'records':[{'text':'irrelevant'}]}), AgentResult('', 'ok', evidence={'records':[]})]
        with patch('app.evidence_reranker.verify',return_value=([],'ok')):
            result=ResearchAgent().run(None,{'question':'rice?'},bus)
        self.assertEqual(bus.ask.call_count,2)
        self.assertNotIn('irrelevant',result.answer)

if __name__=='__main__': unittest.main()
