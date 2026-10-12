import unittest
from unittest.mock import patch, MagicMock
from app.evidence_reranker import verify, bibliography


class EvidenceVerificationTests(unittest.TestCase):
    def test_bibliography_and_duplicate_text_never_reach_answers(self):
        ref='Jones. 2009. Journal 10(3): 311–20. Singh. 2005. Crop Protection 24:1042–53. Cohen. 2005. Nature.'
        self.assertTrue(bibliography(ref))
        model=MagicMock(); model.predict.return_value=[2.0, -3.0]
        rows=[{'text':ref}, {'text':'cultivar description'}, {'text':'cultivar description'}, {'text':'herbicide dosage'}]
        with patch('app.evidence_reranker.model',return_value=model):
            accepted,status=verify({'question':'Which rice variety is better?'},rows)
        self.assertEqual(status,'ok')
        self.assertEqual([r['text'] for r in accepted],['cultivar description'])
        self.assertEqual(len(model.predict.call_args.args[0]),2)

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

    def test_rejected_documents_trigger_web_search(self):
        from app.research import ResearchAgent
        from app.agent_system import AgentResult
        bus=MagicMock();bus.ask.side_effect=[AgentResult('', 'ok', evidence={'records':[{'text':'irrelevant'}]}), AgentResult('', 'ok', evidence={'records':[]})]
        with patch('app.evidence_reranker.verify',return_value=([],'ok')):
            result=ResearchAgent().run(None,{'question':'rice?'},bus)
        self.assertEqual(bus.ask.call_count,2)
        self.assertNotIn('irrelevant',result.answer)

if __name__=='__main__': unittest.main()
