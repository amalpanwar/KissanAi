"""Crop follow-ups use the same source retrieval as other agronomy questions."""
import unittest
from unittest.mock import patch
import test_research as fixtures
from app.agent_system import build_coordinator

class CropFollowupRoutingTests(unittest.TestCase):
    def test_followup_retains_crop_in_document_search(self):
        fixture=fixtures.ResearchTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        with patch('app.multilingual_retrieval.translate_query',side_effect=fixtures.translation):
            result=build_coordinator(fixture.advisor).answer('इसी फसल की किस्म बताएं','Rice')
        self.assertIn('crop.pdf',result['references'])
        self.assertTrue(any('Rice' in c.args[0][0] for c in fixture.embedder.encode.call_args_list))
