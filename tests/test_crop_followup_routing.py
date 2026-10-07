import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from app.agent_system import build_coordinator
from app.crop_guide import build_crop_production_followup


class CropFollowupRoutingTests(unittest.TestCase):
    def advisor(self):
        return SimpleNamespace(_is_crop_guide_followup_intent=lambda q:True,
            _normalize_hinglish=lambda q:q,
            _extract_crop_from_query=lambda q:'Wheat' if 'गेहूं' in q else None,
            _extract_preferred_crop_from_context=lambda c:'Rice' if 'Rice' in c else None,
            _get_generator_for_model=MagicMock(side_effect=AssertionError('No generator')))

    def test_rice_followup_routes_to_documents_with_context(self):
        with patch('app.crop_guide.build_crop_production_followup',return_value=('धान की किस्में: document evidence',['rice.pdf'])) as guide:
            for q in ['इसी फसल के लिए किस्म btaye?', 'इसकी किस्म बताएं', 'which varieties?']:
                result=build_coordinator(self.advisor()).answer(q,'पसंदीदा फसल: Rice |')
                self.assertEqual(result['agent_trace']['plan'],['agronomy'])
                self.assertEqual(result['references'],['rice.pdf'])
                self.assertEqual(guide.call_args.kwargs['crop_hint'],'Rice')

    def test_new_crop_overrides_rice_and_missing_crop_asks(self):
        with patch('app.crop_guide.build_crop_production_followup',return_value=('किस्में',['wheat.pdf'])) as guide:
            result=build_coordinator(self.advisor()).answer('गेहूं की किस्म बताएं','पसंदीदा फसल: Rice |')
            self.assertEqual(guide.call_args.kwargs['crop_hint'],'Wheat')
            result=build_coordinator(self.advisor()).answer('किस्म बताएं','')
            self.assertEqual(result['agent_status'],'needs_input')

    def test_supplemental_documents_searched_without_primary_guide(self):
        with patch('app.crop_guide._build_guide_points_for_crop',return_value=(None,[],[])), patch('app.crop_guide._build_supplemental_pdf_answer',return_value=('Rice varieties from document',['rice-varieties.pdf'])) as search:
            answer,refs=build_crop_production_followup('किस्म बताएं',crop_hint='Rice')
            self.assertIn('Rice varieties',answer)
            self.assertEqual(refs,['rice-varieties.pdf'])
            search.assert_called_once()
